from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import shutil
import uuid
import zipfile
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import librosa
import numpy as np
import torch

from ..helpers import compute_activations
from ..models import load_pretrained_model
from ..postprocessing import (
	estimate_tempo_from_beats,
	postprocess_functional_structure,
	postprocess_metrical_structure,
)
from ..spectrogram import extract_spectrograms
from ..typings import AllInOneOutput, AnalysisResult
from .config import ServerSettings, StorageEvictionMode
from .demucs_runtime import DemucsSeparator
from .storage import (
	StoredJobMetadata,
	StoredJobState,
	calculate_storage_usage,
	ensure_input_directory,
	ensure_storage_root,
	ensure_struct_directory,
	ensure_stems_directory,
	input_path,
	job_root,
	list_existing_jobs,
	stems_dir,
	stems_zip_path,
	struct_path,
	write_metadata,
)

STEM_ORDER = ['bass', 'drums', 'other', 'vocals']
_STEM_REMAP = {
	'guitar': 'other',
	'piano': 'other',
	'vocals': 'vocals',
	'drums': 'drums',
	'bass': 'bass',
	'other': 'other',
}


class JobStatus(str, Enum):
	QUEUED = 'queued'
	PROCESSING = 'processing'
	PARTIAL = 'partial'
	COMPLETED = 'completed'
	ERROR = 'error'
	CANCELLED = 'cancelled'


@dataclass
class JobMetadata:
	user_hash: str
	filename: str
	segment_start: Optional[float] = None
	segment_end: Optional[float] = None


@dataclass
class JobRecord:
	job_id: str
	metadata: JobMetadata
	content_hash: str
	stored: Optional[StoredJobMetadata] = None
	sample_rate: Optional[int] = None
	stems_path: Optional[Path] = None
	result: Optional[Dict] = None

	def __post_init__(self):
		if self.stored is None:
			self.stored = StoredJobMetadata(
				job_id=self.job_id,
				user_hash=self.metadata.user_hash,
				filename=self.metadata.filename,
				content_hash=self.content_hash,
				segment_start=self.metadata.segment_start,
				segment_end=self.metadata.segment_end,
			)

	@property
	def status(self) -> JobStatus:
		return _STATE_TO_STATUS.get(self.stored.state, JobStatus.ERROR)

	@status.setter
	def status(self, status: JobStatus):
		state = _STATUS_TO_STATE.get(status)
		if state is None:
			raise ValueError(f'Unsupported job status: {status!r}')
		self.stored.mark_state(state)

	@property
	def error(self) -> Optional[str]:
		return self.stored.error


_STATE_TO_STATUS = {
	StoredJobState.QUEUED: JobStatus.QUEUED,
	StoredJobState.STEMS_PENDING: JobStatus.QUEUED,
	StoredJobState.STEMS_RUNNING: JobStatus.PROCESSING,
	StoredJobState.STRUCT_PENDING: JobStatus.PARTIAL,
	StoredJobState.STRUCT_RUNNING: JobStatus.PROCESSING,
	StoredJobState.COMPLETED: JobStatus.COMPLETED,
	StoredJobState.ERROR: JobStatus.ERROR,
	StoredJobState.CANCELLED: JobStatus.CANCELLED,
}

_STATUS_TO_STATE = {status: state for state, status in _STATE_TO_STATUS.items()}


logger = logging.getLogger(__name__)


class AnalysisJobManager:
	def __init__(self, settings: ServerSettings):
		self.settings = settings
		self.demucs_device = self._select_device(settings.demucs_device or settings.device)
		self.analysis_device = self._select_device(settings.analysis_device or settings.device)
		self._configured_analysis_device = self.analysis_device
		self._storage_root = ensure_storage_root(Path(settings.storage_root).expanduser())
		self._storage_usage_bytes: int = 0

		self._stem_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.max_pending_jobs)
		self._structure_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=settings.max_pending_jobs)
		self._jobs: Dict[str, JobRecord] = {}
		self._content_index: Dict[str, str] = {}
		self._pending_stem: set[str] = set()
		self._pending_structure: set[str] = set()
		self._stem_pause_event = asyncio.Event()
		self._stem_pause_event.set()
		self._structure_mode_event = asyncio.Event()
		self._structure_mode_active = False

		self._stem_worker_task: Optional[asyncio.Task] = None
		self._structure_worker_task: Optional[asyncio.Task] = None
		self._shutdown = False

		self._demucs = DemucsSeparator(settings.demucs_model, self.demucs_device)
		self._harmonix_model = None
		self._structure_lock = asyncio.Lock()

	async def startup(self):
		await self._restore_jobs()
		await self._refresh_storage_usage()
		if self.settings.preload_demucs:
			await self._demucs.preload()
		await self._maybe_begin_structure_mode()

		self._stem_worker_task = asyncio.create_task(self._stem_worker_loop())
		self._structure_worker_task = asyncio.create_task(self._structure_worker_loop())

	async def shutdown(self):
		self._shutdown = True
		for task in [self._stem_worker_task, self._structure_worker_task]:
			if task is not None:
				task.cancel()
		for task in [self._stem_worker_task, self._structure_worker_task]:
			if task is not None:
				with contextlib.suppress(asyncio.CancelledError):
					await task

		self._pending_stem.clear()
		self._pending_structure.clear()
		self._jobs.clear()
		self._content_index.clear()

		self._demucs.release()
		await self._release_harmonix_model()

	async def submit(self, audio_bytes: bytes, metadata: JobMetadata) -> JobRecord:
		content_hash = hashlib.sha256(audio_bytes).hexdigest()

		existing_id = self._content_index.get(content_hash)
		if existing_id:
			record = self._jobs[existing_id]
			if record.result is None and record.stored.has_structure:
				record.result = await self._load_result_from_disk(record)
			if record.stems_path is None and record.stored.has_stems:
				potential_path = stems_zip_path(self._storage_root, record.content_hash)
				if potential_path.exists():
					record.stems_path = potential_path
			return record

		job_id = str(uuid.uuid4())
		stored = StoredJobMetadata(
			job_id=job_id,
			user_hash=metadata.user_hash,
			filename=metadata.filename,
			content_hash=content_hash,
			segment_start=metadata.segment_start,
			segment_end=metadata.segment_end,
		)

		record = JobRecord(job_id=job_id, metadata=metadata, content_hash=content_hash, stored=stored)
		self._jobs[job_id] = record
		self._content_index[content_hash] = job_id

		await self._ensure_storage_capacity(len(audio_bytes))
		input_file = await self._store_audio(record, audio_bytes)
		await self._validate_duration(input_file)
		await self._refresh_storage_usage()

		await self._persist_metadata(record.stored)
		await self._enqueue_stem_job(record)

		return record

	async def get_job(self, job_id: str) -> Optional[JobRecord]:
		return self._jobs.get(job_id)

	def list_jobs(self) -> Tuple[JobRecord, ...]:
		return tuple(self._jobs.values())

	def _select_device(self, configured: str | None) -> str:
		configured = (configured or 'auto').lower()
		if configured == 'auto':
			return 'cuda' if torch.cuda.is_available() else 'cpu'
		if configured.startswith('cuda') and not torch.cuda.is_available():
			raise RuntimeError('CUDA requested but no GPU is available. Start the server with --device cpu instead.')
		return configured

	async def _restore_jobs(self):
		for stored in list_existing_jobs(self._storage_root):
			metadata = JobMetadata(
				user_hash=stored.user_hash,
				filename=stored.filename,
				segment_start=stored.segment_start,
				segment_end=stored.segment_end,
			)
			record = JobRecord(
				job_id=stored.job_id,
				metadata=metadata,
				content_hash=stored.content_hash,
				stored=stored,
				sample_rate=stored.sample_rate,
			)
			zip_path = stems_zip_path(self._storage_root, stored.content_hash)
			if stored.has_stems and zip_path.exists():
				record.stems_path = zip_path
			if stored.has_structure:
				record.result = await self._load_result_from_disk(record)

			self._jobs[record.job_id] = record
			self._content_index[record.content_hash] = record.job_id

			if stored.state in {StoredJobState.QUEUED, StoredJobState.STEMS_PENDING, StoredJobState.STEMS_RUNNING}:
				stored.mark_state(StoredJobState.STEMS_PENDING)
				await self._persist_metadata(stored)
				await self._enqueue_stem_job(record)
			elif stored.state in {StoredJobState.STRUCT_PENDING, StoredJobState.STRUCT_RUNNING}:
				stored.mark_state(StoredJobState.STRUCT_PENDING)
				await self._persist_metadata(stored)
				await self._enqueue_structure_job(record)

	async def _store_audio(self, record: JobRecord, audio_bytes: bytes) -> Path:
		directory = ensure_input_directory(self._storage_root, record.content_hash)
		target_path = directory / 'original.wav'

		if target_path.exists():
			return target_path

		def _write():
			target_path.write_bytes(audio_bytes)

		await asyncio.to_thread(_write)
		return target_path

	async def _enqueue_stem_job(self, record: JobRecord):
		if record.job_id in self._pending_stem:
			return
		record.stored.mark_state(StoredJobState.STEMS_PENDING)
		await self._persist_metadata(record.stored)
		await self._stem_queue.put(record.job_id)
		self._pending_stem.add(record.job_id)

	async def _enqueue_structure_job(self, record: JobRecord):
		if record.job_id in self._pending_structure:
			return
		record.stored.mark_state(StoredJobState.STRUCT_PENDING)
		await self._persist_metadata(record.stored)
		await self._structure_queue.put(record.job_id)
		self._pending_structure.add(record.job_id)

	async def _stem_worker_loop(self):
		try:
			while True:
				await self._stem_pause_event.wait()
				try:
					job_id = await asyncio.wait_for(self._stem_queue.get(), timeout=0.5)
				except asyncio.TimeoutError:
					continue
				batch = [job_id]
				while len(batch) < self.settings.demucs_batch_size:
					try:
						next_id = self._stem_queue.get_nowait()
					except asyncio.QueueEmpty:
						break
					else:
						batch.append(next_id)
				try:
					await self._process_stem_batch(batch)
				finally:
					for job_id in batch:
						with contextlib.suppress(KeyError):
							self._pending_stem.remove(job_id)
						self._stem_queue.task_done()
				await self._maybe_begin_structure_mode()
		except asyncio.CancelledError:
			return

	async def _structure_worker_loop(self):
		try:
			while True:
				await self._structure_mode_event.wait()
				try:
					job_id = await asyncio.wait_for(self._structure_queue.get(), timeout=0.5)
				except asyncio.TimeoutError:
					await self._maybe_end_structure_mode()
					continue
				batch = [job_id]
				while len(batch) < self.settings.structure_batch_size:
					try:
						next_id = self._structure_queue.get_nowait()
					except asyncio.QueueEmpty:
						break
					else:
						batch.append(next_id)
				try:
					await self._process_structure_batch(batch)
				finally:
					for job_id in batch:
						with contextlib.suppress(KeyError):
							self._pending_structure.remove(job_id)
						self._structure_queue.task_done()
				await self._maybe_end_structure_mode()
		except asyncio.CancelledError:
			return

	async def _process_stem_batch(self, job_ids: Iterable[str]):
		records = [self._jobs.get(job_id) for job_id in job_ids]
		records = [
			record
			for record in records
			if record is not None and record.stored.state != StoredJobState.CANCELLED
		]
		if not records:
			return

		await self._demucs.preload()

		for record in records:
			record.stored.mark_state(StoredJobState.STEMS_RUNNING)
			await self._persist_metadata(record.stored)

		try:
			for record in records:
				try:
					await self._prepare_stems(record)
					await self._enqueue_structure_job(record)
				except Exception as exc:  # pylint: disable=broad-except
					logger.exception('Demucs separation failed for job %s', record.job_id)
					record.stored.mark_state(StoredJobState.ERROR, error=str(exc))
					await self._persist_metadata(record.stored)
		finally:
			self._clear_device_cache(self.demucs_device)
			if self.settings.demucs_release_after_batch:
				self._demucs.release()
		await self._enforce_storage_limit()

	async def _prepare_stems(self, record: JobRecord):
		input_audio = input_path(self._storage_root, record.content_hash)
		if not input_audio.exists():
			raise FileNotFoundError(f'Input audio not found for job {record.job_id}')

		destination = ensure_stems_directory(self._storage_root, record.content_hash)

		def _clean_directory():
			for child in destination.iterdir():
				if child.is_file():
					child.unlink(missing_ok=True)

		await asyncio.to_thread(_clean_directory)

		stems_paths = await self._demucs.separate_to_directory(input_audio, destination)

		resolved = {}
		for name in STEM_ORDER:
			stem_path = stems_paths.get(name)
			if stem_path is None:
				for source_name, candidate in stems_paths.items():
					if _STEM_REMAP.get(source_name) == name:
						stem_path = candidate
						break
			if stem_path is None:
				raise RuntimeError(f'Missing stem {name} in Demucs output')
			resolved[name] = stem_path

		await self._write_stems_zip(record, resolved.values())

		record.sample_rate = self._demucs.sample_rate
		record.stored.mark_stems_ready(record.sample_rate)
		await self._persist_metadata(record.stored)

	async def _process_structure_batch(self, job_ids: Iterable[str]):
		records = [self._jobs.get(job_id) for job_id in job_ids]
		records = [
			record
			for record in records
			if record is not None and record.stored.state not in {StoredJobState.CANCELLED, StoredJobState.ERROR}
		]
		if not records:
			return

		await self._maybe_begin_structure_mode(force=True)
		await self._ensure_harmonix_model()

		for record in records:
			record.stored.mark_state(StoredJobState.STRUCT_RUNNING)
			await self._persist_metadata(record.stored)

		try:
			for record in records:
				if record.stored.state == StoredJobState.ERROR:
					continue
				try:
					result = await self._run_structure(record)
					record.result = self._serialize_result(result)
					await self._write_structure_result(record)
					record.stored.mark_structure_ready()
					record.stored.mark_state(StoredJobState.COMPLETED)
					await self._persist_metadata(record.stored)
				except Exception as exc:  # pylint: disable=broad-except
					logger.exception('Structure analysis failed for job %s', record.job_id)
					record.stored.mark_state(StoredJobState.ERROR, error=str(exc))
					await self._persist_metadata(record.stored)
		finally:
			self._clear_device_cache(self.analysis_device)
		await self._enforce_storage_limit()

	async def _maybe_begin_structure_mode(self, force: bool = False):
		if self._structure_mode_active:
			return
		ready = self._structure_ready_count()
		if ready == 0 and not force:
			return
		if force or ready >= self.settings.structure_prefetch_threshold or not self._stem_backlog():
			await self._begin_structure_mode()

	async def _begin_structure_mode(self):
		if self._structure_mode_active:
			return
		self._structure_mode_active = True
		self._stem_pause_event.clear()
		while self._pending_stem:
			await asyncio.sleep(0.05)
		is_loaded = getattr(self._demucs, 'is_loaded', None)
		try:
			loaded = is_loaded() if callable(is_loaded) else True
		except Exception:  # pylint: disable=broad-except
			loaded = True
		if loaded:
			with contextlib.suppress(Exception):
				self._demucs.release()
		self._structure_mode_event.set()

	async def _maybe_end_structure_mode(self):
		if not self._structure_mode_active:
			return
		if not self._structure_queue.empty() or self._pending_structure or self._structure_ready_count() > 0:
			return
		await self._move_harmonix_model('cpu')
		self._structure_mode_active = False
		self._structure_mode_event.clear()
		if self.settings.preload_demucs:
			await self._demucs.preload()
		else:
			self._demucs.release()
		self._stem_pause_event.set()

	def _structure_ready_count(self) -> int:
		return sum(
			1
			for record in self._jobs.values()
			if record.stored.state == StoredJobState.STRUCT_PENDING
		)

	def _stem_backlog(self) -> bool:
		return (not self._stem_queue.empty()) or bool(self._pending_stem)

	async def _run_structure(self, record: JobRecord) -> AnalysisResult:
		audio_path = input_path(self._storage_root, record.content_hash)
		if not audio_path.exists():
			raise FileNotFoundError(f'Input audio missing for structure analysis of job {record.job_id}')

		demix_dir = stems_dir(self._storage_root, record.content_hash)
		if not demix_dir.exists():
			raise FileNotFoundError(f'Stem directory missing for job {record.job_id}')

		spec_dir = job_root(self._storage_root, record.content_hash) / 'spec'

		def _prepare_spec_dir():
			spec_dir.mkdir(parents=True, exist_ok=True)
			for child in spec_dir.iterdir():
				if child.is_file():
					child.unlink(missing_ok=True)

		await asyncio.to_thread(_prepare_spec_dir)

		spec_paths = await asyncio.to_thread(
			extract_spectrograms,
			[demix_dir],
			spec_dir,
			False,
		)

		async with self._structure_lock:
			return await self._run_structure_with_retry(audio_path, spec_paths[0])

	async def _persist_metadata(self, metadata: StoredJobMetadata):
		await asyncio.to_thread(write_metadata, self._storage_root, metadata)

	async def _write_stems_zip(self, record: JobRecord, stem_paths: Iterable[Path]):
		zip_path = stems_zip_path(self._storage_root, record.content_hash)
		zip_path.parent.mkdir(parents=True, exist_ok=True)
		temp_path = zip_path.with_suffix('.tmp.zip')

		def _zip():
			with zipfile.ZipFile(temp_path, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
				for path in stem_paths:
					archive.write(path, arcname=Path(path).name)
			temp_path.replace(zip_path)

		await asyncio.to_thread(_zip)
		record.stems_path = zip_path
		await self._refresh_storage_usage()

	async def ensure_stems_archive(self, record: JobRecord) -> Optional[Path]:
		if not record.stored.has_stems:
			return None
		if record.stems_path and record.stems_path.exists():
			return record.stems_path
		directory = stems_dir(self._storage_root, record.content_hash)
		if not directory.exists():
			return None
		stem_files = [path for path in directory.iterdir() if path.is_file()]
		if not stem_files:
			return None
		await self._write_stems_zip(record, stem_files)
		return record.stems_path if record.stems_path and record.stems_path.exists() else None

	async def _write_structure_result(self, record: JobRecord):
		if record.result is None:
			return
		struct_dir = ensure_struct_directory(self._storage_root, record.content_hash)
		target = struct_path(self._storage_root, record.content_hash)
		temp_path = target.with_suffix('.tmp')

		def _write():
			with temp_path.open('w', encoding='utf-8') as fh:
				json.dump(record.result, fh, indent=2)
			temp_path.replace(target)

		await asyncio.to_thread(_write)
		await self._refresh_storage_usage()

	async def _load_result_from_disk(self, record: JobRecord) -> Optional[Dict]:
		target = struct_path(self._storage_root, record.content_hash)
		if not target.exists():
			return None

		def _read():
			with target.open('r', encoding='utf-8') as fh:
				return json.load(fh)

		return await asyncio.to_thread(_read)

	async def _ensure_harmonix_model(self):
		if self._harmonix_model is not None:
			return
		self._harmonix_model = await asyncio.to_thread(
			load_pretrained_model,
			self.settings.harmonix_model,
			None,
			'cpu',
		)

	async def load_structure_result(self, record: JobRecord) -> Optional[Dict]:
		if record.result is not None:
			return record.result
		if not record.stored.has_structure:
			return None
		record.result = await self._load_result_from_disk(record)
		return record.result

	async def _release_harmonix_model(self):
		if self._harmonix_model is None:
			return

		def _cleanup():
			model = self._harmonix_model
			if hasattr(model, 'to'):
				model.to('cpu')

		await asyncio.to_thread(_cleanup)
		self._harmonix_model = None
		if torch.cuda.is_available():
			with contextlib.suppress(Exception):
				torch.cuda.empty_cache()
				torch.cuda.ipc_collect()
	async def _refresh_storage_usage(self) -> int:
		usage = await asyncio.to_thread(calculate_storage_usage, self._storage_root)
		self._storage_usage_bytes = usage
		return usage

	async def _ensure_storage_capacity(self, additional_bytes: int):
		if self.settings.max_storage_bytes is None:
			return
		watermark = self.settings.storage_soft_watermark_bytes or self.settings.max_storage_bytes
		usage = await self._refresh_storage_usage()
		projected = usage + max(0, additional_bytes)
		if projected <= watermark:
			return
		if self.settings.storage_eviction_mode == StorageEvictionMode.REFUSE:
			raise RuntimeError('Storage watermark exceeded; refusing new job submission.')
		await self._evict_jobs(projected - watermark)
		usage = await self._refresh_storage_usage()
		if usage + max(0, additional_bytes) > self.settings.max_storage_bytes:
			raise RuntimeError('Storage limit exceeded; unable to free sufficient space.')

	async def _enforce_storage_limit(self):
		if self.settings.max_storage_bytes is None:
			return
		watermark = self.settings.storage_soft_watermark_bytes or self.settings.max_storage_bytes
		usage = await self._refresh_storage_usage()
		if usage <= watermark:
			return
		if self.settings.storage_eviction_mode == StorageEvictionMode.REFUSE:
			return
		await self._evict_jobs(usage - watermark)

	async def _evict_jobs(self, required_bytes: int) -> int:
		if required_bytes <= 0:
			return 0
		freed = 0
		candidates = sorted(self._jobs.values(), key=lambda record: record.stored.created_at)
		for record in candidates:
			if record.stored.state in {StoredJobState.STEMS_RUNNING, StoredJobState.STRUCT_RUNNING}:
				continue
			freed += await self._evict_single_job(record)
			if freed >= required_bytes:
				break
		return freed

	async def _evict_single_job(self, record: JobRecord) -> int:
		before = await self._job_storage_usage(record)
		record.stored.mark_state(StoredJobState.CANCELLED, error='Cancelled due to storage pressure.')
		record.stored.has_stems = False
		record.stored.has_structure = False
		record.stored.sample_rate = None
		record.result = None
		record.stems_path = None
		self._pending_stem.discard(record.job_id)
		self._pending_structure.discard(record.job_id)
		self._content_index.pop(record.content_hash, None)
		await self._purge_job_artifacts(record)
		await self._persist_metadata(record.stored)
		await self._refresh_storage_usage()
		after = await self._job_storage_usage(record)
		return max(0, before - after)

	async def _purge_job_artifacts(self, record: JobRecord):
		root = job_root(self._storage_root, record.content_hash)

		def _purge():
			if not root.exists():
				return
			for child in root.iterdir():
				if child.name == 'metadata.json':
					continue
				if child.is_dir():
					shutil.rmtree(child, ignore_errors=True)
				else:
					child.unlink(missing_ok=True)

		await asyncio.to_thread(_purge)

	async def _job_storage_usage(self, record: JobRecord) -> int:
		root = job_root(self._storage_root, record.content_hash)

		def _calc() -> int:
			if not root.exists():
				return 0
			total = 0
			for dirpath, _dirnames, filenames in os.walk(root):
				for filename in filenames:
					path = Path(dirpath) / filename
					with contextlib.suppress(OSError):
						total += path.stat().st_size
			return total

		return await asyncio.to_thread(_calc)

	def get_metrics(self) -> Dict[str, object]:
		state_counts: Dict[str, int] = {state.value: 0 for state in StoredJobState}
		for record in self._jobs.values():
			state_counts[record.stored.state.value] += 1

		return {
			'stem_queue_size': self._stem_queue.qsize(),
			'structure_queue_size': self._structure_queue.qsize(),
			'pending_stem_jobs': len(self._pending_stem),
			'pending_structure_jobs': len(self._pending_structure),
			'storage_usage_bytes': self._storage_usage_bytes,
			'storage_max_bytes': self.settings.max_storage_bytes,
			'storage_watermark_bytes': self.settings.storage_soft_watermark_bytes or self.settings.max_storage_bytes,
			'jobs_by_state': state_counts,
			'demucs_loaded': self._demucs.is_loaded(),
			'harmonix_loaded': self._harmonix_model is not None,
			'demucs_device': self.demucs_device,
			'analysis_device': self.analysis_device,
		}
		self._harmonix_model = None

	async def _run_structure_with_retry(self, input_path: Path, spec_path: Path) -> AnalysisResult:
		self.analysis_device = self._configured_analysis_device
		try:
			result = await self._run_structure_once(input_path, spec_path)
			self.analysis_device = self._configured_analysis_device
			return result
		except RuntimeError as exc:
			if self._should_retry_on_cpu(exc):
				previous_device = self.analysis_device
				logger.warning('Structure inference failed on %s due to OOM; retrying on CPU.', previous_device)
				await self._move_harmonix_model('cpu')
				self.analysis_device = 'cpu'
				self._clear_device_cache(previous_device)
				result = await self._run_structure_once(input_path, spec_path)
				self.analysis_device = self._configured_analysis_device
				return result
			raise

	async def _run_structure_once(self, input_path: Path, spec_path: Path) -> AnalysisResult:
		return await asyncio.to_thread(
			self._structure_inference_sync,
			input_path,
			spec_path,
		)

	def _structure_inference_sync(self, input_path: Path, spec_path: Path) -> AnalysisResult:
		spec = np.load(spec_path)
		spec_tensor = torch.from_numpy(spec).unsqueeze(0).float()
		logits = self._forward_harmonix(spec_tensor)
		cfg = getattr(self._harmonix_model, 'cfg', None)
		if cfg is None:
			raise RuntimeError('Harmonix model is missing configuration metadata.')
		metrical = postprocess_metrical_structure(logits, cfg)
		functional = postprocess_functional_structure(logits, cfg)
		bpm = estimate_tempo_from_beats(metrical['beats'])
		result = AnalysisResult(
			path=input_path,
			bpm=bpm,
			segments=functional,
			**metrical,
		)
		if self.settings.include_activations:
			result.activations = compute_activations(logits)
		if self.settings.include_embeddings:
			result.embeddings = logits.embeddings[0].detach().cpu().numpy()
		return result

	def _forward_harmonix(self, spec_tensor: torch.Tensor) -> AllInOneOutput:
		desired_device = self.analysis_device
		actual_device = desired_device if self._is_gpu_device(desired_device) and torch.cuda.is_available() else 'cpu'
		self.analysis_device = actual_device
		spec_on_device = spec_tensor.to(actual_device)
		models = getattr(self._harmonix_model, 'models', None)
		try:
			if models:
				output = self._forward_ensemble(spec_on_device, actual_device, models)
			else:
				output = self._forward_single_model(spec_on_device, actual_device, self._harmonix_model)
		finally:
			if actual_device != 'cpu':
				del spec_on_device
				self._clear_device_cache(actual_device)
		return output

	def _forward_single_model(self, spec_tensor: torch.Tensor, device: str, model) -> AllInOneOutput:
		use_gpu = self._is_gpu_device(device) and torch.cuda.is_available()
		try:
			with torch.no_grad():
				if use_gpu:
					model.to(device)
				output = model(spec_tensor)
		except RuntimeError:
			if use_gpu:
				model.to('cpu')
				self._clear_device_cache(device)
			raise
		output_cpu = self._detach_output(output)
		if use_gpu:
			model.to('cpu')
			self._clear_device_cache(device)
		return output_cpu

	def _forward_ensemble(self, spec_tensor: torch.Tensor, device: str, models: Iterable[torch.nn.Module]) -> AllInOneOutput:
		use_gpu = self._is_gpu_device(device) and torch.cuda.is_available()
		sum_logits_beat = None
		sum_logits_downbeat = None
		sum_logits_section = None
		sum_logits_function = None
		embedding_tensors = []
		count = 0
		for sub_model in models:
			try:
				with torch.no_grad():
					if use_gpu:
						sub_model.to(device)
					output = sub_model(spec_tensor)
			except RuntimeError:
				if use_gpu:
					sub_model.to('cpu')
					self._clear_device_cache(device)
				raise
			output_cpu = self._detach_output(output)
			if sum_logits_beat is None:
				sum_logits_beat = output_cpu.logits_beat.clone()
				sum_logits_downbeat = output_cpu.logits_downbeat.clone()
				sum_logits_section = output_cpu.logits_section.clone()
				sum_logits_function = output_cpu.logits_function.clone()
			else:
				sum_logits_beat += output_cpu.logits_beat
				sum_logits_downbeat += output_cpu.logits_downbeat
				sum_logits_section += output_cpu.logits_section
				sum_logits_function += output_cpu.logits_function
			embedding_tensors.append(output_cpu.embeddings.unsqueeze(-1))
			count += 1
			if use_gpu:
				sub_model.to('cpu')
				self._clear_device_cache(device)
		if count == 0:
			raise RuntimeError('Harmonix ensemble is empty; cannot perform inference.')
		embeddings = torch.cat(embedding_tensors, dim=-1)
		return AllInOneOutput(
			logits_beat=sum_logits_beat / count,
			logits_downbeat=sum_logits_downbeat / count,
			logits_section=sum_logits_section / count,
			logits_function=sum_logits_function / count,
			embeddings=embeddings,
		)

	@staticmethod
	def _detach_output(output: AllInOneOutput) -> AllInOneOutput:
		return AllInOneOutput(
			logits_beat=output.logits_beat.detach().cpu(),
			logits_downbeat=output.logits_downbeat.detach().cpu(),
			logits_section=output.logits_section.detach().cpu(),
			logits_function=output.logits_function.detach().cpu(),
			embeddings=output.embeddings.detach().cpu(),
		)

	def _should_retry_on_cpu(self, exc: RuntimeError) -> bool:
		message = str(exc).lower()
		return self._is_gpu_device(self.analysis_device) and ('out of memory' in message or 'oom' in message)

	async def _move_harmonix_model(self, device: str):
		if self._harmonix_model is None:
			return

		def _move():
			model = self._harmonix_model
			if hasattr(model, 'to'):
				model.to(device)
			sub_models = getattr(model, 'models', None)
			if sub_models:
				for sub_model in sub_models:
					if hasattr(sub_model, 'to'):
						sub_model.to(device)

		await asyncio.to_thread(_move)

	def _clear_device_cache(self, device: str):
		if not self._is_gpu_device(device):
			return
		if torch.cuda.is_available():
			with contextlib.suppress(Exception):
				torch.cuda.empty_cache()
				torch.cuda.ipc_collect()

	def _serialize_result(self, result: AnalysisResult) -> Dict:
		data = asdict(result)
		data['path'] = str(result.path)
		if data.get('activations') is not None:
			data['activations'] = {key: value.tolist() for key, value in data['activations'].items()}
		if data.get('embeddings') is not None:
			data['embeddings'] = data['embeddings'].tolist()
		data['segments'] = [
			{**segment, 'start': float(segment['start']), 'end': float(segment['end'])}
			for segment in data['segments']
		]
		return data

	async def _validate_duration(self, path: Path):
		duration = await asyncio.to_thread(librosa.get_duration, filename=str(path))
		if duration > self.settings.max_audio_duration_seconds:
			raise ValueError(
				f'Audio duration {duration:.2f}s exceeds limit of {self.settings.max_audio_duration_seconds}s'
			)

	@staticmethod
	def _is_gpu_device(device: str) -> bool:
		try:
			return torch.device(device).type != 'cpu'
		except (TypeError, RuntimeError):
			return False
