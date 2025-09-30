from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import shutil
import tempfile
import zipfile
import time
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple

import librosa
import torch

from ..spectrogram import extract_spectrograms
from ..helpers import run_inference
from ..models import load_pretrained_model
from ..typings import AnalysisResult
from .config import ProcessingMode, ServerSettings
from .demucs_runtime import DemucsSeparator

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
  created_at: float = field(default_factory=time.time)
  status: JobStatus = JobStatus.QUEUED
  error: Optional[str] = None
  stems_path: Optional[Path] = None
  sample_rate: Optional[int] = None
  result: Optional[Dict] = None


@dataclass
class CacheEntry:
  stems_path: Path
  sample_rate: int
  result: Dict
  created_at: float


logger = logging.getLogger(__name__)


class AnalysisJobManager:
  def __init__(self, settings: ServerSettings):
    self.settings = settings
    self.demucs_device = self._select_device(settings.demucs_device or settings.device)
    self.analysis_device = self._select_device(settings.analysis_device or settings.device)
    self.device = self.analysis_device
    self._queue: asyncio.Queue[Tuple[JobRecord, bytes]] = asyncio.Queue(maxsize=settings.max_pending_jobs)
    self._jobs: Dict[str, JobRecord] = {}
    self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
    self._cache_dir = Path(tempfile.mkdtemp(prefix='allin1_cache_'))
    self._dispatcher_task: Optional[asyncio.Task] = None
    self._active_tasks: set[asyncio.Task] = set()
    self._shutdown = False

    self._demucs = DemucsSeparator(settings.demucs_model, self.demucs_device)
    self._harmonix_model = None
    self._analysis_lock = asyncio.Lock()

  async def startup(self):
    await self._demucs.preload()
    self._harmonix_model = await asyncio.to_thread(
      load_pretrained_model,
      self.settings.harmonix_model,
      None,
      self.analysis_device,
    )
    self._dispatcher_task = asyncio.create_task(self._dispatcher())

  async def shutdown(self):
    self._shutdown = True
    if self._dispatcher_task:
      self._dispatcher_task.cancel()
      with contextlib.suppress(asyncio.CancelledError):
        await self._dispatcher_task
    for task in list(self._active_tasks):
      task.cancel()
    if self._active_tasks:
      await asyncio.gather(*self._active_tasks, return_exceptions=True)
    self._cleanup_temp_stems()

  async def submit(self, audio_bytes: bytes, metadata: JobMetadata) -> JobRecord:
    content_hash = hashlib.sha256(audio_bytes).hexdigest()

    for job in self._jobs.values():
      if job.content_hash == content_hash and job.status in {JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.PARTIAL}:
        return job

    cached = self._cache.get(content_hash)
    if cached and not cached.stems_path.exists():
      self._cache.pop(content_hash, None)
      cached = None
    if cached:
      job_id = str(uuid.uuid4())
      record = JobRecord(job_id=job_id, metadata=metadata, content_hash=content_hash)
      record.status = JobStatus.COMPLETED
      record.sample_rate = cached.sample_rate
      record.result = cached.result
      record.stems_path = self._materialize_stems_zip(record.job_id, cached.stems_path)
      self._jobs[job_id] = record
      self._touch_cache(content_hash)
      return record

    job_id = str(uuid.uuid4())
    record = JobRecord(job_id=job_id, metadata=metadata, content_hash=content_hash)
    self._jobs[job_id] = record

    await self._queue.put((record, audio_bytes))
    return record

  async def get_job(self, job_id: str) -> Optional[JobRecord]:
    return self._jobs.get(job_id)

  def list_jobs(self) -> Tuple[JobRecord, ...]:
    return tuple(self._jobs.values())

  def _select_device(self, configured: str | None) -> str:
    configured = configured or 'auto'
    lowered = configured.lower()
    if lowered == 'auto':
      return 'cuda' if torch.cuda.is_available() else 'cpu'

    if lowered.startswith('cuda') and not torch.cuda.is_available():
      raise RuntimeError('CUDA requested but no GPU is available. Start the server with --device cpu instead.')

    return configured

  async def _dispatcher(self):
    while not self._shutdown:
      record, audio_bytes = await self._queue.get()
      if self.settings.processing_mode == ProcessingMode.SEQUENTIAL:
        await self._process_job(record, audio_bytes)
      else:
        task = asyncio.create_task(self._process_with_semaphore(record, audio_bytes))
        self._active_tasks.add(task)
        task.add_done_callback(self._active_tasks.discard)

  async def _process_with_semaphore(self, record: JobRecord, audio_bytes: bytes):
    semaphore = getattr(self, '_semaphore', None)
    if semaphore is None:
      semaphore = asyncio.Semaphore(self.settings.max_parallel_tasks)
      self._semaphore = semaphore
    async with semaphore:
      await self._process_job(record, audio_bytes)

  async def _process_job(self, record: JobRecord, audio_bytes: bytes):
    record.status = JobStatus.PROCESSING
    try:
      async with AsyncTemporaryFile(suffix=f'_{record.job_id}_{record.metadata.filename}') as input_path:
        await input_path.write(audio_bytes)
        await input_path.flush()
        await self._validate_duration(input_path.path)
        await self._run_pipeline(record, input_path.path)
    except Exception as exc:  # pylint: disable=broad-except
      record.status = JobStatus.ERROR
      record.error = str(exc)

  async def _run_pipeline(self, record: JobRecord, input_path: Path):
    with tempfile.TemporaryDirectory(prefix=f'allin1_{record.job_id}_') as tmp_dir:
      tmp_dir_path = Path(tmp_dir)
      demix_dir = tmp_dir_path / 'demix'

      stems_paths = await self._demucs.separate_to_directory(input_path, demix_dir)
      stem_files = {}
      for name in STEM_ORDER:
        stem_path = stems_paths.get(name)
        if stem_path is None:
          for source_name, path in stems_paths.items():
            if _STEM_REMAP.get(source_name) == name:
              stem_path = path
              break
        if stem_path is None:
          raise RuntimeError(f'Missing stem {name} in Demucs output')
        stem_files[name] = stem_path

      temp_zip = tempfile.NamedTemporaryFile(prefix=f'allin1_{record.job_id}_', suffix='.zip', delete=False)
      temp_zip_path = Path(temp_zip.name)
      temp_zip.close()
      with zipfile.ZipFile(temp_zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
        for name, stem_path in stem_files.items():
          zip_file.write(stem_path, arcname=f'{name}.wav')

      record.stems_path = temp_zip_path
      record.sample_rate = self._demucs.sample_rate
      record.status = JobStatus.PARTIAL

      self._clear_device_cache(self.demucs_device)

      spec_dir = tmp_dir_path / 'spec'
      demix_paths = [stem_path.parent]
      spec_paths = await asyncio.to_thread(
        extract_spectrograms,
        demix_paths,
        spec_dir,
        False,
      )

      async with self._analysis_lock:
        result = await self._run_structure_with_retry(input_path, spec_paths[0])

      record.result = self._serialize_result(result)
      record.status = JobStatus.COMPLETED

      self._update_cache(record)

  async def _validate_duration(self, path: Path):
    duration = await asyncio.to_thread(librosa.get_duration, filename=str(path))
    if duration > self.settings.max_audio_duration_seconds:
      raise ValueError(
        f'Audio duration {duration:.2f}s exceeds limit of {self.settings.max_audio_duration_seconds}s'
      )

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

  def _update_cache(self, record: JobRecord):
    if self.settings.cache_size == 0 or record.result is None or record.stems_path is None:
      return
    cache_path = self._cache_dir / f'{record.content_hash}.zip'
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_cache_path = cache_path.with_name(cache_path.name + '.tmp')
    try:
      shutil.copyfile(record.stems_path, temp_cache_path)
      temp_cache_path.replace(cache_path)
    except OSError:
      temp_cache_path.unlink(missing_ok=True)
      return

    existing = self._cache.pop(record.content_hash, None)
    if existing and existing.stems_path != cache_path:
      existing.stems_path.unlink(missing_ok=True)

    self._cache[record.content_hash] = CacheEntry(
      stems_path=cache_path,
      sample_rate=record.sample_rate,
      result=record.result,
      created_at=time.time(),
    )
    while len(self._cache) > self.settings.cache_size:
      _, evicted = self._cache.popitem(last=False)
      evicted.stems_path.unlink(missing_ok=True)

  def _touch_cache(self, key: str):
    if key in self._cache:
      entry = self._cache.pop(key)
      self._cache[key] = entry

  def _materialize_stems_zip(self, job_id: str, source: Path) -> Path:
    temp_file = tempfile.NamedTemporaryFile(prefix=f'allin1_{job_id}_', suffix='.zip', delete=False)
    temp_file.close()
    shutil.copyfile(source, temp_file.name)
    return Path(temp_file.name)

  def _cleanup_temp_stems(self):
    for record in self._jobs.values():
      if record.stems_path:
        record.stems_path.unlink(missing_ok=True)
    shutil.rmtree(self._cache_dir, ignore_errors=True)

  async def _run_structure_with_retry(self, input_path: Path, spec_path: Path) -> AnalysisResult:
    try:
      return await self._run_structure_once(input_path, spec_path)
    except RuntimeError as exc:
      if self._should_retry_on_cpu(exc):
        previous_device = self.analysis_device
        logger.warning('Structure inference failed on %s due to OOM; retrying on CPU.', previous_device)
        await self._move_harmonix_model('cpu')
        self.analysis_device = 'cpu'
        self._clear_device_cache(previous_device)
        return await self._run_structure_once(input_path, spec_path)
      raise

  async def _run_structure_once(self, input_path: Path, spec_path: Path) -> AnalysisResult:
    result = await asyncio.to_thread(
      run_inference,
      input_path,
      spec_path,
      self._harmonix_model,
      self.analysis_device,
      self.settings.include_activations,
      self.settings.include_embeddings,
    )
    self._clear_device_cache(self.analysis_device)
    return result

  def _should_retry_on_cpu(self, exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return (
      self._is_gpu_device(self.analysis_device)
      and ('out of memory' in message or 'oom' in message)
    )

  async def _move_harmonix_model(self, device: str):
    def _move():
      if hasattr(self._harmonix_model, 'to'):
        self._harmonix_model.to(device)
      sub_models = getattr(self._harmonix_model, 'models', None)
      if sub_models:
        for model in sub_models:
          if hasattr(model, 'to'):
            model.to(device)

    await asyncio.to_thread(_move)

  def _clear_device_cache(self, device: str):
    if not self._is_gpu_device(device):
      return
    if torch.cuda.is_available():
      with contextlib.suppress(Exception):
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

  @staticmethod
  def _is_gpu_device(device: str) -> bool:
    try:
      return torch.device(device).type != 'cpu'
    except (TypeError, RuntimeError):
      return False


class AsyncTemporaryFile:
  def __init__(self, suffix: str):
    self._file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    self.path = Path(self._file.name)

  async def __aenter__(self):
    return self

  async def __aexit__(self, exc_type, exc, tb):
    self._file.close()
    self.path.unlink(missing_ok=True)

  async def write(self, data: bytes):
    await asyncio.to_thread(self._file.write, data)

  async def flush(self):
    await asyncio.to_thread(self._file.flush)
