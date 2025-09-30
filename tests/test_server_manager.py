import asyncio
from pathlib import Path

import pytest

from allin1.server import job_manager
from allin1.server.config import ProcessingMode, ServerSettings


@pytest.mark.asyncio
async def test_job_manager_caching(monkeypatch):
  async def dummy_validate(self, path: Path):
    return None

  monkeypatch.setattr(job_manager.AnalysisJobManager, '_validate_duration', dummy_validate, raising=False)

  class DummySeparator:
    sample_rate = 44100

    def __init__(self, *args, **kwargs):
      pass

    async def preload(self):
      return None

    async def separate_to_directory(self, input_path: Path, output_dir: Path):
      output_dir.mkdir(parents=True, exist_ok=True)
      mapping = {}
      for name in job_manager.STEM_ORDER:
        stem_path = output_dir / f'{name}.wav'
        stem_path.write_bytes(f'fake-{name}'.encode())
        mapping[name] = stem_path
      return mapping

  monkeypatch.setattr(job_manager, 'DemucsSeparator', DummySeparator)

  def fake_extract(demix_paths, spec_dir, multiprocess):
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / 'spec.npy'
    spec_path.write_bytes(b'spec')
    return [spec_path]

  monkeypatch.setattr(job_manager, 'extract_spectrograms', fake_extract)

  from allin1.typings import AnalysisResult, Segment

  def fake_run_inference(path, spec_path, model, device, include_activations, include_embeddings):
    return AnalysisResult(
      path=Path(path),
      bpm=120,
      beats=[0.0],
      downbeats=[0.0],
      beat_positions=[0],
      segments=[Segment(start=0.0, end=1.0, label='intro')],
      activations=None,
      embeddings=None,
    )

  monkeypatch.setattr(job_manager, 'run_inference', fake_run_inference)
  monkeypatch.setattr(job_manager, 'load_pretrained_model', lambda *args, **kwargs: object())

  settings = ServerSettings(processing_mode=ProcessingMode.SEQUENTIAL, cache_size=2, device='cpu')
  manager = job_manager.AnalysisJobManager(settings)
  await manager.startup()

  metadata = job_manager.JobMetadata(user_hash='track-a', filename='clip.wav')
  record_one = await manager.submit(b'audio-bytes', metadata)
  await _wait_for_status(manager, record_one.job_id, job_manager.JobStatus.COMPLETED)
  assert record_one.result is not None
  assert record_one.stems_path is not None

  record_cached = await manager.submit(b'audio-bytes', metadata)
  assert record_cached.status == job_manager.JobStatus.COMPLETED
  assert record_cached.result == record_one.result
  assert record_cached.stems_path is not None
  assert record_cached.stems_path.exists()
  assert record_cached.stems_path != record_one.stems_path

  cache_paths = [entry.stems_path for entry in manager._cache.values()]
  assert all(path.parent == manager._cache_dir for path in cache_paths)

  await manager.shutdown()
  assert not manager._cache_dir.exists()


@pytest.mark.asyncio
async def test_structure_inference_cpu_fallback(monkeypatch):
  async def dummy_validate(self, path: Path):
    return None

  monkeypatch.setattr(job_manager.AnalysisJobManager, '_validate_duration', dummy_validate, raising=False)

  def fake_select(self, configured):
    return configured or 'cpu'

  monkeypatch.setattr(job_manager.AnalysisJobManager, '_select_device', fake_select, raising=False)

  class DummySeparator:
    sample_rate = 44100

    def __init__(self, *args, **kwargs):
      pass

    async def preload(self):
      return None

    async def separate_to_directory(self, input_path: Path, output_dir: Path):
      output_dir.mkdir(parents=True, exist_ok=True)
      mapping = {}
      for name in job_manager.STEM_ORDER:
        stem_path = output_dir / f'{name}.wav'
        stem_path.write_bytes(f'fake-{name}'.encode())
        mapping[name] = stem_path
      return mapping

  monkeypatch.setattr(job_manager, 'DemucsSeparator', DummySeparator)

  def fake_extract(demix_paths, spec_dir, multiprocess):
    spec_dir.mkdir(parents=True, exist_ok=True)
    spec_path = spec_dir / 'spec.npy'
    spec_path.write_bytes(b'spec')
    return [spec_path]

  monkeypatch.setattr(job_manager, 'extract_spectrograms', fake_extract)

  from allin1.typings import AnalysisResult, Segment

  calls = {'count': 0, 'devices': []}

  def flaky_run_inference(path, spec_path, model, device, include_activations, include_embeddings):
    calls['count'] += 1
    calls['devices'].append(device)
    if calls['count'] == 1:
      raise RuntimeError('CUDA out of memory. Tried to allocate...')
    return AnalysisResult(
      path=Path(path),
      bpm=120,
      beats=[0.0],
      downbeats=[0.0],
      beat_positions=[0],
      segments=[Segment(start=0.0, end=1.0, label='intro')],
      activations=None,
      embeddings=None,
    )

  monkeypatch.setattr(job_manager, 'run_inference', flaky_run_inference)

  class DummyModel:
    def __init__(self):
      self._devices = []

    def to(self, device):
      self._devices.append(device)
      return self

  monkeypatch.setattr(job_manager, 'load_pretrained_model', lambda *args, **kwargs: DummyModel())

  settings = ServerSettings(processing_mode=ProcessingMode.SEQUENTIAL, cache_size=1, device='cuda')
  manager = job_manager.AnalysisJobManager(settings)
  await manager.startup()

  metadata = job_manager.JobMetadata(user_hash='track-b', filename='clip.wav')
  record = await manager.submit(b'audio-bytes', metadata)
  await _wait_for_status(manager, record.job_id, job_manager.JobStatus.COMPLETED)

  assert calls['count'] == 2
  assert calls['devices'] == ['cuda', 'cpu']
  assert manager.analysis_device == 'cpu'

  await manager.shutdown()


async def _wait_for_status(manager, job_id: str, status: job_manager.JobStatus, timeout: float = 2.0):
  deadline = asyncio.get_event_loop().time() + timeout
  while asyncio.get_event_loop().time() < deadline:
    record = await manager.get_job(job_id)
    if record and record.status == status:
      return record
    await asyncio.sleep(0.01)
  raise AssertionError(f'Job {job_id} did not reach status {status}')
