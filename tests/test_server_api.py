import tempfile
from pathlib import Path
import zipfile

import pytest
from fastapi.testclient import TestClient

from allin1.server import app as app_module
from allin1.server import job_manager
from allin1.server.config import ServerSettings


class FakeManager:
  def __init__(self, settings):
    self.settings = settings
    self.jobs: dict[str, job_manager.JobRecord] = {}
    self._counter = 0

  async def startup(self):
    return None

  async def shutdown(self):
    return None

  def list_jobs(self):
    return tuple(self.jobs.values())

  async def get_job(self, job_id: str):
    return self.jobs.get(job_id)

  async def submit(self, audio_bytes: bytes, metadata: job_manager.JobMetadata):
    self._counter += 1
    job_id = f'job-{self._counter}'
    record = job_manager.JobRecord(job_id=job_id, metadata=metadata, content_hash=f'hash-{self._counter}')
    record.status = job_manager.JobStatus.COMPLETED
    record.sample_rate = 44100
    temp_zip = tempfile.NamedTemporaryFile(prefix=f'test_job_{job_id}_', suffix='.zip', delete=False)
    temp_zip_path = Path(temp_zip.name)
    temp_zip.close()
    with zipfile.ZipFile(temp_zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as zip_file:
      zip_file.writestr('bass.wav', b'bass-bytes')
      zip_file.writestr('drums.wav', b'drums-bytes')
    record.stems_path = temp_zip_path
    record.stored.mark_stems_ready(sample_rate=44100)
    record.stored.mark_structure_ready()
    record.result = {'bpm': 120, 'segments': []}
    self.jobs[job_id] = record
    return record

  async def load_structure_result(self, record: job_manager.JobRecord):
    return record.result

  async def ensure_stems_archive(self, record: job_manager.JobRecord):
    return record.stems_path


@pytest.fixture
def client(monkeypatch):
  monkeypatch.setattr(app_module, 'AnalysisJobManager', FakeManager)
  settings = ServerSettings()
  app = app_module.create_app(settings)
  with TestClient(app) as test_client:
    yield test_client


def test_submit_and_fetch(client):
  response = client.post(
    '/jobs',
    data={'track_hash': 'user-1'},
    files={'file': ('clip.wav', b'wave-data', 'audio/wav')},
  )
  assert response.status_code == 200
  payload = response.json()
  assert payload['status'] == job_manager.JobStatus.COMPLETED.value
  job_id = payload['job_id']

  detail = client.get(f'/jobs/{job_id}')
  assert detail.status_code == 200
  assert detail.json()['structure_ready'] is True

  stems_response = client.get(f'/jobs/{job_id}/stems')
  assert stems_response.status_code == 200
  assert stems_response.headers['content-type'] == 'application/zip'

  struct_response = client.get(f'/jobs/{job_id}/structure')
  assert struct_response.status_code == 200
  assert struct_response.json()['bpm'] == 120


def test_structure_pending_returns_202(client):
  manager: FakeManager = client.app.state.manager  # type: ignore[attr-defined]
  metadata = job_manager.JobMetadata(user_hash='track-2', filename='clip.wav')
  pending = job_manager.JobRecord(job_id='pending', metadata=metadata, content_hash='hash')
  pending.status = job_manager.JobStatus.PARTIAL
  pending.stems_path = None
  manager.jobs[pending.job_id] = pending

  response = client.get('/jobs/pending/structure')
  assert response.status_code == 202
  assert 'not ready' in response.json()['detail']
