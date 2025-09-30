from __future__ import annotations

from typing import Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import ServerSettings, get_settings
from .job_manager import AnalysisJobManager, JobRecord, JobStatus, JobMetadata


class JobSummary(BaseModel):
  job_id: str
  status: JobStatus
  user_hash: str = Field(..., alias='track_hash')
  content_hash: str
  segment_start: Optional[float] = None
  segment_end: Optional[float] = None
  stems_ready: bool = False
  structure_ready: bool = False
  error: Optional[str] = None

  class Config:
    populate_by_name = True


def create_app(settings: ServerSettings | None = None) -> FastAPI:
  settings = settings or get_settings()

  app = FastAPI(title='All-In-One Analysis Service', version='0.1.0')
  app.state.settings = settings

  app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
  )

  manager = AnalysisJobManager(settings)
  app.state.manager = manager

  @app.on_event('startup')
  async def on_startup():
    await manager.startup()

  @app.on_event('shutdown')
  async def on_shutdown():
    await manager.shutdown()

  async def get_manager() -> AnalysisJobManager:
    return manager

  @app.post('/jobs', response_model=JobSummary)
  async def submit_job(
    file: UploadFile = File(...),
    track_hash: str = Form(..., description='Hash identifier provided by client'),
    segment_start: Optional[float] = Form(None),
    segment_end: Optional[float] = Form(None),
    job_manager: AnalysisJobManager = Depends(get_manager),
  ):
    audio_bytes = await file.read()
    if not audio_bytes:
      raise HTTPException(status_code=400, detail='Empty audio payload')

    metadata = JobMetadata(
      user_hash=track_hash,
      filename=file.filename or 'upload',
      segment_start=segment_start,
      segment_end=segment_end,
    )

    record = await job_manager.submit(audio_bytes, metadata)
    return _to_summary(record)

  @app.get('/jobs', response_model=list[JobSummary])
  async def list_jobs(job_manager: AnalysisJobManager = Depends(get_manager)):
    return [_to_summary(job) for job in job_manager.list_jobs()]

  @app.get('/jobs/{job_id}', response_model=JobSummary)
  async def get_job(job_id: str, job_manager: AnalysisJobManager = Depends(get_manager)):
    record = await job_manager.get_job(job_id)
    if record is None:
      raise HTTPException(status_code=404, detail='Job not found')
    return _to_summary(record)

  @app.get('/jobs/{job_id}/structure')
  async def fetch_structure(job_id: str, job_manager: AnalysisJobManager = Depends(get_manager)):
    record = await job_manager.get_job(job_id)
    if record is None:
      raise HTTPException(status_code=404, detail='Job not found')
    result = await job_manager.load_structure_result(record)
    if result is None:
      raise HTTPException(status_code=202, detail='Structure not ready')
    return JSONResponse(result)

  @app.get('/jobs/{job_id}/stems')
  async def fetch_stems(job_id: str, job_manager: AnalysisJobManager = Depends(get_manager)):
    record = await job_manager.get_job(job_id)
    if record is None:
      raise HTTPException(status_code=404, detail='Job not found')
    stems_path = await job_manager.ensure_stems_archive(record)
    if stems_path is None:
      raise HTTPException(status_code=202, detail='Stems not ready')
    return FileResponse(
      path=stems_path,
      media_type='application/zip',
      filename=f'{job_id}_stems.zip',
    )

  if settings.health_metrics_enabled:
    @app.get('/health/metrics')
    async def health_metrics(job_manager: AnalysisJobManager = Depends(get_manager)):
      return JSONResponse(job_manager.get_metrics())

  return app


def _to_summary(record: JobRecord) -> JobSummary:
  return JobSummary(
    job_id=record.job_id,
    status=record.status,
    track_hash=record.metadata.user_hash,
    content_hash=record.content_hash,
    segment_start=record.metadata.segment_start,
    segment_end=record.metadata.segment_end,
    stems_ready=record.stored.has_stems,
    structure_ready=record.stored.has_structure,
    error=record.error,
  )


app = create_app()

__all__ = ["create_app", "app"]
