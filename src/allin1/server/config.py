from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessingMode(str, Enum):
  SEQUENTIAL = 'sequential'
  PARALLEL = 'parallel'


class StorageEvictionMode(str, Enum):
  REFUSE = 'refuse'
  EVICT_OLDEST = 'evict_oldest'


class ServerSettings(BaseSettings):
  """Configuration for the HTTP server runtime."""

  model_config = SettingsConfigDict(env_prefix='ALLIN1_', env_file='.env', env_nested_delimiter='__')

  host: str = Field('127.0.0.1', description='Bind address for the HTTP server.')
  port: int = Field(8000, ge=1, le=65535, description='Listening port for the HTTP server.')
  allowed_origins: List[str] = Field(
    default_factory=lambda: ['http://localhost'],
    description='HTTP origins allowed for CORS requests.',
  )
  max_audio_duration_seconds: int = Field(
    600,
    ge=1,
    description='Maximum length of the uploaded audio segment in seconds (default 10 minutes).',
  )
  max_pending_jobs: int = Field(
    16,
    ge=1,
    description='Maximum number of queued analysis jobs waiting to be processed.',
  )
  processing_mode: ProcessingMode = Field(
    ProcessingMode.SEQUENTIAL,
    description='Whether to handle queued jobs sequentially or in parallel.',
  )
  max_parallel_tasks: int = Field(
    2,
    ge=1,
    description='Maximum number of concurrent analysis tasks when running in parallel mode.',
  )
  cache_size: int = Field(
    8,
    ge=0,
    description='Number of completed analyses to keep in cache for immediate reuse.',
  )
  preload_demucs: bool = Field(
    True,
    description='Load the Demucs separator into memory during startup.',
  )
  demucs_batch_size: int = Field(
    5,
    ge=1,
    description='Maximum number of jobs to group together for a Demucs separation batch.',
  )
  structure_batch_size: int = Field(
    1,
    ge=1,
    description='Maximum number of jobs to process together during Harmonix analysis.',
  )
  demucs_release_after_batch: bool = Field(
    True,
    description='Release Demucs model memory after each batch completes.',
  )
  cleanup_interval_seconds: int = Field(
    30,
    ge=1,
    description='Frequency in seconds for background cleanup tasks (e.g., purging stale jobs).',
  )
  device: str = Field(
    'auto',
    description='Computation device for Demucs and Harmonix models (auto, cpu, cuda, etc.).',
  )
  demucs_device: str | None = Field(
    None,
    description='Override device string for the Demucs separator (defaults to the global device).',
  )
  analysis_device: str | None = Field(
    None,
    description='Override device string for the Harmonix structure model (defaults to the global device).',
  )

  include_activations: bool = Field(False, description='Return activations arrays alongside structural output.')
  include_embeddings: bool = Field(False, description='Return embedding arrays alongside structural output.')

  demucs_model: str = Field('htdemucs', description='Demucs model identifier to use for source separation.')
  harmonix_model: str = Field('harmonix-all', description='Name of the Harmonix checkpoint ensemble to keep in memory.')
  storage_root: Path = Field(
    Path('processed_audio'),
    description='Root directory for persisted uploads, stems, and structure artifacts.',
  )
  max_storage_bytes: int | None = Field(
    None,
    ge=1,
    description='Hard limit on total disk usage for persisted artifacts (bytes).',
  )
  storage_soft_watermark_bytes: int | None = Field(
    None,
    ge=1,
    description='Watermark that triggers eviction/back-pressure before reaching the hard limit (bytes).',
  )
  storage_eviction_mode: StorageEvictionMode = Field(
    StorageEvictionMode.REFUSE,
    description='Strategy when storage exceeds watermark or limit: refuse new jobs or evict oldest.',
  )
  health_metrics_enabled: bool = Field(
    True,
    description='Expose health and telemetry metrics endpoint when enabled.',
  )


@lru_cache()
def get_settings() -> ServerSettings:
  """Return cached settings instance."""
  return ServerSettings()
