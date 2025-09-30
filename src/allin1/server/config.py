from enum import Enum
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProcessingMode(str, Enum):
  SEQUENTIAL = 'sequential'
  PARALLEL = 'parallel'


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


@lru_cache()
def get_settings() -> ServerSettings:
  """Return cached settings instance."""
  return ServerSettings()
