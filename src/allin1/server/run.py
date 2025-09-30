import argparse
from typing import List, Optional

import uvicorn

from .app import create_app
from .config import ProcessingMode, ServerSettings, get_settings


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description='Run the All-In-One analysis HTTP server.')
  parser.add_argument('--host', type=str, help='Bind host for the HTTP server')
  parser.add_argument('--port', type=int, help='Listening port for the HTTP server')
  parser.add_argument('--allowed-origin', action='append', help='Allowed CORS origin (can be provided multiple times)')
  parser.add_argument('--max-audio-duration', type=int, help='Maximum audio duration in seconds')
  parser.add_argument('--max-pending-jobs', type=int, help='Maximum number of queued jobs')
  parser.add_argument('--processing-mode', choices=[mode.value for mode in ProcessingMode], help='Processing mode for queued jobs')
  parser.add_argument('--max-parallel-tasks', type=int, help='Maximum number of concurrent jobs when running in parallel mode')
  parser.add_argument('--cache-size', type=int, help='Number of completed tracks to keep in cache')
  parser.add_argument('--include-activations', action='store_true', help='Include activation matrices in responses')
  parser.add_argument('--include-embeddings', action='store_true', help='Include embedding vectors in responses')
  parser.add_argument('--demucs-model', type=str, help='Demucs model name for source separation')
  parser.add_argument('--harmonix-model', type=str, help='Harmonix ensemble name for structure analysis')
  parser.add_argument('--device', type=str, help='Computation device to use (e.g., cpu, cuda, auto)')
  parser.add_argument('--demucs-device', type=str, help='Device override for the Demucs separator')
  parser.add_argument('--analysis-device', type=str, help='Device override for the structure model')
  parser.add_argument('--reload', action='store_true', help='Enable auto-reload (development only)')
  parser.add_argument('--log-level', type=str, default='info', help='Logging level for Uvicorn')
  return parser


def _override_settings(settings: ServerSettings, args: argparse.Namespace) -> ServerSettings:
  updates = {}
  if args.host is not None:
    updates['host'] = args.host
  if args.port is not None:
    updates['port'] = args.port
  if args.allowed_origin:
    updates['allowed_origins'] = args.allowed_origin
  if args.max_audio_duration is not None:
    updates['max_audio_duration_seconds'] = args.max_audio_duration
  if args.max_pending_jobs is not None:
    updates['max_pending_jobs'] = args.max_pending_jobs
  if args.processing_mode is not None:
    updates['processing_mode'] = ProcessingMode(args.processing_mode)
  if args.max_parallel_tasks is not None:
    updates['max_parallel_tasks'] = args.max_parallel_tasks
  if args.cache_size is not None:
    updates['cache_size'] = args.cache_size
  if args.include_activations:
    updates['include_activations'] = True
  if args.include_embeddings:
    updates['include_embeddings'] = True
  if args.demucs_model is not None:
    updates['demucs_model'] = args.demucs_model
  if args.harmonix_model is not None:
    updates['harmonix_model'] = args.harmonix_model
  if args.device is not None:
    updates['device'] = args.device
  if args.demucs_device is not None:
    updates['demucs_device'] = args.demucs_device
  if args.analysis_device is not None:
    updates['analysis_device'] = args.analysis_device

  if not updates:
    return settings

  return settings.model_copy(update=updates)


def main(argv: Optional[List[str]] = None):
  parser = build_parser()
  args = parser.parse_args(argv)

  settings = get_settings()
  settings = _override_settings(settings, args)

  config = uvicorn.Config(
    app=lambda: create_app(settings),
    host=settings.host,
    port=settings.port,
    log_level=args.log_level,
    reload=args.reload,
    factory=True,
  )
  server = uvicorn.Server(config)
  server.run()


if __name__ == '__main__':
  main()
