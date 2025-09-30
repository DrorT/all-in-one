#!/usr/bin/env python3
"""End-to-end smoke test runner for the All-In-One analysis server.

This script automates the validation workflow discussed in the recent
server refactor:

1. (Optional) Run the targeted pytest suite.
2. Launch the FastAPI server locally via uvicorn.
3. Submit an audio file for analysis through the public HTTP API.
4. Poll the job status until it completes or fails.
5. (Optional) Download the resulting stems ZIP and structure JSON.

Usage example:

    python scripts/smoke_test_server.py \
        /path/to/audio.wav \
        --host 127.0.0.1 --port 8000 \
        --pytest tests/test_server_manager.py \
        --output-dir ./smoke_outputs

The script assumes all dependencies are installed in the active Python
environment (notably httpx and uvicorn) and will surface subprocess
errors verbosely to aid debugging.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterable, Optional

import httpx

DEFAULT_PYTEST_TARGETS = ("tests/test_server_manager.py",)
DEFAULT_WAIT_TIMEOUT = 15 * 60  # 15 minutes
HEALTH_ENDPOINTS = ("/health/metrics", "/docs", "/openapi.json")


class SmokeTestError(RuntimeError):
  """Domain-specific exception for clearer stack traces."""


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "audio_path",
    type=Path,
    help="Path to the audio file that should be analysed.",
  )
  parser.add_argument(
    "--host",
    default="127.0.0.1",
    help="Host interface for the uvicorn server (default: 127.0.0.1).",
  )
  parser.add_argument(
    "--port",
    type=int,
    default=8000,
    help="Port for the uvicorn server (default: 8000).",
  )
  parser.add_argument(
    "--pytest",
    nargs="*",
    default=DEFAULT_PYTEST_TARGETS,
    help=(
      "Pytest targets to exercise before launching the server."
      " Pass '--pytest' with no arguments to skip this step."
    ),
  )
  parser.add_argument(
    "--pytest-extra",
    nargs=argparse.REMAINDER,
    help="Additional arguments forwarded verbatim to pytest after '--'.",
  )
  parser.add_argument(
    "--wait-timeout",
    type=int,
    default=DEFAULT_WAIT_TIMEOUT,
    help="Maximum seconds to wait for job completion (default: 900).",
  )
  parser.add_argument(
    "--poll-interval",
    type=float,
    default=2.0,
    help="Seconds between status polls (default: 2.0).",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    help=(
      "Directory to store downloaded artifacts."
      " Defaults to a temporary directory that will be displayed on success."
    ),
  )
  parser.add_argument(
    "--skip-download",
    action="store_true",
    help="Skip downloading stems.zip and structure.json after completion.",
  )
  parser.add_argument(
    "--keep-temp",
    action="store_true",
    help="Do not delete the temporary artifact directory on success.",
  )
  parser.add_argument(
    "--uvicorn-extra",
    nargs=argparse.REMAINDER,
    help="Additional arguments forwarded to the uvicorn process after '--'.",
  )
  return parser.parse_args()


def run_pytest(targets: Iterable[str], extra: Optional[Iterable[str]]) -> None:
  targets = list(targets or [])
  if not targets:
    print("[pytest] Skipping pytest step as requested.")
    return

  command = [sys.executable, "-m", "pytest", *targets]
  if extra:
    command.extend(extra)

  print(f"[pytest] Running: {' '.join(command)}")
  subprocess.run(command, check=True)
  print("[pytest] Completed successfully.")


@contextlib.contextmanager
def running_server(host: str, port: int, uvicorn_extra: Optional[Iterable[str]] = None):
  env = os.environ.copy()
  command = [
    sys.executable,
    "-m",
    "uvicorn",
    "allin1.server.app:app",
    "--host",
    host,
    "--port",
    str(port),
  ]
  if uvicorn_extra:
    command.extend(uvicorn_extra)

  print(f"[uvicorn] Launching server: {' '.join(command)}")
  process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

  try:
    wait_for_server(host, port, process)
    yield process
  finally:
    print("[uvicorn] Terminating server...")
    with contextlib.suppress(ProcessLookupError):
      process.send_signal(signal.SIGINT)
    try:
      process.wait(timeout=10)
    except subprocess.TimeoutExpired:
      print("[uvicorn] Force killing server process.")
      with contextlib.suppress(ProcessLookupError):
        process.kill()


def wait_for_server(host: str, port: int, process: subprocess.Popen, timeout: int = 120) -> None:
  base_url = f"http://{host}:{port}"
  deadline = time.time() + timeout
  last_exception: Optional[Exception] = None
  with httpx.Client(timeout=5.0) as client:
    while time.time() < deadline:
      if process.poll() is not None:
        raise SmokeTestError("uvicorn process exited prematurely.")
      for endpoint in HEALTH_ENDPOINTS:
        try:
          response = client.get(f"{base_url}{endpoint}")
          if response.status_code < 500:
            print(f"[uvicorn] Server is ready (checked {endpoint}).")
            return
        except Exception as exc:  # pylint: disable=broad-except
          last_exception = exc
      time.sleep(1.0)
  raise SmokeTestError(f"Timed out waiting for server readiness: {last_exception}")


def submit_job(client: httpx.Client, base_url: str, audio_path: Path) -> str:
  if not audio_path.exists():
    raise SmokeTestError(f"Audio file does not exist: {audio_path}")

  user_hash = hashlib.sha256(audio_path.read_bytes()).hexdigest()[:16]
  print(f"[client] Submitting job for {audio_path}...")
  with audio_path.open("rb") as audio_handle:
    files = {
      "file": (audio_path.name, audio_handle, "application/octet-stream"),
    }
    data = {
      "filename": audio_path.name,
      "track_hash": user_hash,
    }
    response = client.post(f"{base_url}/jobs", files=files, data=data)
  response.raise_for_status()
  payload = response.json()
  job_id = payload.get("job_id")
  if not job_id:
    raise SmokeTestError(f"Job submission response missing job_id: {json.dumps(payload)}")
  print(f"[client] Job accepted with id={job_id}")
  return job_id


def poll_job_status(
  client: httpx.Client,
  base_url: str,
  job_id: str,
  timeout: int,
  interval: float,
) -> dict:
  deadline = time.time() + timeout
  last_payload: dict | None = None
  while time.time() < deadline:
    response = client.get(f"{base_url}/jobs/{job_id}")
    response.raise_for_status()
    payload = response.json()
    last_payload = payload
    status = payload.get("status")
    print(f"[client] Job {job_id} status: {status}")
    if status in {"completed", "error", "cancelled"}:
      return payload
    time.sleep(interval)
  raise SmokeTestError(f"Job {job_id} did not finish within timeout; last payload: {last_payload}")


def download_artifacts(client: httpx.Client, base_url: str, job_id: str, output_dir: Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)

  stems_path = output_dir / f"{job_id}_stems.zip"
  struct_path = output_dir / f"{job_id}_structure.json"

  print(f"[client] Downloading stems archive to {stems_path}")
  response = client.get(f"{base_url}/jobs/{job_id}/stems")
  response.raise_for_status()
  stems_path.write_bytes(response.content)

  print(f"[client] Downloading structure JSON to {struct_path}")
  response = client.get(f"{base_url}/jobs/{job_id}/structure")
  response.raise_for_status()
  struct_path.write_bytes(response.content)


def main() -> int:
  args = parse_args()

  if args.pytest is not None:
    run_pytest(args.pytest, args.pytest_extra)

  temp_dir: Optional[tempfile.TemporaryDirectory[str]] = None
  artifacts_dir: Path
  if args.output_dir:
    artifacts_dir = args.output_dir
  else:
    temp_dir = tempfile.TemporaryDirectory(prefix="allin1_smoke_")
    artifacts_dir = Path(temp_dir.name)

  uvicorn_extra = args.uvicorn_extra or []
  if uvicorn_extra and uvicorn_extra[0] == "--":
    uvicorn_extra = uvicorn_extra[1:]

  try:
    with running_server(args.host, args.port, uvicorn_extra):
      base_url = f"http://{args.host}:{args.port}"
      with httpx.Client(timeout=None) as client:
        job_id = submit_job(client, base_url, args.audio_path)
        payload = poll_job_status(
          client,
          base_url,
          job_id,
          timeout=args.wait_timeout,
          interval=args.poll_interval,
        )
        status = payload.get("status")
        if status != "completed":
          raise SmokeTestError(f"Job {job_id} finished with status={status}: {json.dumps(payload)}")

        print(f"[client] Job {job_id} completed successfully.")
        if not args.skip_download:
          download_artifacts(client, base_url, job_id, artifacts_dir)
        print(f"[client] Artifacts directory: {artifacts_dir}")
  except Exception as exc:  # pylint: disable=broad-except
    if temp_dir is not None and not args.keep_temp:
      temp_dir.cleanup()
    raise

  if temp_dir is not None and not args.keep_temp:
    print(f"[cleanup] Removing temporary directory {artifacts_dir}")
    temp_dir.cleanup()

  print("[smoke-test] All steps completed successfully.")
  return 0


if __name__ == "__main__":
  try:
    raise SystemExit(main())
  except SmokeTestError as error:
    print(f"[smoke-test] FAILURE: {error}", file=sys.stderr)
    raise SystemExit(1)
