#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional, Tuple

import librosa
import numpy as np
import requests
import soundfile as sf


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description='Submit an audio track to an allin1-serve instance and download the results.',
  )
  parser.add_argument('audio_path', type=Path, help='Path to the source audio to analyze.')
  parser.add_argument('-o', '--output-dir', type=Path, default=Path('./client_outputs'), help='Directory to store stems and struct JSON.')
  parser.add_argument('-u', '--server-url', type=str, default='http://127.0.0.1:8000', help='Base URL of the running allin1 server.')
  parser.add_argument('--track-hash', type=str, help='Optional client-provided hash identifier; defaults to SHA256 of the original audio file.')
  parser.add_argument('--segment-start', type=float, help='Optional segment start time in seconds to send as metadata.')
  parser.add_argument('--segment-end', type=float, help='Optional segment end time in seconds to send as metadata.')
  parser.add_argument('--poll-interval', type=float, default=5.0, help='Seconds between status polls.')
  parser.add_argument('--timeout', type=float, default=1800.0, help='Maximum seconds to wait for job completion (default 30 minutes).')
  parser.add_argument(
    '--no-convert',
    action='store_true',
    help='Skip resampling/conversion and upload the file as-is.',
  )
  return parser.parse_args()


def main():
  args = parse_args()
  audio_path = args.audio_path.expanduser().resolve()
  if not audio_path.is_file():
    raise SystemExit(f'Audio file not found: {audio_path}')

  track_hash = args.track_hash or _sha256(audio_path.read_bytes())

  if args.no_convert:
    converted_path = audio_path
    cleanup = None
  else:
    converted_path, cleanup = _convert_to_wav(audio_path)

  try:
    job_id = _submit_job(
      server_url=args.server_url,
      audio_path=converted_path,
      original_filename=audio_path.name,
      track_hash=track_hash,
      segment_start=args.segment_start,
      segment_end=args.segment_end,
    )
    print(f'[*] Submitted job {job_id} (track_hash={track_hash})')

    output_dir = args.output_dir.expanduser().resolve() / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _wait_for_results(
      server_url=args.server_url,
      job_id=job_id,
      output_dir=output_dir,
      poll_interval=args.poll_interval,
      timeout=args.timeout,
    )

    print(f'[+] Analysis complete. Results saved to {output_dir}')
  finally:
    if cleanup:
      cleanup()


def _submit_job(
  *,
  server_url: str,
  audio_path: Path,
  original_filename: str,
  track_hash: str,
  segment_start: Optional[float],
  segment_end: Optional[float],
) -> str:
  with audio_path.open('rb') as fh:
    files = {
      'file': (original_filename, fh, 'audio/wav'),
    }
    data = {
      'track_hash': track_hash,
    }
    if segment_start is not None:
      data['segment_start'] = str(segment_start)
    if segment_end is not None:
      data['segment_end'] = str(segment_end)
    response = requests.post(f'{server_url.rstrip('/')}/jobs', data=data, files=files, timeout=60)

  response.raise_for_status()
  payload = response.json()
  return payload['job_id']

def _wait_for_results(
  *,
  server_url: str,
  job_id: str,
  output_dir: Path,
  poll_interval: float,
  timeout: float,
):
  stems_saved = False
  deadline = time.monotonic() + timeout
  base_url = server_url.rstrip('/')

  while True:
    if time.monotonic() > deadline:
      raise TimeoutError(f'Job {job_id} did not complete within {timeout} seconds')

    status = requests.get(f'{base_url}/jobs/{job_id}', timeout=30)
    if status.status_code == 404:
      raise RuntimeError(f'Job {job_id} not found on server')
    status.raise_for_status()
    summary = status.json()

    print(
      f"[*] Status: {summary['status']} (stems_ready={summary['stems_ready']}, structure_ready={summary['structure_ready']})"
    )

    if summary['status'] == 'error':
      raise RuntimeError(f"Server reported error: {summary.get('error', 'unknown error')}")

    if summary['stems_ready'] and not stems_saved:
      stems_saved = _download_stems(base_url, job_id, output_dir)
      if stems_saved:
        print(f'[+] Stems saved to {output_dir}')

    if summary['status'] == 'completed':
      _download_structure(base_url, job_id, output_dir)
      break

    time.sleep(poll_interval)


def _download_stems(server_url: str, job_id: str, output_dir: Path) -> bool:
  response = requests.get(f'{server_url}/jobs/{job_id}/stems', timeout=60)
  if response.status_code == 202:
    return False
  response.raise_for_status()

  with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
    archive.extractall(output_dir / 'stems')
  return True


def _download_structure(server_url: str, job_id: str, output_dir: Path):
  response = requests.get(f'{server_url}/jobs/{job_id}/structure', timeout=60)
  if response.status_code == 202:
    raise RuntimeError('Structure endpoint returned 202 even though job is marked completed')
  response.raise_for_status()

  struct_path = output_dir / 'structure.json'
  struct_path.write_text(json.dumps(response.json(), indent=2))


def _convert_to_wav(source_path: Path) -> Tuple[Path, Optional[Callable[[], None]]]:
  audio, _ = librosa.load(source_path.as_posix(), sr=44100, mono=False)
  if audio.ndim == 1:
    audio = np.expand_dims(audio, axis=0)
  audio = audio.T  # shape (samples, channels)

  tmp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
  tmp_path = Path(tmp_file.name)
  tmp_file.close()

  sf.write(tmp_path, audio, 44100, subtype='PCM_16')

  def cleanup():
    try:
      tmp_path.unlink()
    except FileNotFoundError:
      pass

  return tmp_path, cleanup


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    sys.exit('\nInterrupted by user')
