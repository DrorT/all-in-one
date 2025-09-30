from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional


_HASH_SHARD_SLICE = (2, 4)
_METADATA_FILENAME = 'metadata.json'
_INPUT_FILENAME = 'original.wav'
_STRUCT_FILENAME = 'structure.json'
_STEMS_DIRNAME = 'stems'
_STEMS_ZIP_FILENAME = 'stems.zip'
_INPUT_DIRNAME = 'input'
_STRUCT_DIRNAME = 'struct'
_ARTIFACT_VERSION = 1


class StoredJobState(str, Enum):
  QUEUED = 'queued'
  STEMS_PENDING = 'stems_pending'
  STEMS_RUNNING = 'stems_running'
  STRUCT_PENDING = 'struct_pending'
  STRUCT_RUNNING = 'struct_running'
  COMPLETED = 'completed'
  ERROR = 'error'
  CANCELLED = 'cancelled'


@dataclass
class StoredJobMetadata:
  job_id: str
  user_hash: str
  filename: str
  content_hash: str
  state: StoredJobState = StoredJobState.QUEUED
  created_at: float = field(default_factory=lambda: time.time())
  updated_at: float = field(default_factory=lambda: time.time())
  segment_start: float | None = None
  segment_end: float | None = None
  error: str | None = None
  has_stems: bool = False
  has_structure: bool = False
  sample_rate: int | None = None
  artifact_version: int = _ARTIFACT_VERSION

  def to_dict(self) -> dict:
    data = asdict(self)
    data['state'] = self.state.value
    return data

  @classmethod
  def from_dict(cls, payload: dict) -> StoredJobMetadata:
    payload = dict(payload)
    payload['state'] = StoredJobState(payload['state'])
    return cls(**payload)

  def mark_state(self, state: StoredJobState, *, error: str | None = None):
    self.state = state
    self.error = error
    self.updated_at = time.time()

  def mark_stems_ready(self, sample_rate: int | None = None):
    self.has_stems = True
    if sample_rate is not None:
      self.sample_rate = sample_rate
    self.updated_at = time.time()

  def mark_structure_ready(self):
    self.has_structure = True
    self.updated_at = time.time()


def ensure_storage_root(root: Path) -> Path:
  root.mkdir(parents=True, exist_ok=True)
  return root


def job_root(root: Path, content_hash: str) -> Path:
  shard_a, shard_b = _HASH_SHARD_SLICE
  return root / content_hash[:shard_a] / content_hash[shard_a:shard_b] / content_hash


def input_path(root: Path, content_hash: str) -> Path:
  return job_root(root, content_hash) / _INPUT_DIRNAME / _INPUT_FILENAME


def stems_dir(root: Path, content_hash: str) -> Path:
  return job_root(root, content_hash) / _STEMS_DIRNAME


def stems_zip_path(root: Path, content_hash: str) -> Path:
  return job_root(root, content_hash) / _STEMS_ZIP_FILENAME


def struct_path(root: Path, content_hash: str) -> Path:
  return job_root(root, content_hash) / _STRUCT_DIRNAME / _STRUCT_FILENAME


def metadata_path(root: Path, content_hash: str) -> Path:
  return job_root(root, content_hash) / _METADATA_FILENAME


def write_metadata(root: Path, metadata: StoredJobMetadata):
  path = metadata_path(root, metadata.content_hash)
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = path.with_suffix('.tmp')
  with tmp_path.open('w', encoding='utf-8') as fh:
    json.dump(metadata.to_dict(), fh, indent=2)
  tmp_path.replace(path)


def read_metadata(root: Path, content_hash: str) -> Optional[StoredJobMetadata]:
  path = metadata_path(root, content_hash)
  if not path.exists():
    return None
  try:
    with path.open('r', encoding='utf-8') as fh:
      payload = json.load(fh)
  except (OSError, json.JSONDecodeError):
    return None
  try:
    return StoredJobMetadata.from_dict(payload)
  except ValueError:
    return None


def list_existing_jobs(root: Path) -> Iterator[StoredJobMetadata]:
  if not root.exists():
    return
  for metadata_file in root.glob('**/' + _METADATA_FILENAME):
    try:
      with metadata_file.open('r', encoding='utf-8') as fh:
        payload = json.load(fh)
      yield StoredJobMetadata.from_dict(payload)
    except (OSError, json.JSONDecodeError, ValueError):
      continue


def ensure_input_directory(root: Path, content_hash: str) -> Path:
  directory = job_root(root, content_hash) / _INPUT_DIRNAME
  directory.mkdir(parents=True, exist_ok=True)
  return directory


def ensure_stems_directory(root: Path, content_hash: str) -> Path:
  directory = stems_dir(root, content_hash)
  directory.mkdir(parents=True, exist_ok=True)
  return directory


def ensure_struct_directory(root: Path, content_hash: str) -> Path:
  directory = job_root(root, content_hash) / _STRUCT_DIRNAME
  directory.mkdir(parents=True, exist_ok=True)
  return directory


def calculate_storage_usage(root: Path) -> int:
  total = 0
  if not root.exists():
    return total
  for dirpath, _dirnames, filenames in os.walk(root):
    for filename in filenames:
      path = Path(dirpath) / filename
      with contextlib.suppress(OSError):
        total += path.stat().st_size
  return total