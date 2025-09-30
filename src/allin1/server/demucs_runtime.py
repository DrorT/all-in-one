from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict

import torch
import torchaudio
from demucs.apply import apply_model
from demucs.pretrained import get_model


class DemucsSeparator:
  """In-process Demucs separator that keeps the model resident in memory."""

  def __init__(self, model_name: str = 'htdemucs', device: str | torch.device = 'cuda'):
    self.model_name = model_name
    self.device = torch.device(device)
    self._model = None
    self.sample_rate = None

  async def preload(self):
    if self._model is None:
      self._model = await asyncio.to_thread(get_model, self.model_name)
      self.sample_rate = getattr(self._model, 'samplerate', 44100)
      self._model.to(self.device)
      self._model.eval()

  async def separate_to_directory(self, audio_path: Path, output_dir: Path) -> Dict[str, Path]:
    if self._model is None:
      await self.preload()

    return await asyncio.to_thread(self._separate_sync, audio_path, output_dir)

  def _separate_sync(self, audio_path: Path, output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    waveform, original_sr = torchaudio.load(audio_path)

    if waveform.shape[0] == 1:
      waveform = waveform.repeat(2, 1)

    if original_sr != self.sample_rate:
      waveform = torchaudio.functional.resample(
        waveform,
        original_sr,
        self.sample_rate,
      )

    mix = waveform.unsqueeze(0).to(self.device)

    with torch.no_grad():
      sources = apply_model(
        self._model,
        mix,
        device=self.device,
        split=True,
        overlap=0.25,
      )[0]

    sources = sources.cpu()

    stem_paths: Dict[str, Path] = {}
    for name, tensor in zip(self._model.sources, sources):
      stem_path = output_dir / f'{name}.wav'
      torchaudio.save(
        stem_path,
        tensor,
        self.sample_rate,
        encoding='PCM_S',
      )
      stem_paths[name] = stem_path

    return stem_paths
