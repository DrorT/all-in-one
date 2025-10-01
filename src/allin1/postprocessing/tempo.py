import numpy as np
import librosa
from typing import List, Tuple, Dict, Any
from pathlib import Path


def analyze_bpm_with_librosa(
  audio_path: Path,
  model_beats: List[float],
  sr: int | None = None,
) -> Dict[str, Any]:
  """Estimate BPM using librosa and compare with model-derived beats.

  Parameters
  ----------
  audio_path : Path
    Path to the audio file to analyze.
  model_beats : List[float]
    Beat timestamps (seconds) predicted by the model.
  sr : Optional[int]
    Target sampling rate for librosa.load. Defaults to librosa's native behavior.

  Returns
  -------
  Dict[str, Any]
    Dictionary containing BPM estimates and comparison metrics. On failure, an
    ``error`` field is included and librosa-specific outputs are set to neutral
    defaults.
  """

  model_bpm = estimate_tempo_from_beats(model_beats)

  try:
    y, sr_actual = librosa.load(str(audio_path), sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr_actual)

    if isinstance(tempo, np.ndarray):
      tempo_value = float(tempo[0]) if tempo.size else None
    else:
      tempo_value = float(tempo) if tempo is not None else None

    librosa_beats = librosa.frames_to_time(beat_frames, sr=sr_actual).tolist()
    bpm_difference = (
      abs(tempo_value - model_bpm)
      if tempo_value is not None and model_bpm is not None
      else None
    )

    beat_comparison: Dict[str, Any] = {}
    if model_beats and librosa_beats:
      model_beats_arr = np.array(model_beats)
      librosa_beats_arr = np.array(librosa_beats)

      if model_beats_arr.size > 1 and librosa_beats_arr.size > 1:
        min_len = min(model_beats_arr.size, librosa_beats_arr.size)
        mae = float(
          np.mean(
            np.abs(model_beats_arr[:min_len] - librosa_beats_arr[:min_len])
          )
        )
        beat_comparison = {
          'mean_absolute_error': mae,
          'model_beat_count': int(model_beats_arr.size),
          'librosa_beat_count': int(librosa_beats_arr.size),
        }

    return {
      'librosa_bpm': tempo_value,
      'librosa_beats': librosa_beats,
      'model_bpm': model_bpm,
      'bpm_difference': bpm_difference,
      'beat_comparison': beat_comparison,
    }

  except Exception as exc:  # pylint: disable=broad-except
    return {
      'librosa_bpm': None,
      'librosa_beats': [],
      'model_bpm': model_bpm,
      'bpm_difference': None,
      'beat_comparison': {},
      'error': str(exc),
    }


def estimate_tempo_from_beats(
  beats: List[float],
):
  if len(beats) < 2:
    # The song has less than 2 beats. Perhaps it doesn't have much percussive elements.
    return None

  beats = np.array(beats)
  beat_interval = np.diff(beats)
  bpm = 60. / beat_interval
  bpm = bpm.round().astype(int)
  bincount = np.bincount(bpm)
  bpm_range = np.arange(len(bincount))
  bpm_strength = bincount / bincount.sum()
  bpm_cand = np.stack([bpm_range, bpm_strength], axis=-1)
  bpm_cand = bpm_cand[np.argsort(bpm_strength)[::-1]]
  bpm_cand = bpm_cand[bpm_cand[:, 1] > 0]

  bpm_est = bpm_cand[0, 0]
  bpm_est = int(bpm_est)

  return bpm_est


def analyze_structure_with_librosa(
  audio_path: Path,
  model_beats: List[float],
  model_downbeats: List[float] = None,
) -> Dict[str, Any]:
  """
  Analyze BPM, downbeats, and track structure using librosa and compare with model results.
  
  Parameters
  ----------
  audio_path : Path
    Path to the original audio file
  model_beats : List[float]
    Beat times detected by the model
  model_downbeats : List[float], optional
    Downbeat times detected by the model
    
  Returns
  -------
  Dict[str, Any]
    Dictionary containing librosa BPM, beats, downbeats, structure analysis, and comparison metrics
  """
  try:
    # Load the audio file
    y, sr = librosa.load(str(audio_path), sr=None)
    duration = len(y) / sr if sr else 0.0
    
    # Use librosa's beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Handle tempo which might be an array or scalar
    if isinstance(tempo, np.ndarray):
        tempo_value = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo_value = float(tempo)
    
    # Convert beat frames to time
    librosa_beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Calculate BPM difference
    model_bpm = estimate_tempo_from_beats(model_beats)
    bpm_diff = abs(tempo_value - model_bpm) if model_bpm is not None else None
    
    # Calculate beat alignment metrics
    beat_comparison = {}
    if len(model_beats) > 0 and len(librosa_beat_times) > 0:
      # Find closest beats between model and librosa
      model_beats_array = np.array(model_beats)
      librosa_beats_array = np.array(librosa_beat_times)
      
      # Calculate mean absolute error between beat times
      # We need to align the beats first - find the best offset
      if len(model_beats_array) > 1 and len(librosa_beats_array) > 1:
        # Try different alignments to find the best match
        min_offset = -0.5  # seconds
        max_offset = 0.5   # seconds
        num_offsets = 100
        
        offsets = np.linspace(min_offset, max_offset, num_offsets)
        best_mae = float('inf')
        best_offset = 0
        
        for offset in offsets:
          shifted_librosa = librosa_beats_array + offset
          # Trim to the shorter length
          min_len = min(len(model_beats_array), len(shifted_librosa))
          if min_len > 0:
            mae = np.mean(np.abs(model_beats_array[:min_len] - shifted_librosa[:min_len]))
            if mae < best_mae:
              best_mae = mae
              best_offset = offset
        
        beat_comparison = {
          'mean_absolute_error': best_mae,
          'best_offset': best_offset,
          'model_beat_count': len(model_beats),
          'librosa_beat_count': len(librosa_beat_times),
        }
    
    # Perform downbeat detection
    try:
      # Estimate downbeats using librosa
      # First, compute harmonic-percussive source separation
      y_harmonic = librosa.effects.harmonic(y)
      
      # Compute chroma features from the harmonic component
      chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
      
      # Use librosa's beat tracking with downbeat estimation
      # This is a simplified approach - librosa doesn't have direct downbeat tracking
      # We'll use the first beat of each measure as an approximation
      # Assuming 4/4 time signature, every 4 beats is a downbeat
      
      # Find downbeats by analyzing chroma changes and beat positions
      downbeat_estimates = []
      if len(librosa_beat_times) > 0:
        # Simple heuristic: look for significant chroma changes at beat positions
        chroma_changes = np.diff(chroma, axis=1)
        beat_frames = librosa.time_to_frames(librosa_beat_times, sr=sr)
        
        # Normalize beat frames to chroma feature indices
        beat_indices = (beat_frames * chroma.shape[1] / len(y)).astype(int)
        beat_indices = np.clip(beat_indices, 0, chroma.shape[1] - 1)
        
        # Calculate chroma change at each beat
        beat_chroma_changes = []
        for i in range(len(beat_indices) - 1):
          if beat_indices[i] < chroma_changes.shape[1]:
            change = np.mean(np.abs(chroma_changes[:, beat_indices[i]]))
            beat_chroma_changes.append(change)
        
        # Find peaks in chroma changes (potential section boundaries)
        if len(beat_chroma_changes) > 0:
          beat_chroma_changes = np.array(beat_chroma_changes)
          threshold = np.percentile(beat_chroma_changes, 75)  # Top 25% of changes
          potential_downbeats = np.where(beat_chroma_changes > threshold)[0]
          
          # Convert to actual times
          for idx in potential_downbeats:
            if idx < len(librosa_beat_times) - 1:
              downbeat_estimates.append(librosa_beat_times[idx + 1])
        
        # Also add the first beat as a downbeat
        if len(librosa_beat_times) > 0:
          downbeat_estimates.insert(0, librosa_beat_times[0])
      
      # Perform structure analysis using librosa's segment detection
      try:
        # Use a standard hop length for structural analysis
        hop_length = 1024
        
        # Apply temporal segmentation directly on chroma features
        # This is more efficient than computing a self-similarity matrix first
        # Estimate k based on track duration (approximately one segment per 30 seconds)
        k = max(4, min(12, int(duration / 30)))  # Between 4-12 segments
        
        # Find segment boundaries
        bounds_frames = librosa.segment.agglomerative(chroma, k=k)
        
        # Convert the frame indices back into time (seconds)
        bound_times = librosa.frames_to_time(bounds_frames, sr=sr, hop_length=hop_length)
        
        # Ensure all boundaries are within the actual duration
        bound_times = np.clip(bound_times, 0, duration)
        
        # Create segment descriptions
        segments = []
        for i in range(len(bound_times) - 1):
          start = float(bound_times[i])
          end = float(bound_times[i + 1])
          # Ensure segments are valid
          if start < end and start < duration:
            segments.append({
              'start': start,
              'end': min(end, duration),
              'label': f'Section {i+1}'
            })
        
        # Add the last segment to the end of the track if needed
        if len(bound_times) > 0 and (len(segments) == 0 or segments[-1]['end'] < duration):
          last_start = float(bound_times[-1])
          if last_start < duration:
            segments.append({
              'start': last_start,
              'end': float(duration),
              'label': f'Section {len(segments)+1}'
            })
        
        # Find repeating sections by comparing chroma patterns
        repeating_sections = []
        if len(segments) > 1:
          segment_chromas = []
          for seg in segments:
            start_frame = librosa.time_to_frames(seg['start'], sr=sr, hop_length=hop_length)
            end_frame = librosa.time_to_frames(seg['end'], sr=sr, hop_length=hop_length)
            if end_frame > start_frame and end_frame <= chroma.shape[1]:
              segment_chroma = np.mean(chroma[:, start_frame:end_frame], axis=1)
              segment_chromas.append(segment_chroma)
          
          # Compare segments for similarity
          for i in range(len(segment_chromas)):
            for j in range(i + 1, len(segment_chromas)):
              similarity = np.corrcoef(segment_chromas[i], segment_chromas[j])[0, 1]
              if similarity > 0.7:  # Threshold for similarity
                repeating_sections.append({
                  'section1': segments[i]['label'],
                  'section2': segments[j]['label'],
                  'similarity': float(similarity)
                })
      
      except Exception as e:
        print(f"Warning: Structure analysis failed: {e}")
        segments = []
        repeating_sections = []
      
      # Compare downbeats with model if available
      downbeat_comparison = {}
      if model_downbeats and len(downbeat_estimates) > 0:
        model_downbeats_array = np.array(model_downbeats)
        librosa_downbeats_array = np.array(downbeat_estimates)
        
        # Calculate alignment metrics
        if len(model_downbeats_array) > 0 and len(librosa_downbeats_array) > 0:
          # Find closest downbeats
          min_offset = -1.0  # seconds
          max_offset = 1.0   # seconds
          num_offsets = 100
          
          offsets = np.linspace(min_offset, max_offset, num_offsets)
          best_mae = float('inf')
          best_offset = 0
          
          for offset in offsets:
            shifted_librosa = librosa_downbeats_array + offset
            # Trim to the shorter length
            min_len = min(len(model_downbeats_array), len(shifted_librosa))
            if min_len > 0:
              mae = np.mean(np.abs(model_downbeats_array[:min_len] - shifted_librosa[:min_len]))
              if mae < best_mae:
                best_mae = mae
                best_offset = offset
          
          downbeat_comparison = {
            'mean_absolute_error': best_mae,
            'best_offset': best_offset,
            'model_downbeat_count': len(model_downbeats),
            'librosa_downbeat_count': len(downbeat_estimates),
          }
      
      return {
        'librosa_bpm': tempo_value,
        'librosa_beats': librosa_beat_times.tolist(),
        'librosa_downbeats': downbeat_estimates,
        'librosa_segments': segments,
        'librosa_repeating_sections': repeating_sections,
        'model_bpm': model_bpm,
        'model_downbeats': model_downbeats,
        'bpm_difference': bpm_diff,
        'beat_comparison': beat_comparison,
        'downbeat_comparison': downbeat_comparison,
      }
      
    except Exception as e:
      # Return a dictionary with error information if structure analysis fails
      return {
        'librosa_bpm': tempo_value,
        'librosa_beats': librosa_beat_times.tolist(),
        'librosa_downbeats': [],
        'librosa_segments': [],
        'librosa_repeating_sections': [],
        'model_bpm': model_bpm,
        'model_downbeats': model_downbeats,
        'bpm_difference': bpm_diff,
        'beat_comparison': beat_comparison,
        'downbeat_comparison': {},
        'error': str(e),
      }
    
  except Exception as e:
    # Return a dictionary with error information if librosa analysis fails
    return {
      'librosa_bpm': None,
      'librosa_beats': [],
      'model_bpm': estimate_tempo_from_beats(model_beats),
      'bpm_difference': None,
      'beat_comparison': {},
      'error': str(e),
    }


def analyze_structure_lightweight(
  audio_path: Path,
  model_beats: List[float],
  model_downbeats: List[float] = None,
  max_duration: float = 180.0,  # Limit to 3 minutes
) -> Dict[str, Any]:
  """
  Lightweight analysis of BPM, downbeats, and track structure using librosa.
  Avoids CPU-intensive operations like self-similarity matrix calculation.
  
  Parameters
  ----------
  audio_path : Path
    Path to the original audio file
  model_beats : List[float]
    Beat times detected by the model
  model_downbeats : List[float], optional
    Downbeat times detected by the model
  max_duration : float, optional
    Maximum duration of audio to analyze (seconds)
    
  Returns
  -------
  Dict[str, Any]
    Dictionary containing librosa BPM, beats, downbeats, simple structure analysis, and comparison metrics
  """
  try:
    # Load the audio file with duration limit
    y, sr = librosa.load(str(audio_path), sr=None, duration=max_duration)
    duration = len(y) / sr
    
    # Use librosa's beat tracking
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Handle tempo which might be an array or scalar
    if isinstance(tempo, np.ndarray):
      tempo_value = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
      tempo_value = float(tempo)
    
    # Convert beat frames to time
    librosa_beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Calculate BPM difference
    model_bpm = estimate_tempo_from_beats(model_beats)
    bpm_diff = abs(tempo_value - model_bpm) if model_bpm is not None else None
    
    # Calculate beat alignment metrics
    beat_comparison = {}
    if len(model_beats) > 0 and len(librosa_beat_times) > 0:
      # Find closest beats between model and librosa
      model_beats_array = np.array(model_beats)
      librosa_beats_array = np.array(librosa_beat_times)
      
      # Calculate mean absolute error between beat times
      # We need to align the beats first - find the best offset
      if len(model_beats_array) > 1 and len(librosa_beats_array) > 1:
        # Try different alignments to find the best match
        min_offset = -0.5  # seconds
        max_offset = 0.5   # seconds
        num_offsets = 100
        
        offsets = np.linspace(min_offset, max_offset, num_offsets)
        best_mae = float('inf')
        best_offset = 0
        
        for offset in offsets:
          shifted_librosa = librosa_beats_array + offset
          # Trim to the shorter length
          min_len = min(len(model_beats_array), len(shifted_librosa))
          if min_len > 0:
            mae = np.mean(np.abs(model_beats_array[:min_len] - shifted_librosa[:min_len]))
            if mae < best_mae:
              best_mae = mae
              best_offset = offset
        
        beat_comparison = {
          'mean_absolute_error': best_mae,
          'best_offset': best_offset,
          'model_beat_count': len(model_beats),
          'librosa_beat_count': len(librosa_beat_times),
        }
    
    # Simple downbeat detection (lightweight version)
    downbeat_estimates = []
    if len(librosa_beat_times) >= 4:
      # Simple heuristic: assume 4/4 time signature
      # Every 4th beat is a downbeat
      downbeat_estimates = librosa_beat_times[::4].tolist()
      # Always include the first beat
      if librosa_beat_times[0] not in downbeat_estimates:
        downbeat_estimates.insert(0, float(librosa_beat_times[0]))
    
    # Simple structure detection (lightweight version)
    segments = []
    # Simple time-based segmentation (divide into equal parts)
    num_segments = min(8, int(duration / 30))  # One segment per 30 seconds, max 8
    if num_segments > 0:
      segment_length = duration / num_segments
      for i in range(num_segments):
        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, duration)
        segments.append({
          'start': float(start_time),
          'end': float(end_time),
          'label': f'Section {i+1}'
        })
    
    # Compare downbeats with model if available
    downbeat_comparison = {}
    if model_downbeats and len(downbeat_estimates) > 0:
      model_downbeats_array = np.array(model_downbeats)
      librosa_downbeats_array = np.array(downbeat_estimates)
      
      # Calculate alignment metrics
      if len(model_downbeats_array) > 0 and len(librosa_downbeats_array) > 0:
        # Find closest downbeats
        min_offset = -1.0  # seconds
        max_offset = 1.0   # seconds
        num_offsets = 100
        
        offsets = np.linspace(min_offset, max_offset, num_offsets)
        best_mae = float('inf')
        best_offset = 0
        
        for offset in offsets:
          shifted_librosa = librosa_downbeats_array + offset
          # Trim to the shorter length
          min_len = min(len(model_downbeats_array), len(shifted_librosa))
          if min_len > 0:
            mae = np.mean(np.abs(model_downbeats_array[:min_len] - shifted_librosa[:min_len]))
            if mae < best_mae:
              best_mae = mae
              best_offset = offset
        
        downbeat_comparison = {
          'mean_absolute_error': best_mae,
          'best_offset': best_offset,
          'model_downbeat_count': len(model_downbeats),
          'librosa_downbeat_count': len(downbeat_estimates),
        }
    
    return {
      'librosa_bpm': tempo_value,
      'librosa_beats': librosa_beat_times.tolist(),
      'librosa_downbeats': downbeat_estimates,
      'librosa_segments': segments,
      'librosa_repeating_sections': [],  # Not calculated in lightweight version
      'model_bpm': model_bpm,
      'model_downbeats': model_downbeats,
      'bpm_difference': bpm_diff,
      'beat_comparison': beat_comparison,
      'downbeat_comparison': downbeat_comparison,
      'analysis_type': 'lightweight'
    }
    
  except Exception as e:
    # Return a dictionary with error information if analysis fails
    return {
      'librosa_bpm': None,
      'librosa_beats': [],
      'librosa_downbeats': [],
      'librosa_segments': [],
      'librosa_repeating_sections': [],
      'model_bpm': estimate_tempo_from_beats(model_beats),
      'model_downbeats': model_downbeats,
      'bpm_difference': None,
      'beat_comparison': {},
      'downbeat_comparison': {},
      'error': str(e),
      'analysis_type': 'lightweight'
    }
