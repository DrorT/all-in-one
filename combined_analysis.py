#!/usr/bin/env python3
"""
Combined audio analysis using madmom for beat detection and librosa for segmentation.

This script provides functionality for analyzing audio files, using madmom for
beat detection and librosa for segmentation, then outputs both results together.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import librosa
import numpy as np
from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def analyze_beats_madmom(audio_path: str) -> Dict[str, Any]:
    """
    Analyze audio file for beats using madmom.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary containing beat detection results
    """
    print("Running madmom beat detection...")
    
    # Beat tracking
    beat_processor = RNNBeatProcessor()
    beat_predictions = beat_processor(str(audio_path))
    
    beat_processor_dbn = DBNBeatTrackingProcessor(
        min_bpm=55.0,
        max_bpm=215.0,
        transition_lambda=100,
        correct=True,
        fps=100
    )
    beats = beat_processor_dbn(beat_predictions)
    
    # Downbeat tracking
    downbeat_processor = RNNDownBeatProcessor()
    downbeat_predictions = downbeat_processor(str(audio_path))
    
    downbeat_processor_dbn = DBNDownBeatTrackingProcessor(
        beats_per_bar=[4],  # Most rock songs are in 4/4 time
        min_bpm=55.0,
        max_bpm=215.0,
        transition_lambda=100,
        correct=True,
        fps=100
    )
    downbeats = downbeat_processor_dbn(downbeat_predictions)
    
    # Calculate tempo
    if len(beats) > 1:
        intervals = np.diff(beats)
        tempo = 60.0 / np.median(intervals)
    else:
        tempo = 0.0
    
    # Calculate beat consistency
    if len(intervals) > 1:
        beat_consistency = 1.0 - (np.std(intervals) / np.mean(intervals))
        beat_consistency = max(0.0, min(1.0, beat_consistency))
    else:
        beat_consistency = 0.0
    
    return {
        "beats": beats.tolist(),
        "downbeats": downbeats.tolist(),
        "tempo": float(tempo),
        "beat_count": len(beats),
        "downbeat_count": len(downbeats),
        "beat_consistency": float(beat_consistency)
    }


def analyze_segmentation_librosa(audio_path: str) -> Dict[str, Any]:
    """
    Analyze audio file for segmentation using librosa.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary containing segmentation results
    """
    print("Running librosa segmentation...")
    
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=None)
    
    # Get audio duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Extract features for segmentation
    # Compute MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
    # Compute other spectral features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    
    # Stack features
    features = np.vstack([
        mfccs,
        spectral_centroids,
        spectral_rolloff,
        spectral_bandwidth,
        spectral_contrast,
        chroma,
        tonnetz
    ])
    
    # Normalize features
    features = (features - np.mean(features, axis=1, keepdims=True)) / np.std(features, axis=1, keepdims=True)
    
    # Compute novelty curve based on changes in features
    # Compute the difference between consecutive feature vectors
    diff = np.diff(features, axis=1)
    
    # Compute the magnitude of the difference
    novelty = np.linalg.norm(diff, axis=0)
    
    # Smooth the novelty curve
    novelty = gaussian_filter1d(novelty, sigma=5)
    
    # Find peaks in the novelty curve
    # Find peaks with a minimum height and distance
    peaks, _ = find_peaks(
        novelty, 
        height=np.mean(novelty) + np.std(novelty),
        distance=30  # Minimum distance between peaks (in frames)
    )
    
    # Convert frame indices to time (seconds)
    segment_times = librosa.frames_to_time(peaks, sr=sr)
    
    # Add start and end times
    segment_times = np.concatenate([[0], segment_times, [duration]])
    
    # Create segments
    segments = []
    for i in range(len(segment_times) - 1):
        segments.append({
            "start": segment_times[i],
            "end": segment_times[i + 1],
            "label": f"Section {i + 1}"
        })
    
    # Find repeating sections
    repeating_sections = []
    
    # Compare each segment with every other segment
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            # Get feature vectors for each segment
            start_frame_i = librosa.time_to_frames(segments[i]["start"], sr=sr)
            end_frame_i = librosa.time_to_frames(segments[i]["end"], sr=sr)
            
            start_frame_j = librosa.time_to_frames(segments[j]["start"], sr=sr)
            end_frame_j = librosa.time_to_frames(segments[j]["end"], sr=sr)
            
            # Ensure indices are within bounds
            start_frame_i = max(0, min(start_frame_i, features.shape[1] - 1))
            end_frame_i = max(0, min(end_frame_i, features.shape[1] - 1))
            start_frame_j = max(0, min(start_frame_j, features.shape[1] - 1))
            end_frame_j = max(0, min(end_frame_j, features.shape[1] - 1))
            
            # Get mean feature vectors for each segment
            if end_frame_i > start_frame_i and end_frame_j > start_frame_j:
                features_i = np.mean(features[:, start_frame_i:end_frame_i], axis=1)
                features_j = np.mean(features[:, start_frame_j:end_frame_j], axis=1)
                
                # Compute similarity
                similarity = 1 - np.linalg.norm(features_i - features_j) / np.linalg.norm(features_i)
                
                # If similarity is high enough, consider them repeating
                if similarity > 0.8:  # Threshold for similarity
                    repeating_sections.append({
                        "section1": segments[i]["label"],
                        "section2": segments[j]["label"],
                        "similarity": float(similarity)
                    })
    
    return {
        "segments": segments,
        "segment_count": len(segments),
        "repeating_sections": repeating_sections,
        "repeating_section_count": len(repeating_sections)
    }


def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """
    Analyze audio file using madmom for beats and librosa for segmentation.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary containing combined analysis results
    """
    print(f"Analyzing audio file: {audio_path}")
    
    # Load audio to get basic info
    y, sr = librosa.load(str(audio_path), sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    
    # Run beat detection with madmom
    beat_results = analyze_beats_madmom(audio_path)
    
    # Run segmentation with librosa
    segmentation_results = analyze_segmentation_librosa(audio_path)
    
    # Combine results
    results = {
        "audio_path": str(audio_path),
        "sample_rate": sr,
        "duration": duration,
        "beats": beat_results["beats"],
        "downbeats": beat_results["downbeats"],
        "tempo": beat_results["tempo"],
        "beat_count": beat_results["beat_count"],
        "downbeat_count": beat_results["downbeat_count"],
        "beat_consistency": beat_results["beat_consistency"],
        "segments": segmentation_results["segments"],
        "segment_count": segmentation_results["segment_count"],
        "repeating_sections": segmentation_results["repeating_sections"],
        "repeating_section_count": segmentation_results["repeating_section_count"]
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Combined audio analysis using madmom for beats and librosa for segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python combined_analysis.py audio.wav
  python combined_analysis.py audio.mp3 --output ./results
        """
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to analyze"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Directory to save the results (optional)"
    )
    
    args = parser.parse_args()
    
    # Check if the audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    try:
        # Run the analysis
        results = analyze_audio(audio_path)
        
        # Print results
        print("\nResults:")
        print(f"  Tempo: {results['tempo']:.2f} BPM")
        print(f"  Beat count: {results['beat_count']}")
        print(f"  Downbeat count: {results['downbeat_count']}")
        print(f"  Beat consistency: {results['beat_consistency']:.2f}")
        print(f"  Segment count: {results['segment_count']}")
        print(f"  Repeating sections: {results['repeating_section_count']}")
        
        if results['repeating_sections']:
            print("\n  Repeating sections found:")
            for section in results['repeating_sections'][:10]:  # Show first 10
                print(f"    {section['section1']} ↔ {section['section2']} (similarity: {section['similarity']:.2f})")
            
            if len(results['repeating_sections']) > 10:
                print(f"    ... and {len(results['repeating_sections']) - 10} more")
        
        # Save results to JSON if output directory is specified
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save JSON results
            json_path = output_dir / f"{audio_path.stem}_combined_analysis.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {json_path}")
            
            # Save beats to text file
            beats_path = output_dir / f"{audio_path.stem}_beats.txt"
            with open(beats_path, 'w') as f:
                for beat in results['beats']:
                    f.write(f"{beat}\n")
            print(f"Beats saved to: {beats_path}")
            
            # Save downbeats to text file
            downbeats_path = output_dir / f"{audio_path.stem}_downbeats.txt"
            with open(downbeats_path, 'w') as f:
                for downbeat in results['downbeats']:
                    f.write(f"{downbeat}\n")
            print(f"Downbeats saved to: {downbeats_path}")
            
            # Save segments to text file
            segments_path = output_dir / f"{audio_path.stem}_segments.txt"
            with open(segments_path, 'w') as f:
                for segment in results['segments']:
                    f.write(f"{segment['start']}\t{segment['end']}\t{segment['label']}\n")
            print(f"Segments saved to: {segments_path}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()