#!/usr/bin/env python3
"""
Audio segmentation using librosa library.

This script provides functionality for segmenting audio files using librosa's
segmentation algorithms, which can be more accurate than other libraries for
certain types of music.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


def analyze_audio(audio_path: str) -> Dict[str, Any]:
    """
    Analyze audio file for segmentation using librosa.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Dictionary containing segmentation results
    """
    print(f"Loading audio file: {audio_path}")
    
    # Load audio
    y, sr = librosa.load(str(audio_path), sr=None)
    
    # Get audio duration
    duration = librosa.get_duration(y=y, sr=sr)
    
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Sample rate: {sr} Hz")
    
    # Use librosa for segmentation
    print("Running segmentation analysis...")
    
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
    print("Computing novelty curve...")
    
    # Compute the difference between consecutive feature vectors
    diff = np.diff(features, axis=1)
    
    # Compute the magnitude of the difference
    novelty = np.linalg.norm(diff, axis=0)
    
    # Smooth the novelty curve
    novelty = gaussian_filter1d(novelty, sigma=5)
    
    # Find peaks in the novelty curve
    print("Detecting segment boundaries...")
    
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
    segment_list = []
    for i in range(len(segment_times) - 1):
        segment_list.append({
            "start": segment_times[i],
            "end": segment_times[i + 1],
            "label": f"Section {i + 1}"
        })
    
    # Find repeating sections
    print("Finding repeating sections...")
    repeating_sections = []
    
    # Compare each segment with every other segment
    for i in range(len(segment_list)):
        for j in range(i + 1, len(segment_list)):
            # Get feature vectors for each segment
            start_frame_i = librosa.time_to_frames(segment_list[i]["start"], sr=sr)
            end_frame_i = librosa.time_to_frames(segment_list[i]["end"], sr=sr)
            
            start_frame_j = librosa.time_to_frames(segment_list[j]["start"], sr=sr)
            end_frame_j = librosa.time_to_frames(segment_list[j]["end"], sr=sr)
            
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
                        "section1": segment_list[i]["label"],
                        "section2": segment_list[j]["label"],
                        "similarity": float(similarity)
                    })
    
    # Prepare results
    results = {
        "audio_path": str(audio_path),
        "sample_rate": sr,
        "duration": duration,
        "segments": segment_list,
        "segment_count": len(segment_list),
        "repeating_sections": repeating_sections,
        "repeating_section_count": len(repeating_sections)
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Audio segmentation using librosa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python librosa_segmentation.py audio.wav
  python librosa_segmentation.py audio.mp3 --output ./results
        """
    )
    
    parser.add_argument(
        "audio_file",
        help="Path to the audio file to segment"
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
            json_path = output_dir / f"{audio_path.stem}_librosa_segmentation.json"
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {json_path}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()