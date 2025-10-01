#!/usr/bin/env python3
"""
Test script to demonstrate the improvement in Madmom BPM calculation
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.allin1.comprehensive_analysis import MadmomAnalyzer

def test_tempo_calculation(audio_path):
    """Test the improved tempo calculation on an audio file"""
    print(f"Testing tempo calculation on: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return
    
    # Initialize Madmom analyzer
    try:
        analyzer = MadmomAnalyzer()
    except ImportError as e:
        print(f"Madmom not available: {e}")
        return
    
    # Extract features with the improved tempo calculation
    features = analyzer.extract_beats_and_downbeats(audio_path)
    
    if features is None:
        print("Failed to extract features")
        return
    
    print(f"\nResults for {Path(audio_path).name}:")
    print(f"Number of beats detected: {len(features.beats)}")
    print(f"Calculated tempo: {features.tempo:.2f} BPM")
    print(f"Beat consistency: {features.beat_consistency:.3f}")
    
    if len(features.beats) > 1:
        beat_intervals = np.diff(features.beats)
        print(f"Beat interval stats:")
        print(f"  Mean: {np.mean(beat_intervals):.3f}s")
        print(f"  Median: {np.median(beat_intervals):.3f}s")
        print(f"  Std: {np.std(beat_intervals):.3f}s")
        print(f"  Min: {np.min(beat_intervals):.3f}s")
        print(f"  Max: {np.max(beat_intervals):.3f}s")
    
    # Create a simple visualization of beat intervals
    if len(features.beats) > 1:
        plt.figure(figsize=(12, 6))
        
        # Plot 1: Beat positions
        plt.subplot(2, 1, 1)
        plt.scatter(features.beats, np.ones_like(features.beats), alpha=0.6, s=50)
        plt.xlabel('Time (seconds)')
        plt.title(f'Beat Positions - {Path(audio_path).name}')
        plt.yticks([])
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Beat intervals
        plt.subplot(2, 1, 2)
        beat_intervals = np.diff(features.beats)
        beat_times_mid = (features.beats[:-1] + features.beats[1:]) / 2
        plt.plot(beat_times_mid, beat_intervals, 'o-', alpha=0.7)
        plt.axhline(y=60.0/features.tempo, color='r', linestyle='--', 
                   label=f'Average interval ({60.0/features.tempo:.3f}s)')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Beat Interval (seconds)')
        plt.title('Beat Intervals Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = f"tempo_analysis_{Path(audio_path).stem}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {output_path}")
        plt.close()

def compare_with_simple_method(audio_path):
    """Compare the improved method with the original simple method"""
    print(f"\nComparing tempo calculation methods on: {audio_path}")
    
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return
    
    # Initialize Madmom analyzer
    try:
        analyzer = MadmomAnalyzer()
    except ImportError as e:
        print(f"Madmom not available: {e}")
        return
    
    # Extract features
    features = analyzer.extract_beats_and_downbeats(audio_path)
    
    if features is None or len(features.beats) <= 1:
        print("Not enough beats for comparison")
        return
    
    # Calculate tempo using the original simple method
    beat_intervals = np.diff(features.beats)
    simple_avg_interval = np.mean(beat_intervals)
    simple_tempo = 60.0 / simple_avg_interval if simple_avg_interval > 0 else 0.0
    
    # Calculate tempo using median (part of improved method)
    median_interval = np.median(beat_intervals)
    median_tempo = 60.0 / median_interval if median_interval > 0 else 0.0
    
    print(f"Original simple method: {simple_tempo:.2f} BPM")
    print(f"Median-based method: {median_tempo:.2f} BPM")
    print(f"Improved method result: {features.tempo:.2f} BPM")
    print(f"Difference (simple vs improved): {abs(simple_tempo - features.tempo):.2f} BPM")
    print(f"Difference (median vs improved): {abs(median_tempo - features.tempo):.2f} BPM")
    
    # Create comparison visualization
    plt.figure(figsize=(10, 6))
    
    # Plot histogram of beat intervals
    plt.hist(beat_intervals, bins=30, alpha=0.7, density=True, label='Beat Intervals')
    
    # Mark the different tempo estimates
    plt.axvline(x=simple_avg_interval, color='red', linestyle='--', linewidth=2,
               label=f'Simple Mean ({simple_tempo:.1f} BPM)')
    plt.axvline(x=median_interval, color='green', linestyle='--', linewidth=2,
               label=f'Median ({median_tempo:.1f} BPM)')
    plt.axvline(x=60.0/features.tempo, color='blue', linestyle='-', linewidth=2,
               label=f'Improved Method ({features.tempo:.1f} BPM)')
    
    plt.xlabel('Beat Interval (seconds)')
    plt.ylabel('Density')
    plt.title(f'Beat Interval Distribution - {Path(audio_path).name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the comparison plot
    output_path = f"tempo_comparison_{Path(audio_path).stem}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison plot saved to: {output_path}")
    plt.close()

if __name__ == "__main__":
    # Find audio files in the current directory or common locations
    audio_files = []
    
    # Check for common audio file extensions
    for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a']:
        audio_files.extend(Path('.').glob(ext))
        audio_files.extend(Path('test_audio').glob(ext))
        audio_files.extend(Path('audio').glob(ext))
    
    if not audio_files:
        print("No audio files found. Please specify an audio file path.")
        print("Usage: python test_madmom_tempo_improvement.py [audio_file_path]")
        
        # Try to find any audio file in the project
        for root, dirs, files in os.walk('.'):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    audio_files.append(Path(root) / file)
                    if len(audio_files) >= 3:  # Limit to 3 files for testing
                        break
            if len(audio_files) >= 3:
                break
    
    if audio_files:
        print(f"Found {len(audio_files)} audio file(s) for testing")
        
        # Test the first few files
        for i, audio_file in enumerate(audio_files[:3]):
            print(f"\n{'='*60}")
            print(f"Testing file {i+1}: {audio_file}")
            print('='*60)
            
            test_tempo_calculation(audio_file)
            compare_with_simple_method(audio_file)
    else:
        print("No audio files found for testing.")
        print("Please place an audio file in the current directory or specify a path.")