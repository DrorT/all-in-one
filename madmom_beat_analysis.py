#!/usr/bin/env python3
"""
Standalone script to analyze BPM, beats, and downbeats using madmom.
Based on: https://madmom.readthedocs.io/en/latest/modules/audio/beat_tracking.html
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path
import madmom
from madmom.io.audio import load_audio_file
from madmom.audio.signal import SignalProcessor
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.audio.spectrogram import SpectrogramProcessor, FilteredSpectrogramProcessor
from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor


def analyze_structure(audio_path, output_dir=None, plot=False):
    """
    Analyze BPM, beats, and downbeats using madmom.
    
    Parameters
    ----------
    audio_path : str or Path
        Path to the audio file
    output_dir : str or Path, optional
        Directory to save the results
    plot : bool, optional
        Whether to create and save a visualization
        
    Returns
    -------
    dict
        Dictionary containing the analysis results
    """
    print(f"Loading audio file: {audio_path}")
    
    # Load the audio file using madmom
    signal, sample_rate = load_audio_file(str(audio_path))
    duration = len(signal) / sample_rate
    print(f"Audio duration: {duration:.2f} seconds")
    print(f"Sample rate: {sample_rate} Hz")
    
    # Beat tracking with madmom
    print("\nAnalyzing beats...")
    beat_processor = RNNBeatProcessor()
    beat_activations = beat_processor(signal)
    
    # Track beats
    beat_tracker = BeatTrackingProcessor(fps=100)
    beats = beat_tracker(beat_activations)
    
    # Convert beat frames to time (madmom already returns times in seconds)
    beat_times = beats
    
    # Calculate tempo from beat times
    if len(beat_times) > 1:
        # Calculate beat intervals
        beat_intervals = np.diff(beat_times)
        avg_interval = np.mean(beat_intervals)
        
        # Calculate tempo in BPM
        tempo_value = 60.0 / avg_interval if avg_interval > 0 else 0.0
        
        # Calculate beat consistency (standard deviation of intervals)
        beat_consistency = 1.0 - (np.std(beat_intervals) / avg_interval)
        beat_consistency = max(0.0, min(1.0, beat_consistency))  # Clamp between 0 and 1
    else:
        avg_interval = 0
        tempo_value = 0.0
        beat_consistency = 0
    
    # Downbeat tracking with madmom
    print("\nAnalyzing downbeats...")
    try:
        downbeat_processor = RNNDownBeatProcessor()
        downbeat_activations = downbeat_processor(signal)
        
        # Track downbeats
        downbeat_tracker = DBNDownBeatTrackingProcessor(fps=100, beats_per_bar=[4, 3])  # Common time signatures
        downbeats = downbeat_tracker(downbeat_activations)
        
        # Extract downbeat times (first column is time, second is beat number)
        downbeat_times = downbeats[:, 0] if len(downbeats) > 0 else np.array([])
        
        # Extract beat numbers (1-based, where 1 is downbeat)
        beat_numbers = downbeats[:, 1] if len(downbeats) > 0 else np.array([])
        
        # Find just the downbeats (where beat number is 1)
        downbeat_indices = np.where(beat_numbers == 1)[0]
        downbeat_times_only = downbeat_times[downbeat_indices] if len(downbeat_indices) > 0 else []
        
    except Exception as e:
        print(f"Warning: Downbeat detection failed: {e}")
        downbeat_times = []
        downbeat_times_only = []
    
    # Structure analysis using beat and downbeat patterns
    print("\nAnalyzing structure...")
    segments = []
    repeating_sections = []
    
    try:
        if len(downbeat_times_only) > 1:
            # Create segments based on downbeat patterns
            # Look for patterns in downbeat intervals to identify sections
            downbeat_intervals = np.diff(downbeat_times_only)
            
            # Group consecutive downbeats with similar intervals
            current_segment_start = 0.0
            current_pattern = []
            segments = []
            
            for i, interval in enumerate(downbeat_intervals):
                current_pattern.append(interval)
                
                # Check if we have enough beats to detect a pattern (4 bars)
                if len(current_pattern) >= 4:
                    # Check if the pattern is consistent (all intervals similar)
                    pattern_std = np.std(current_pattern)
                    pattern_mean = np.mean(current_pattern)
                    
                    # If pattern is consistent, consider it a section
                    if pattern_std < 0.1 * pattern_mean:  # 10% tolerance
                        segment_end = downbeat_times_only[i + 1]
                        segments.append({
                            'start': current_segment_start,
                            'end': segment_end,
                            'label': f'Section {len(segments)+1}'
                        })
                        current_segment_start = segment_end
                        current_pattern = []
            
            # Add the last segment if needed
            if current_segment_start < duration:
                segments.append({
                    'start': current_segment_start,
                    'end': duration,
                    'label': f'Section {len(segments)+1}'
                })
            
            # If no segments were detected, create a simple structure
            if not segments:
                segments = [{
                    'start': 0.0,
                    'end': duration,
                    'label': 'Section 1'
                }]
            
            # Look for repeating sections by comparing beat patterns
            if len(segments) > 1:
                # For simplicity, we'll just mark sections with similar durations as potentially repeating
                for i in range(len(segments)):
                    for j in range(i + 1, len(segments)):
                        duration_i = segments[i]['end'] - segments[i]['start']
                        duration_j = segments[j]['end'] - segments[j]['start']
                        
                        # If durations are similar within 10%, consider them repeating
                        if abs(duration_i - duration_j) / max(duration_i, duration_j) < 0.1:
                            repeating_sections.append({
                                'section1': segments[i]['label'],
                                'section2': segments[j]['label'],
                                'similarity': 0.8  # Fixed similarity for this simple approach
                            })
        else:
            # No downbeats detected, create a single segment
            segments = [{
                'start': 0.0,
                'end': duration,
                'label': 'Section 1'
            }]
    
    except Exception as e:
        print(f"Warning: Structure analysis failed: {e}")
        segments = [{
            'start': 0.0,
            'end': duration,
            'label': 'Section 1'
        }]
        repeating_sections = []
    
    # Prepare results
    results = {
        'audio_path': str(audio_path),
        'sample_rate': sample_rate,
        'duration': duration,
        'tempo': tempo_value,
        'beat_count': len(beat_times),
        'beat_times': beat_times.tolist(),
        'beat_intervals': beat_intervals.tolist() if len(beat_times) > 1 else [],
        'avg_beat_interval': float(avg_interval),
        'beat_consistency': float(beat_consistency),
        'downbeat_count': len(downbeat_times_only),
        'downbeat_times': downbeat_times_only.tolist(),
        'segments': segments,
        'repeating_sections': repeating_sections,
    }
    
    # Print results
    print(f"\nResults:")
    print(f"  Tempo (BPM): {tempo_value:.2f}")
    print(f"  Beat count: {len(beat_times)}")
    print(f"  Downbeat count: {len(downbeat_times_only)}")
    print(f"  Average beat interval: {avg_interval:.3f} seconds")
    print(f"  Beat consistency: {beat_consistency:.2f} (1.0 = perfectly consistent)")
    print(f"  Structure sections: {len(segments)}")
    print(f"  Repeating sections: {len(repeating_sections)}")
    
    if repeating_sections:
        print("\n  Repeating sections found:")
        for repeat in repeating_sections:
            print(f"    {repeat['section1']} ↔ {repeat['section2']} (similarity: {repeat['similarity']:.2f})")
    
    # Save results to JSON if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = output_dir / f"{Path(audio_path).stem}_madmom_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")
        
        # Save beat times as text file
        beats_path = output_dir / f"{Path(audio_path).stem}_beats.txt"
        with open(beats_path, 'w') as f:
            f.write("# Beat times (seconds)\n")
            for beat_time in beat_times:
                f.write(f"{beat_time:.4f}\n")
        print(f"Beat times saved to: {beats_path}")
        
        # Save downbeat times as text file
        if downbeat_times_only:
            downbeats_path = output_dir / f"{Path(audio_path).stem}_downbeats.txt"
            with open(downbeats_path, 'w') as f:
                f.write("# Downbeat times (seconds)\n")
                for downbeat_time in downbeat_times_only:
                    f.write(f"{downbeat_time:.4f}\n")
            print(f"Downbeat times saved to: {downbeats_path}")
        
        # Save structure information as text file
        if segments:
            structure_path = output_dir / f"{Path(audio_path).stem}_structure.txt"
            with open(structure_path, 'w') as f:
                f.write("# Track structure\n")
                f.write("# Format: start_time end_time section_label\n")
                for seg in segments:
                    f.write(f"{seg['start']:.2f} {seg['end']:.2f} {seg['label']}\n")
            print(f"Structure information saved to: {structure_path}")
    
    # Create visualization if requested
    if plot:
        print("\nCreating visualization...")
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(14, 10))
            
            # Plot waveform
            plt.subplot(4, 1, 1)
            time_axis = np.arange(len(signal)) / sample_rate
            plt.plot(time_axis, signal)
            plt.title('Waveform')
            plt.ylabel('Amplitude')
            plt.xlim(0, duration)
            
            # Plot beat and downbeat times on waveform
            plt.subplot(4, 1, 2)
            plt.plot(time_axis, signal)
            plt.vlines(beat_times, min(signal), max(signal), color='r', alpha=0.8, linestyle='--', label='Beats')
            if downbeat_times_only:
                plt.vlines(downbeat_times_only, min(signal), max(signal), color='b', alpha=0.8, linestyle='-', linewidth=2, label='Downbeats')
            plt.title('Waveform with Beat and Downbeat Positions')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.xlim(0, duration)
            
            # Plot structure sections
            plt.subplot(4, 1, 3)
            plt.plot(time_axis, signal, alpha=0.3)
            
            # Color code different sections
            colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
            for i, seg in enumerate(segments):
                plt.axvspan(seg['start'], seg['end'], alpha=0.3, color=colors[i], label=seg['label'])
            
            plt.title('Track Structure')
            plt.ylabel('Amplitude')
            if len(segments) <= 8:  # Only show legend if not too many sections
                plt.legend(loc='upper right')
            plt.xlim(0, duration)
            
            # Plot beat intervals
            if len(beat_times) > 1:
                plt.subplot(4, 1, 4)
                plt.plot(beat_times[:-1], beat_intervals, 'o-', color='g')
                plt.axhline(y=avg_interval, color='r', linestyle='--', label=f'Average: {avg_interval:.3f}s')
                plt.title('Beat Intervals')
                plt.xlabel('Time (seconds)')
                plt.ylabel('Interval (seconds)')
                plt.legend()
                plt.xlim(0, duration)
            
            plt.tight_layout()
            
            if output_dir:
                plot_path = output_dir / f"{Path(audio_path).stem}_madmom_analysis.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Visualization saved to: {plot_path}")
            else:
                plt.show()
        except ImportError:
            print("Warning: matplotlib not available for visualization")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BPM, beats, downbeats, and track structure using madmom",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python madmom_beat_analysis.py audio.wav
  python madmom_beat_analysis.py audio.mp3 --output ./results
  python madmom_beat_analysis.py audio.wav --plot --output ./results
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
    
    parser.add_argument(
        "--plot", "-p",
        action="store_true",
        help="Create and save a visualization"
    )
    
    args = parser.parse_args()
    
    # Check if the audio file exists
    audio_path = Path(args.audio_file)
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    try:
        # Run the analysis
        results = analyze_structure(
            audio_path=audio_path,
            output_dir=args.output,
            plot=args.plot
        )
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()