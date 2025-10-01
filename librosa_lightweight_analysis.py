#!/usr/bin/env python3
"""
Lightweight version of librosa beat analysis that avoids CPU-intensive operations.
"""

import sys
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def analyze_beats_lightweight(audio_path, output_dir=None, plot=False):
    """
    Analyze BPM and beats using librosa with lightweight operations.
    
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
    
    # Load the audio file with optional duration limit for very long files
    y, sr = librosa.load(str(audio_path), sr=None, duration=180)  # Limit to 3 minutes
    duration = len(y) / sr
    print(f"Audio duration: {duration:.2f} seconds (limited to 3 minutes)")
    print(f"Sample rate: {sr} Hz")
    
    # Track beats using librosa
    print("\nAnalyzing beats...")
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Convert beat frames to time
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Handle tempo which might be an array or scalar
    if isinstance(tempo, np.ndarray):
        tempo_value = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo_value = float(tempo)
    
    # Calculate additional metrics
    if len(beat_times) > 1:
        # Calculate average beat interval
        beat_intervals = np.diff(beat_times)
        avg_interval = np.mean(beat_intervals)
        
        # Calculate beat consistency (standard deviation of intervals)
        beat_consistency = 1.0 - (np.std(beat_intervals) / avg_interval)
        beat_consistency = max(0.0, min(1.0, beat_consistency))  # Clamp between 0 and 1
    else:
        avg_interval = 0
        beat_consistency = 0
    
    # Simple downbeat detection (lightweight version)
    print("\nDetecting downbeats (lightweight method)...")
    downbeat_estimates = []
    
    try:
        # Simple heuristic: assume 4/4 time signature
        # Every 4th beat is a downbeat
        if len(beat_times) >= 4:
            downbeat_estimates = beat_times[::4]  # Take every 4th beat
            # Always include the first beat
            if beat_times[0] not in downbeat_estimates:
                downbeat_estimates = np.insert(downbeat_estimates, 0, beat_times[0])
    except Exception as e:
        print(f"Warning: Downbeat detection failed: {e}")
    
    # Simple structure detection (lightweight version)
    print("\nDetecting structure (lightweight method)...")
    segments = []
    
    try:
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
    except Exception as e:
        print(f"Warning: Structure detection failed: {e}")
    
    # Prepare results
    results = {
        'audio_path': str(audio_path),
        'sample_rate': sr,
        'duration': duration,
        'tempo': tempo_value,
        'beat_count': len(beats),
        'beat_times': beat_times.tolist(),
        'beat_intervals': beat_intervals.tolist() if len(beat_times) > 1 else [],
        'avg_beat_interval': float(avg_interval),
        'beat_consistency': float(beat_consistency),
        'downbeat_count': len(downbeat_estimates),
        'downbeat_times': downbeat_estimates.tolist() if len(downbeat_estimates) > 0 else [],
        'segments': segments,
        'analysis_type': 'lightweight'
    }
    
    # Print results
    print(f"\nResults:")
    print(f"  Tempo (BPM): {tempo_value:.2f}")
    print(f"  Beat count: {len(beats)}")
    print(f"  Downbeat count: {len(downbeat_estimates)}")
    print(f"  Average beat interval: {avg_interval:.3f} seconds")
    print(f"  Beat consistency: {beat_consistency:.2f} (1.0 = perfectly consistent)")
    print(f"  Structure sections: {len(segments)}")
    
    # Save results to JSON if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = output_dir / f"{Path(audio_path).stem}_lightweight_analysis.json"
        import json
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
        if downbeat_estimates:
            downbeats_path = output_dir / f"{Path(audio_path).stem}_downbeats.txt"
            with open(downbeats_path, 'w') as f:
                f.write("# Downbeat times (seconds)\n")
                for downbeat_time in downbeat_estimates:
                    f.write(f"{downbeat_time:.4f}\n")
            print(f"Downbeat times saved to: {downbeats_path}")
    
    # Create visualization if requested
    if plot:
        print("\nCreating visualization...")
        plt.figure(figsize=(12, 8))
        
        # Plot waveform
        plt.subplot(3, 1, 1)
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title('Waveform')
        plt.ylabel('Amplitude')
        
        # Plot beat and downbeat times on waveform
        plt.subplot(3, 1, 2)
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.vlines(beat_times, -1, 1, color='r', alpha=0.8, linestyle='--', label='Beats')
        if downbeat_estimates:
            plt.vlines(downbeat_estimates, -1, 1, color='b', alpha=0.8, linestyle='-', linewidth=2, label='Downbeats')
        plt.title('Waveform with Beat and Downbeat Positions')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Plot simple structure sections
        plt.subplot(3, 1, 3)
        librosa.display.waveshow(y, sr=sr, alpha=0.3)
        
        # Color code different sections
        colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
        for i, seg in enumerate(segments):
            plt.axvspan(seg['start'], seg['end'], alpha=0.3, color=colors[i], label=seg['label'])
        
        plt.title('Track Structure (Simple Time-based Segmentation)')
        plt.ylabel('Amplitude')
        if len(segments) <= 8:  # Only show legend if not too many sections
            plt.legend(loc='upper right')
        
        plt.tight_layout()
        
        if output_dir:
            plot_path = output_dir / f"{Path(audio_path).stem}_lightweight_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {plot_path}")
        else:
            plt.show()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight analysis of BPM, beats, downbeats, and simple structure using librosa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python librosa_lightweight_analysis.py audio.wav
  python librosa_lightweight_analysis.py audio.mp3 --output ./results
  python librosa_lightweight_analysis.py audio.wav --plot --output ./results
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
        # Run the lightweight analysis
        results = analyze_beats_lightweight(
            audio_path=audio_path,
            output_dir=args.output,
            plot=args.plot
        )
        
        print("\nLightweight analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()