#!/usr/bin/env python3
"""
Standalone script to analyze BPM and beats using librosa.
Based on: https://librosa.org/doc/0.11.0/generated/librosa.beat.beat_track.html
"""

import sys
import argparse
import librosa
import librosa.display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path


def analyze_structure(audio_path, output_dir=None, plot=False):
    """
    Analyze BPM and beats using librosa.
    
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
    
    # Load the audio file
    y, sr = librosa.load(str(audio_path), sr=None)
    duration = len(y) / sr
    print(f"Audio duration: {duration:.2f} seconds")
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
    
    print("\nDetecting downbeats and structure...")
    
    # Perform downbeat detection
    try:
        # Compute harmonic-percussive source separation
        y_harmonic = librosa.effects.harmonic(y)
        
        # Compute chroma features from the harmonic component
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        
        # Find downbeats by analyzing chroma changes and beat positions
        downbeat_estimates = []
        if len(beat_times) > 0:
            # Simple heuristic: look for significant chroma changes at beat positions
            chroma_changes = np.diff(chroma, axis=1)
            beat_frames = librosa.time_to_frames(beat_times, sr=sr)
            
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
                    if idx < len(beat_times) - 1:
                        downbeat_estimates.append(beat_times[idx + 1])
            
            # Also add the first beat as a downbeat
            if len(beat_times) > 0:
                downbeat_estimates.insert(0, beat_times[0])
        
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
    
    except Exception as e:
        print(f"Warning: Downbeat detection failed: {e}")
        downbeat_estimates = []
        segments = []
        repeating_sections = []
    
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
        'downbeat_times': downbeat_estimates,
        'segments': segments,
        'repeating_sections': repeating_sections,
    }
    
    # Print results
    print(f"\nResults:")
    print(f"  Tempo (BPM): {tempo_value:.2f}")
    print(f"  Beat count: {len(beats)}")
    print(f"  Downbeat count: {len(downbeat_estimates)}")
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
        json_path = output_dir / f"{Path(audio_path).stem}_beat_analysis.json"
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
        plt.figure(figsize=(14, 10))
        
        # Plot waveform
        plt.subplot(4, 1, 1)
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.title('Waveform')
        plt.ylabel('Amplitude')
        
        # Plot beat and downbeat times on waveform
        plt.subplot(4, 1, 2)
        librosa.display.waveshow(y, sr=sr, alpha=0.6)
        plt.vlines(beat_times, -1, 1, color='r', alpha=0.8, linestyle='--', label='Beats')
        if downbeat_estimates:
            plt.vlines(downbeat_estimates, -1, 1, color='b', alpha=0.8, linestyle='-', linewidth=2, label='Downbeats')
        plt.title('Waveform with Beat and Downbeat Positions')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Plot structure sections
        plt.subplot(4, 1, 3)
        librosa.display.waveshow(y, sr=sr, alpha=0.3)
        
        # Color code different sections
        colors = plt.cm.Set3(np.linspace(0, 1, len(segments)))
        for i, seg in enumerate(segments):
            plt.axvspan(seg['start'], seg['end'], alpha=0.3, color=colors[i], label=seg['label'])
        
        plt.title('Track Structure')
        plt.ylabel('Amplitude')
        if len(segments) <= 8:  # Only show legend if not too many sections
            plt.legend(loc='upper right')
        
        # Plot beat intervals
        if len(beat_times) > 1:
            plt.subplot(4, 1, 4)
            plt.plot(beat_times[:-1], beat_intervals, 'o-', color='g')
            plt.axhline(y=avg_interval, color='r', linestyle='--', label=f'Average: {avg_interval:.3f}s')
            plt.title('Beat Intervals')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Interval (seconds)')
            plt.legend()
        
        plt.tight_layout()
        
        if output_dir:
            plot_path = output_dir / f"{Path(audio_path).stem}_beat_analysis.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Visualization saved to: {plot_path}")
        else:
            plt.show()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BPM, beats, downbeats, and track structure using librosa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python librosa_beat_analysis.py audio.wav
  python librosa_beat_analysis.py audio.mp3 --output ./results
  python librosa_beat_analysis.py audio.wav --plot --output ./results
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