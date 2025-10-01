#!/usr/bin/env python3
"""
Script to compare beat and downbeat analysis results between librosa and madmom.
"""

import sys
import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Import our analysis scripts
from librosa_beat_analysis import analyze_structure as librosa_analyze
from madmom_beat_analysis import analyze_structure as madmom_analyze


def compare_analysis(audio_path, output_dir=None, plot=False):
    """
    Compare beat and downbeat analysis results between librosa and madmom.
    
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
        Dictionary containing the comparison results
    """
    print(f"Analyzing audio file with both libraries: {audio_path}")
    
    # Run analysis with librosa
    print("\n=== Running librosa analysis ===")
    librosa_results = librosa_analyze(audio_path, output_dir=None, plot=False)
    
    # Run analysis with madmom
    print("\n=== Running madmom analysis ===")
    madmom_results = madmom_analyze(audio_path, output_dir=None, plot=False)
    
    # Compare results
    print("\n=== Comparing results ===")
    
    # Tempo comparison
    librosa_tempo = librosa_results['tempo']
    madmom_tempo = madmom_results['tempo']
    tempo_diff = abs(librosa_tempo - madmom_tempo)
    tempo_percent_diff = (tempo_diff / ((librosa_tempo + madmom_tempo) / 2)) * 100
    
    print(f"Tempo comparison:")
    print(f"  librosa: {librosa_tempo:.2f} BPM")
    print(f"  madmom:  {madmom_tempo:.2f} BPM")
    print(f"  Difference: {tempo_diff:.2f} BPM ({tempo_percent_diff:.1f}%)")
    
    # Beat count comparison
    librosa_beat_count = librosa_results['beat_count']
    madmom_beat_count = madmom_results['beat_count']
    beat_count_diff = abs(librosa_beat_count - madmom_beat_count)
    
    print(f"\nBeat count comparison:")
    print(f"  librosa: {librosa_beat_count} beats")
    print(f"  madmom:  {madmom_beat_count} beats")
    print(f"  Difference: {beat_count_diff} beats")
    
    # Downbeat count comparison
    librosa_downbeat_count = librosa_results['downbeat_count']
    madmom_downbeat_count = madmom_results['downbeat_count']
    downbeat_count_diff = abs(librosa_downbeat_count - madmom_downbeat_count)
    
    print(f"\nDownbeat count comparison:")
    print(f"  librosa: {librosa_downbeat_count} downbeats")
    print(f"  madmom:  {madmom_downbeat_count} downbeats")
    print(f"  Difference: {downbeat_count_diff} downbeats")
    
    # Beat consistency comparison
    librosa_consistency = librosa_results['beat_consistency']
    madmom_consistency = madmom_results['beat_consistency']
    
    print(f"\nBeat consistency comparison:")
    print(f"  librosa: {librosa_consistency:.2f}")
    print(f"  madmom:  {madmom_consistency:.2f}")
    
    # Structure sections comparison
    librosa_sections = len(librosa_results['segments'])
    madmom_sections = len(madmom_results['segments'])
    sections_diff = abs(librosa_sections - madmom_sections)
    
    print(f"\nStructure sections comparison:")
    print(f"  librosa: {librosa_sections} sections")
    print(f"  madmom:  {madmom_sections} sections")
    print(f"  Difference: {sections_diff} sections")
    
    # Repeating sections comparison
    librosa_repeating = len(librosa_results['repeating_sections'])
    madmom_repeating = len(madmom_results['repeating_sections'])
    repeating_diff = abs(librosa_repeating - madmom_repeating)
    
    print(f"\nRepeating sections comparison:")
    print(f"  librosa: {librosa_repeating} repeating sections")
    print(f"  madmom:  {madmom_repeating} repeating sections")
    print(f"  Difference: {repeating_diff} repeating sections")
    
    # Beat position comparison
    librosa_beats = np.array(librosa_results['beat_times'])
    madmom_beats = np.array(madmom_results['beat_times'])
    
    # Find closest beats between the two methods
    beat_position_diffs = []
    for lb in librosa_beats:
        closest_mb = madmom_beats[np.argmin(np.abs(madmom_beats - lb))]
        beat_position_diffs.append(abs(lb - closest_mb))
    
    avg_beat_position_diff = np.mean(beat_position_diffs) if beat_position_diffs else 0
    
    print(f"\nBeat position comparison:")
    print(f"  Average difference: {avg_beat_position_diff:.3f} seconds")
    
    # Downbeat position comparison
    if librosa_results['downbeat_times'] and madmom_results['downbeat_times']:
        librosa_downbeats = np.array(librosa_results['downbeat_times'])
        madmom_downbeats = np.array(madmom_results['downbeat_times'])
        
        # Find closest downbeats between the two methods
        downbeat_position_diffs = []
        for ldb in librosa_downbeats:
            closest_mdb = madmom_downbeats[np.argmin(np.abs(madmom_downbeats - ldb))]
            downbeat_position_diffs.append(abs(ldb - closest_mdb))
        
        avg_downbeat_position_diff = np.mean(downbeat_position_diffs) if downbeat_position_diffs else 0
        
        print(f"\nDownbeat position comparison:")
        print(f"  Average difference: {avg_downbeat_position_diff:.3f} seconds")
    else:
        avg_downbeat_position_diff = 0
        print(f"\nDownbeat position comparison: Not available (one method didn't detect downbeats)")
    
    # Prepare comparison results
    comparison_results = {
        'audio_path': str(audio_path),
        'librosa_results': librosa_results,
        'madmom_results': madmom_results,
        'tempo_comparison': {
            'librosa_tempo': librosa_tempo,
            'madmom_tempo': madmom_tempo,
            'difference': tempo_diff,
            'percent_difference': tempo_percent_diff
        },
        'beat_count_comparison': {
            'librosa_beat_count': librosa_beat_count,
            'madmom_beat_count': madmom_beat_count,
            'difference': beat_count_diff
        },
        'downbeat_count_comparison': {
            'librosa_downbeat_count': librosa_downbeat_count,
            'madmom_downbeat_count': madmom_downbeat_count,
            'difference': downbeat_count_diff
        },
        'beat_consistency_comparison': {
            'librosa_consistency': librosa_consistency,
            'madmom_consistency': madmom_consistency
        },
        'structure_sections_comparison': {
            'librosa_sections': librosa_sections,
            'madmom_sections': madmom_sections,
            'difference': sections_diff
        },
        'repeating_sections_comparison': {
            'librosa_repeating': librosa_repeating,
            'madmom_repeating': madmom_repeating,
            'difference': repeating_diff
        },
        'beat_position_comparison': {
            'average_difference': avg_beat_position_diff
        },
        'downbeat_position_comparison': {
            'average_difference': avg_downbeat_position_diff
        }
    }
    
    # Save results to JSON if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        json_path = output_dir / f"{Path(audio_path).stem}_comparison.json"
        with open(json_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"\nComparison results saved to: {json_path}")
    
    # Create visualization if requested
    if plot:
        print("\nCreating comparison visualization...")
        try:
            plt.figure(figsize=(16, 12))
            
            # Plot beat positions
            plt.subplot(3, 1, 1)
            plt.scatter(librosa_beats, np.ones_like(librosa_beats), 
                        color='blue', alpha=0.7, label='librosa beats')
            plt.scatter(madmom_beats, np.ones_like(madmom_beats) * 0.95, 
                        color='red', alpha=0.7, label='madmom beats')
            plt.title('Beat Position Comparison')
            plt.xlabel('Time (seconds)')
            plt.yticks([])
            plt.legend()
            plt.xlim(0, max(librosa_results['duration'], madmom_results['duration']))
            
            # Plot downbeat positions
            if librosa_results['downbeat_times'] and madmom_results['downbeat_times']:
                plt.subplot(3, 1, 2)
                plt.scatter(librosa_results['downbeat_times'], np.ones_like(librosa_results['downbeat_times']), 
                            color='blue', alpha=0.7, label='librosa downbeats')
                plt.scatter(madmom_results['downbeat_times'], np.ones_like(madmom_results['downbeat_times']) * 0.95, 
                            color='red', alpha=0.7, label='madmom downbeats')
                plt.title('Downbeat Position Comparison')
                plt.xlabel('Time (seconds)')
                plt.yticks([])
                plt.legend()
                plt.xlim(0, max(librosa_results['duration'], madmom_results['duration']))
            
            # Plot structure comparison
            plt.subplot(3, 1, 3)
            
            # Plot librosa segments
            for i, seg in enumerate(librosa_results['segments']):
                plt.axvspan(seg['start'], seg['end'], alpha=0.3, 
                           color='blue', label='librosa' if i == 0 else "")
            
            # Plot madmom segments
            for i, seg in enumerate(madmom_results['segments']):
                plt.axvspan(seg['start'], seg['end'], alpha=0.3, 
                           color='red', label='madmom' if i == 0 else "")
            
            plt.title('Structure Comparison')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.xlim(0, max(librosa_results['duration'], madmom_results['duration']))
            
            plt.tight_layout()
            
            if output_dir:
                plot_path = output_dir / f"{Path(audio_path).stem}_comparison.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"Comparison visualization saved to: {plot_path}")
            else:
                plt.show()
        except Exception as e:
            print(f"Warning: Visualization failed: {e}")
    
    return comparison_results


def main():
    parser = argparse.ArgumentParser(
        description="Compare beat and downbeat analysis results between librosa and madmom",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_beat_analysis.py audio.wav
  python compare_beat_analysis.py audio.mp3 --output ./results
  python compare_beat_analysis.py audio.wav --plot --output ./results
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
        # Run the comparison
        results = compare_analysis(
            audio_path=audio_path,
            output_dir=args.output,
            plot=args.plot
        )
        
        print("\nComparison completed successfully!")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()