#!/home/chester/venvs/pydemucs/bin/python
"""
Beat Detection Comparison Test
Compares Madmom vs beat_this for beat and downbeat detection.
Measures accuracy and performance (execution time).
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from allin1.comprehensive_analysis import (
    MadmomAnalyzer, 
    BeatThisAnalyzer,
    MADMOM_AVAILABLE,
    BEAT_THIS_AVAILABLE
)


def calculate_beat_metrics(predicted: np.ndarray, reference: np.ndarray, tolerance: float = 0.07) -> Dict[str, float]:
    """
    Calculate accuracy metrics for beat detection.
    
    Parameters
    ----------
    predicted : np.ndarray
        Predicted beat times in seconds
    reference : np.ndarray
        Reference beat times in seconds
    tolerance : float
        Tolerance window in seconds for matching beats (default: 0.07s = 70ms)
        
    Returns
    -------
    Dict[str, float]
        Dictionary with precision, recall, f1_score metrics
    """
    if len(predicted) == 0 or len(reference) == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'num_predicted': len(predicted),
            'num_reference': len(reference),
            'num_matched': 0
        }
    
    # Count true positives (predicted beats that match reference beats within tolerance)
    true_positives = 0
    matched_ref = set()
    
    for pred_beat in predicted:
        # Find closest reference beat
        distances = np.abs(reference - pred_beat)
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        # If within tolerance and not already matched, count as true positive
        if min_distance <= tolerance and min_idx not in matched_ref:
            true_positives += 1
            matched_ref.add(min_idx)
    
    # Calculate metrics
    precision = true_positives / len(predicted) if len(predicted) > 0 else 0.0
    recall = true_positives / len(reference) if len(reference) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1_score),
        'num_predicted': int(len(predicted)),
        'num_reference': int(len(reference)),
        'num_matched': int(true_positives)
    }


def calculate_tempo_error(predicted_tempo: float, reference_tempo: float) -> Dict[str, float]:
    """Calculate tempo estimation error metrics"""
    if reference_tempo == 0:
        return {
            'absolute_error': 0.0,
            'relative_error': 0.0,
            'predicted_tempo': predicted_tempo,
            'reference_tempo': reference_tempo
        }
    
    abs_error = abs(predicted_tempo - reference_tempo)
    rel_error = abs_error / reference_tempo * 100.0  # Percentage
    
    return {
        'absolute_error': float(abs_error),
        'relative_error': float(rel_error),
        'predicted_tempo': float(predicted_tempo),
        'reference_tempo': float(reference_tempo)
    }


def compare_beat_detection(audio_path: str, 
                          ground_truth_beats: Optional[np.ndarray] = None,
                          ground_truth_downbeats: Optional[np.ndarray] = None,
                          ground_truth_tempo: Optional[float] = None) -> Dict:
    """
    Compare Madmom and beat_this beat detection on an audio file.
    
    Parameters
    ----------
    audio_path : str
        Path to the audio file
    ground_truth_beats : np.ndarray, optional
        Ground truth beat times (if available)
    ground_truth_downbeats : np.ndarray, optional
        Ground truth downbeat times (if available)
    ground_truth_tempo : float, optional
        Ground truth tempo (if available)
        
    Returns
    -------
    Dict
        Comparison results including metrics and timings
    """
    results = {
        'audio_file': audio_path,
        'madmom': {},
        'beat_this': {},
        'comparison': {}
    }
    
    # Test Madmom
    if MADMOM_AVAILABLE:
        print(f"\n{'='*60}")
        print("Testing Madmom...")
        print(f"{'='*60}")
        
        try:
            madmom_analyzer = MadmomAnalyzer()
            
            start_time = time.time()
            madmom_features = madmom_analyzer.extract_beats_and_downbeats(audio_path)
            madmom_time = time.time() - start_time
            
            if madmom_features:
                results['madmom'] = {
                    'execution_time': madmom_time,
                    'num_beats': len(madmom_features.beats),
                    'num_downbeats': len(madmom_features.downbeats),
                    'tempo': madmom_features.tempo,
                    'beat_consistency': madmom_features.beat_consistency,
                    'beats': madmom_features.beats.tolist(),
                    'downbeats': madmom_features.downbeats.tolist(),
                }
                
                print(f"✓ Madmom completed in {madmom_time:.2f}s")
                print(f"  - Beats detected: {len(madmom_features.beats)}")
                print(f"  - Downbeats detected: {len(madmom_features.downbeats)}")
                print(f"  - Estimated tempo: {madmom_features.tempo:.2f} BPM")
                print(f"  - Beat consistency: {madmom_features.beat_consistency:.3f}")
                
                # Calculate metrics if ground truth is available
                if ground_truth_beats is not None:
                    beat_metrics = calculate_beat_metrics(madmom_features.beats, ground_truth_beats)
                    results['madmom']['beat_metrics'] = beat_metrics
                    print(f"  - Beat F1-Score: {beat_metrics['f1_score']:.3f}")
                    print(f"  - Beat Precision: {beat_metrics['precision']:.3f}")
                    print(f"  - Beat Recall: {beat_metrics['recall']:.3f}")
                
                if ground_truth_downbeats is not None:
                    downbeat_metrics = calculate_beat_metrics(madmom_features.downbeats, ground_truth_downbeats)
                    results['madmom']['downbeat_metrics'] = downbeat_metrics
                    print(f"  - Downbeat F1-Score: {downbeat_metrics['f1_score']:.3f}")
                
                if ground_truth_tempo is not None:
                    tempo_error = calculate_tempo_error(madmom_features.tempo, ground_truth_tempo)
                    results['madmom']['tempo_error'] = tempo_error
                    print(f"  - Tempo error: {tempo_error['absolute_error']:.2f} BPM ({tempo_error['relative_error']:.1f}%)")
            else:
                print("✗ Madmom failed to extract features")
                results['madmom']['error'] = "Failed to extract features"
                
        except Exception as e:
            print(f"✗ Madmom error: {e}")
            results['madmom']['error'] = str(e)
    else:
        print("Madmom not available")
        results['madmom']['error'] = "Not installed"
    
    # Test beat_this
    if BEAT_THIS_AVAILABLE:
        print(f"\n{'='*60}")
        print("Testing beat_this...")
        print(f"{'='*60}")
        
        try:
            beat_this_analyzer = BeatThisAnalyzer()
            
            start_time = time.time()
            beat_this_features = beat_this_analyzer.extract_beats_and_downbeats(audio_path)
            beat_this_time = time.time() - start_time
            
            if beat_this_features:
                results['beat_this'] = {
                    'execution_time': beat_this_time,
                    'num_beats': len(beat_this_features.beats),
                    'num_downbeats': len(beat_this_features.downbeats),
                    'tempo': beat_this_features.tempo,
                    'beat_consistency': beat_this_features.beat_consistency,
                    'beats': beat_this_features.beats.tolist(),
                    'downbeats': beat_this_features.downbeats.tolist(),
                }
                
                print(f"✓ beat_this completed in {beat_this_time:.2f}s")
                print(f"  - Beats detected: {len(beat_this_features.beats)}")
                print(f"  - Downbeats detected: {len(beat_this_features.downbeats)}")
                print(f"  - Estimated tempo: {beat_this_features.tempo:.2f} BPM")
                print(f"  - Beat consistency: {beat_this_features.beat_consistency:.3f}")
                
                # Calculate metrics if ground truth is available
                if ground_truth_beats is not None:
                    beat_metrics = calculate_beat_metrics(beat_this_features.beats, ground_truth_beats)
                    results['beat_this']['beat_metrics'] = beat_metrics
                    print(f"  - Beat F1-Score: {beat_metrics['f1_score']:.3f}")
                    print(f"  - Beat Precision: {beat_metrics['precision']:.3f}")
                    print(f"  - Beat Recall: {beat_metrics['recall']:.3f}")
                
                if ground_truth_downbeats is not None:
                    downbeat_metrics = calculate_beat_metrics(beat_this_features.downbeats, ground_truth_downbeats)
                    results['beat_this']['downbeat_metrics'] = downbeat_metrics
                    print(f"  - Downbeat F1-Score: {downbeat_metrics['f1_score']:.3f}")
                
                if ground_truth_tempo is not None:
                    tempo_error = calculate_tempo_error(beat_this_features.tempo, ground_truth_tempo)
                    results['beat_this']['tempo_error'] = tempo_error
                    print(f"  - Tempo error: {tempo_error['absolute_error']:.2f} BPM ({tempo_error['relative_error']:.1f}%)")
            else:
                print("✗ beat_this failed to extract features")
                results['beat_this']['error'] = "Failed to extract features"
                
        except Exception as e:
            print(f"✗ beat_this error: {e}")
            results['beat_this']['error'] = str(e)
    else:
        print("beat_this not available")
        results['beat_this']['error'] = "Not installed"
    
    # Compare results
    print(f"\n{'='*60}")
    print("Comparison Summary")
    print(f"{'='*60}")
    
    if 'execution_time' in results['madmom'] and 'execution_time' in results['beat_this']:
        madmom_time = results['madmom']['execution_time']
        beat_this_time = results['beat_this']['execution_time']
        speedup = madmom_time / beat_this_time if beat_this_time > 0 else 0
        
        results['comparison']['time_comparison'] = {
            'madmom_time': madmom_time,
            'beat_this_time': beat_this_time,
            'speedup_factor': speedup,
            'faster_method': 'beat_this' if beat_this_time < madmom_time else 'madmom'
        }
        
        print(f"Execution Time:")
        print(f"  - Madmom: {madmom_time:.2f}s")
        print(f"  - beat_this: {beat_this_time:.2f}s")
        print(f"  - Speedup: {speedup:.2f}x ({results['comparison']['time_comparison']['faster_method']} is faster)")
    
    if 'beat_metrics' in results['madmom'] and 'beat_metrics' in results['beat_this']:
        madmom_f1 = results['madmom']['beat_metrics']['f1_score']
        beat_this_f1 = results['beat_this']['beat_metrics']['f1_score']
        
        results['comparison']['accuracy_comparison'] = {
            'madmom_beat_f1': madmom_f1,
            'beat_this_beat_f1': beat_this_f1,
            'f1_difference': abs(madmom_f1 - beat_this_f1),
            'better_method': 'madmom' if madmom_f1 > beat_this_f1 else 'beat_this'
        }
        
        print(f"\nBeat Detection Accuracy (F1-Score):")
        print(f"  - Madmom: {madmom_f1:.3f}")
        print(f"  - beat_this: {beat_this_f1:.3f}")
        print(f"  - Better: {results['comparison']['accuracy_comparison']['better_method']}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare Madmom and beat_this beat detection')
    parser.add_argument('audio_file', help='Path to audio file to analyze')
    parser.add_argument('--ground-truth-beats', help='Path to ground truth beats file (text file with one timestamp per line)')
    parser.add_argument('--ground-truth-downbeats', help='Path to ground truth downbeats file')
    parser.add_argument('--ground-truth-tempo', type=float, help='Ground truth tempo in BPM')
    parser.add_argument('--output', '-o', help='Path to output JSON file', default='beat_comparison_results.json')
    parser.add_argument('--tolerance', type=float, default=0.07, help='Tolerance for beat matching in seconds (default: 0.07)')
    
    args = parser.parse_args()
    
    # Check if audio file exists
    if not os.path.exists(args.audio_file):
        print(f"Error: Audio file not found: {args.audio_file}")
        return 1
    
    # Load ground truth if provided
    ground_truth_beats = None
    ground_truth_downbeats = None
    
    if args.ground_truth_beats:
        if os.path.exists(args.ground_truth_beats):
            ground_truth_beats = np.loadtxt(args.ground_truth_beats)
            print(f"Loaded {len(ground_truth_beats)} ground truth beats")
        else:
            print(f"Warning: Ground truth beats file not found: {args.ground_truth_beats}")
    
    if args.ground_truth_downbeats:
        if os.path.exists(args.ground_truth_downbeats):
            ground_truth_downbeats = np.loadtxt(args.ground_truth_downbeats)
            print(f"Loaded {len(ground_truth_downbeats)} ground truth downbeats")
        else:
            print(f"Warning: Ground truth downbeats file not found: {args.ground_truth_downbeats}")
    
    # Run comparison
    results = compare_beat_detection(
        args.audio_file,
        ground_truth_beats=ground_truth_beats,
        ground_truth_downbeats=ground_truth_downbeats,
        ground_truth_tempo=args.ground_truth_tempo
    )
    
    # Save results to JSON
    output_path = args.output
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
