#!/usr/bin/env python3
"""
Comprehensive Audio Analyzer

This script provides a command-line interface for comprehensive audio analysis
using multiple libraries including Essentia and Discogs genre classification.

Usage:
    python comprehensive_audio_analyzer.py --input audio_file.wav --output results/
    python comprehensive_audio_analyzer.py --input audio_file.wav --output results/
"""

import argparse
import sys
import os
from pathlib import Path
import json

# Add the src directory to the path so we can import allin1 modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from allin1.analyze import analyze as allin1_analyze
from allin1.comprehensive_analysis import analyze_audio_comprehensive


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Audio Analyzer - Extract musical features, tags, and genre classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python comprehensive_audio_analyzer.py --input song.wav --output results/
  
  # With Discogs genre classification (requires model file)
  python comprehensive_audio_analyzer.py --input song.wav --output results/ --discogs-model path/to/discogs-effnet-bs64-1.pb
  
  # Including original allin1 analysis
  python comprehensive_audio_analyzer.py --input song.wav --output results/ --include-original
  
  # Complete analysis with all features
  python comprehensive_audio_analyzer.py --input song.wav --output results/ --discogs-model path/to/discogs-effnet-bs64-1.pb --include-original --verbose
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to the audio file to analyze"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Directory to save analysis results and visualizations"
    )
    
    parser.add_argument(
        "--discogs-model", "-dm",
        type=str,
        default=None,
        help="Path to the Discogs genre classification model file (discogs-effnet-bs64-1.pb)"
    )
    
    parser.add_argument(
        "--include-original", "-io",
        action="store_true",
        help="Include original allin1 structural analysis"
    )
    
    parser.add_argument(
        "--segment-duration", "-sd",
        type=float,
        default=5.0,
        help="Duration in seconds for time-based analysis segments (default: 5.0)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)
    
    if not input_path.suffix.lower() in ['.wav', '.mp3', '.flac', '.m4a', '.ogg']:
        print(f"Warning: File extension '{input_path.suffix}' may not be supported.")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if args.verbose:
        print(f"Input file: {input_path}")
        print(f"Output directory: {output_path}")
        print(f"Segment duration: {args.segment_duration} seconds")
    
    # Perform original allin1 analysis if requested
    original_analysis = None
    if args.include_original:
        print("\n=== Running Original All-in-One Analysis ===")
        try:
            original_analysis = allin1_analyze(
                paths=input_path,
                out_dir=output_path / "original_analysis",
                visualize=True,
                sonify=False,
                overwrite=True
            )
            if args.verbose:
                print("Original analysis completed successfully")
        except Exception as e:
            print(f"Error in original analysis: {e}")
            print("Continuing with comprehensive analysis only...")
    
    # Perform comprehensive analysis
    print("\n=== Running Comprehensive Analysis ===")
    try:
        comprehensive_result = analyze_audio_comprehensive(
            audio_path=input_path,
            output_dir=output_path,
            original_analysis=original_analysis,
            discogs_model_path=args.discogs_model
        )
        
        if args.verbose:
            print("Comprehensive analysis completed successfully")
            
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        sys.exit(1)
    
    # Print summary of results
    print("\n=== Analysis Summary ===")
    print(f"Audio file: {comprehensive_result.path.name}")
    
    print("\n--- Essentia Features ---")
    print(f"Danceability: {comprehensive_result.essentia_features.danceability:.3f}")
    print(f"Energy: {comprehensive_result.essentia_features.energy:.3f}")
    print(f"Valence: {comprehensive_result.essentia_features.valence:.3f}")
    print(f"Acousticness: {comprehensive_result.essentia_features.acousticness:.3f}")
    print(f"Tempo: {comprehensive_result.essentia_features.tempo:.1f} BPM")
    print(f"Key: {comprehensive_result.essentia_features.key} ({'Major' if comprehensive_result.essentia_features.mode == 1 else 'Minor'})")
    
    if comprehensive_result.discogs_info:
        print("\n--- Discogs Information ---")
        if comprehensive_result.discogs_info.artist:
            print(f"Artist: {comprehensive_result.discogs_info.artist}")
        if comprehensive_result.discogs_info.title:
            print(f"Title: {comprehensive_result.discogs_info.title}")
        if comprehensive_result.discogs_info.genres:
            print(f"Genres: {', '.join(comprehensive_result.discogs_info.genres)}")
        if comprehensive_result.discogs_info.styles:
            print(f"Styles: {', '.join(comprehensive_result.discogs_info.styles)}")
        if comprehensive_result.discogs_info.year:
            print(f"Year: {comprehensive_result.discogs_info.year}")
    
    print("\n--- Time-Based Analysis ---")
    print(f"Analyzed {len(comprehensive_result.time_based_features.time_stamps)} time segments")
    print(f"Duration: {comprehensive_result.time_based_features.time_stamps[-1]:.1f} seconds")
    
    # Calculate feature statistics
    features = comprehensive_result.time_based_features.features
    if 'danceability' in features and len(features['danceability']) > 0:
        print(f"Average danceability: {np.mean(features['danceability']):.3f}")
        print(f"Danceability range: {np.min(features['danceability']):.3f} - {np.max(features['danceability']):.3f}")
    
    if 'energy' in features and len(features['energy']) > 0:
        print(f"Average energy: {np.mean(features['energy']):.3f}")
        print(f"Energy range: {np.min(features['energy']):.3f} - {np.max(features['energy']):.3f}")
    
    print(f"\nResults saved to: {output_path}")
    print(f"  - Comprehensive analysis: {output_path / (input_path.stem + '_comprehensive_analysis.json')}")
    print(f"  - Heatmap visualization: {output_path / (input_path.stem + '_heatmap.png')}")
    print(f"  - Timeline visualization: {output_path / (input_path.stem + '_timeline.png')}")
    
    if args.include_original:
        print(f"  - Original analysis: {output_path / 'original_analysis'}")
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues with imports above
    main()