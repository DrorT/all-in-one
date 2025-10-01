#!/usr/bin/env python3
"""
Test script to verify the librosa BPM analysis implementation.
"""

import sys
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from allin1 import analyze
from allin1.postprocessing.tempo import analyze_bpm_with_librosa

def test_librosa_analysis():
    """Test the librosa BPM analysis with a sample audio file."""
    
    # Check if we have any audio files in the current directory
    audio_files = list(Path(".").glob("*.wav")) + list(Path(".").glob("*.mp3"))
    
    if not audio_files:
        print("No audio files found in the current directory.")
        print("Please place a .wav or .mp3 file in the current directory to test.")
        return False
    
    audio_file = audio_files[0]
    print(f"Testing with audio file: {audio_file}")
    
    try:
        # Run the analysis with our new librosa integration
        result = analyze(
            paths=audio_file,
            out_dir="./test_output",
            model="harmonix-all",
            device="cpu",  # Use CPU to avoid CUDA issues
            include_activations=False,
            include_embeddings=False,
            keep_byproducts=True
        )
        
        # Check if librosa analysis was performed
        if result.librosa_analysis:
            print("\nLibrosa Analysis Results:")
            print(f"  Model BPM: {result.bpm}")
            print(f"  Librosa BPM: {result.librosa_analysis.get('librosa_bpm', 'N/A')}")
            print(f"  BPM Difference: {result.librosa_analysis.get('bpm_difference', 'N/A')}")
            
            beat_comparison = result.librosa_analysis.get('beat_comparison', {})
            if beat_comparison:
                print(f"  Beat MAE: {beat_comparison.get('mean_absolute_error', 'N/A')}")
                print(f"  Best Offset: {beat_comparison.get('best_offset', 'N/A')}")
                print(f"  Model Beat Count: {beat_comparison.get('model_beat_count', 'N/A')}")
                print(f"  Librosa Beat Count: {beat_comparison.get('librosa_beat_count', 'N/A')}")
            
            # Check if there was an error
            if 'error' in result.librosa_analysis:
                print(f"  Error: {result.librosa_analysis['error']}")
                return False
            else:
                print("\nLibrosa analysis completed successfully!")
                return True
        else:
            print("No librosa analysis found in the result.")
            return False
            
    except Exception as e:
        print(f"Error during analysis: {e}")
        return False

if __name__ == "__main__":
    success = test_librosa_analysis()
    sys.exit(0 if success else 1)