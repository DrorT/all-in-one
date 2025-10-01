#!/usr/bin/env python3
"""
Example script demonstrating how to use the new librosa BPM analysis feature.
"""

import sys
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from allin1 import analyze

def example_usage():
    """
    Example of how to use the new librosa BPM analysis feature.
    """
    # Replace this with the path to your audio file
    audio_file = "~/Music/test/so_short.mp3"
    
    print("Running analysis with librosa BPM comparison...")
    print(f"Audio file: {audio_file}")
    
    try:
        # Run the analysis with our new librosa integration
        result = analyze(
            paths=audio_file,
            out_dir="./output",
            model="harmonix-all",
            device="cpu",  # Use CPU to avoid CUDA issues
            include_activations=False,
            include_embeddings=False,
            keep_byproducts=True
        )
        
        # Print the model's BPM estimation
        print(f"\nModel BPM: {result.bpm}")
        
        # Print the librosa analysis results
        if result.librosa_analysis:
            print("\nLibrosa Analysis Results:")
            print(f"  Librosa BPM: {result.librosa_analysis.get('librosa_bpm', 'N/A')}")
            print(f"  BPM Difference: {result.librosa_analysis.get('bpm_difference', 'N/A')}")
            
            beat_comparison = result.librosa_analysis.get('beat_comparison', {})
            if beat_comparison:
                print(f"  Beat Alignment MAE: {beat_comparison.get('mean_absolute_error', 'N/A'):.4f} seconds")
                print(f"  Best Beat Offset: {beat_comparison.get('best_offset', 'N/A'):.4f} seconds")
                print(f"  Model Beat Count: {beat_comparison.get('model_beat_count', 'N/A')}")
                print(f"  Librosa Beat Count: {beat_comparison.get('librosa_beat_count', 'N/A')}")
            
            # Check if there was an error
            if 'error' in result.librosa_analysis:
                print(f"  Error: {result.librosa_analysis['error']}")
        else:
            print("\nNo librosa analysis found in the result.")
        
        # Save the results to a JSON file for inspection
        output_file = Path("./output/analysis_result.json")
        output_file.parent.mkdir(exist_ok=True)
        
        # Convert the result to a dictionary for JSON serialization
        result_dict = {
            'path': str(result.path),
            'model_bpm': result.bpm,
            'beats': result.beats,
            'downbeats': result.downbeats,
            'beat_positions': result.beat_positions,
            'segments': [
                {
                    'start': seg.start,
                    'end': seg.end,
                    'label': seg.label
                } for seg in result.segments
            ],
            'librosa_analysis': result.librosa_analysis
        }
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"\nFull results saved to: {output_file}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("Librosa BPM Analysis Example")
    print("=" * 40)
    print("\nThis example demonstrates how to use the new librosa BPM analysis feature.")
    print("Please update the 'audio_file' variable in the script to point to your audio file.")
    print("\nTo run with an actual audio file, uncomment the line below and update the path:")
    print("# audio_file = 'path/to/your/audio.wav'")
    print("\nFor now, this will just show the structure of the analysis without running it.")
    
    # Uncomment the following line to run the actual analysis:
    # example_usage()