#!/usr/bin/env python3
"""
Example script demonstrating downbeat-based segmentation
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from allin1.comprehensive_analysis import segment_by_downbeats, ComprehensiveAnalyzer

def main():
    """Main function to demonstrate downbeat-based segmentation"""
    
    # Path to your audio file
    audio_path = "/path/to/your/audio/file.mp3"  # Replace with your audio file path
    
    # Check if the file exists
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        print("Please update the audio_path variable with a valid audio file path.")
        return
    
    print(f"Analyzing: {audio_path}")
    
    # Method 1: Using the convenience function
    print("\n=== Using convenience function ===")
    result = segment_by_downbeats(
        audio_path=audio_path,
        segment_group_sizes=[1, 2, 4, 8]  # Analyze groups of 1, 2, 4, and 8 segments
    )
    
    if "error" in result:
        print(f"Error: {result['error']}")
        return
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Total duration: {result['total_duration']:.2f} seconds")
    print(f"- Number of downbeats: {result['num_downbeats']}")
    print(f"- Number of segments: {len(result['segments'])}")
    
    # Print some example segments
    print(f"\nExample segments:")
    for i, segment in enumerate(result['segments'][:5]):  # Show first 5 segments
        print(f"  Segment {segment['segment_id']}: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s")
    
    # Print similarity results for 4-segment groups
    if 'groups_of_4' in result['similarities']:
        print(f"\nMost similar 4-segment groups:")
        for pair in result['similarities']['groups_of_4']['most_similar_pairs'][:3]:
            print(f"  Groups {pair['group1_id']} and {pair['group2_id']}: "
                  f"similarity={pair['similarity']:.3f}")
    
    # Method 2: Using the class directly
    print("\n=== Using class directly ===")
    analyzer = ComprehensiveAnalyzer(enable_discogs=False, enable_madmom=True)
    result2 = analyzer.segment_by_downbeats(audio_path, [1, 2, 4, 8])
    
    if "error" in result2:
        print(f"Error: {result['error']}")
        return
    
    print(f"Class method successful: {len(result2['segments'])} segments created")
    
    # Save results to a JSON file
    output_file = f"{Path(audio_path).stem}_downbeat_segments.json"
    with open(output_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_result = {}
        for key, value in result.items():
            if key == 'segments':
                serializable_result[key] = []
                for segment in value:
                    serializable_segment = segment.copy()
                    # Remove non-serializable features if needed
                    serializable_result[key].append(serializable_segment)
            else:
                serializable_result[key] = value
        
        json.dump(serializable_result, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()