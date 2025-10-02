#!/usr/bin/env python3
"""
Test script for downbeat-based segmentation
"""

import os
import sys
import json
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from allin1.comprehensive_analysis import segment_by_downbeats, ComprehensiveAnalyzer

def test_downbeat_segmentation():
    """Test the downbeat-based segmentation functionality"""
    
    # Find an audio file to test with
    test_files = [
        "comprehensive_results/3 Doors Down - Here Without You (YouTwoSid3s & FreshTech Remix) [VzXR5EgpLFU]_comprehensive_analysis.json",
        "test_audio.mp3",
        "test_audio.wav",
        "example.mp3",
        "example.wav"
    ]
    
    audio_path = None
    for file in test_files:
        if os.path.exists(file):
            if file.endswith('.json'):
                # Extract the original audio path from the JSON file
                with open(file, 'r') as f:
                    data = json.load(f)
                    # Try to find the audio file path in the JSON
                    if 'path' in data:
                        potential_path = data['path']
                        if os.path.exists(potential_path):
                            audio_path = potential_path
                            break
            else:
                audio_path = file
                break
    
    if not audio_path:
        print("No suitable audio file found for testing.")
        print("Please place an audio file in the current directory or update the test_files list.")
        return False
    
    print(f"Testing downbeat-based segmentation with: {audio_path}")
    
    try:
        # Test the convenience function
        print("\n=== Testing convenience function ===")
        result = segment_by_downbeats(
            audio_path=audio_path,
            segment_group_sizes=[1, 2, 4, 8]
        )
        
        if "error" in result:
            print(f"Error: {result['error']}")
            return False
        
        # Print results
        print(f"Total duration: {result['total_duration']:.2f} seconds")
        print(f"Number of downbeats: {result['num_downbeats']}")
        print(f"Number of segments: {len(result['segments'])}")
        
        # Print segment information
        print("\n=== Segments ===")
        for i, segment in enumerate(result['segments'][:5]):  # Show first 5 segments
            print(f"Segment {segment['segment_id']}: {segment['start_time']:.2f}s - {segment['end_time']:.2f}s "
                  f"(duration: {segment['end_time'] - segment['start_time']:.2f}s)")
        
        # Print segment groups
        print("\n=== Segment Groups ===")
        for group_name, groups in result['segment_groups'].items():
            print(f"\n{group_name}: {len(groups)} groups")
            for group in groups[:3]:  # Show first 3 groups
                print(f"  Group {group['group_id']}: segments {group['segment_indices']}, "
                      f"{group['start_time']:.2f}s - {group['end_time']:.2f}s")
        
        # Print similarity results
        print("\n=== Similarity Results ===")
        for group_size, sim_data in result['similarities'].items():
            print(f"\n{group_size}:")
            for pair in sim_data['most_similar_pairs'][:3]:  # Show top 3 pairs
                print(f"  Groups {pair['group1_id']} and {pair['group2_id']}: "
                      f"similarity={pair['similarity']:.3f}")
        
        # Test the class method directly
        print("\n=== Testing class method ===")
        analyzer = ComprehensiveAnalyzer(enable_discogs=False, enable_madmom=True)
        result2 = analyzer.segment_by_downbeats(audio_path, [1, 2, 4])
        
        if "error" in result2:
            print(f"Error: {result2['error']}")
            return False
        
        print(f"Class method successful: {len(result2['segments'])} segments created")
        
        print("\n=== Test completed successfully! ===")
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_downbeat_segmentation()
    sys.exit(0 if success else 1)