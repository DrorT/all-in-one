#!/usr/bin/env python
"""
Test script for downbeat-based segmentation
"""

import numpy as np
from src.allin1.comprehensive_analysis import (
    calculate_downbeat_segments,
    ComprehensiveAnalyzer,
    group_similar_segments
)


def test_downbeat_segment_calculation():
    """Test the downbeat segment calculation function"""
    print("=" * 80)
    print("Testing downbeat segment calculation")
    print("=" * 80)
    
    # Test case 1: Normal downbeats
    print("\nTest 1: Normal downbeats")
    downbeats = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    audio_duration = 12.0
    segments = calculate_downbeat_segments(downbeats, audio_duration)
    print(f"Downbeats: {downbeats}")
    print(f"Audio duration: {audio_duration}s")
    print(f"Segments: {segments}")
    
    # Test case 2: First downbeat very early (< 0.3s)
    print("\nTest 2: First downbeat very early (< 0.3s)")
    downbeats = np.array([0.1, 2.0, 4.0, 6.0])
    audio_duration = 8.0
    segments = calculate_downbeat_segments(downbeats, audio_duration)
    print(f"Downbeats: {downbeats}")
    print(f"Audio duration: {audio_duration}s")
    print(f"Segments: {segments}")
    print("Note: First segment should start from 0 to 2nd downbeat")
    
    # Test case 3: Last downbeat very close to end (< 0.3s from end)
    print("\nTest 3: Last downbeat very close to end (< 0.3s from end)")
    downbeats = np.array([2.0, 4.0, 6.0, 7.8])
    audio_duration = 8.0
    segments = calculate_downbeat_segments(downbeats, audio_duration)
    print(f"Downbeats: {downbeats}")
    print(f"Audio duration: {audio_duration}s")
    print(f"Segments: {segments}")
    print("Note: Last segment should be extended to end of track")
    
    # Test case 4: Both first and last downbeats edge cases
    print("\nTest 4: Both first and last downbeats edge cases")
    downbeats = np.array([0.15, 2.0, 4.0, 6.0, 7.85])
    audio_duration = 8.0
    segments = calculate_downbeat_segments(downbeats, audio_duration)
    print(f"Downbeats: {downbeats}")
    print(f"Audio duration: {audio_duration}s")
    print(f"Segments: {segments}")
    
    # Test case 5: No downbeats
    print("\nTest 5: No downbeats")
    downbeats = np.array([])
    audio_duration = 10.0
    segments = calculate_downbeat_segments(downbeats, audio_duration)
    print(f"Downbeats: {downbeats}")
    print(f"Audio duration: {audio_duration}s")
    print(f"Segments: {segments}")
    print("Note: Should return single segment for entire track")
    
    # Test case 6: Single downbeat
    print("\nTest 6: Single downbeat")
    downbeats = np.array([3.0])
    audio_duration = 6.0
    segments = calculate_downbeat_segments(downbeats, audio_duration)
    print(f"Downbeats: {downbeats}")
    print(f"Audio duration: {audio_duration}s")
    print(f"Segments: {segments}")
    

def demonstrate_full_analysis():
    """Demonstrate the full analysis workflow with downbeat-based segmentation"""
    print("\n" + "=" * 80)
    print("Full Analysis Workflow Demonstration")
    print("=" * 80)
    print("\nThis would analyze an audio file with:")
    print("1. Downbeat detection using Madmom")
    print("2. Segment creation between downbeats")
    print("3. Feature extraction for each segment")
    print("4. Genre analysis per segment (if Discogs model available)")
    print("5. Grouping of consecutive similar segments")
    print("\nExample usage:")
    print("""
    from src.allin1.comprehensive_analysis import analyze_audio_comprehensive
    
    result = analyze_audio_comprehensive(
        audio_path='path/to/audio.mp3',
        output_dir='output/',
        enable_madmom=True,
        enable_discogs=True,
        enable_segmentation=True,
        segmentation_method='kmeans',
        n_clusters=4
    )
    
    # Access results
    print(f"Downbeats detected: {len(result.madmom_features.downbeats)}")
    print(f"Segments created: {len(result.segmentation_result.segments)}")
    print(f"Segment groups: {len(result.grouped_segmentation.segment_groups)}")
    
    # Iterate over segment groups
    for group in result.grouped_segmentation.segment_groups:
        print(f"Group {group.group_id}: {group.start_time:.2f}s - {group.end_time:.2f}s")
        print(f"  Contains {len(group.segment_ids)} segments")
        print(f"  Dominant genre: {group.dominant_genre}")
        print(f"  Avg energy: {group.avg_features.get('energy', 0):.2f}")
    """)


if __name__ == "__main__":
    test_downbeat_segment_calculation()
    demonstrate_full_analysis()
    
    print("\n" + "=" * 80)
    print("Tests completed!")
    print("=" * 80)
