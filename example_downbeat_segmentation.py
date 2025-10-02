#!/usr/bin/env python
"""
Example script demonstrating downbeat-based segmentation with real audio
"""

import sys
from pathlib import Path
from src.allin1.comprehensive_analysis import analyze_audio_comprehensive


def analyze_with_downbeat_segmentation(audio_path: str, output_dir: str = "output"):
    """
    Analyze an audio file using downbeat-based segmentation
    
    Parameters
    ----------
    audio_path : str
        Path to the audio file
    output_dir : str
        Directory to save output files
    """
    print("=" * 80)
    print("Downbeat-Based Audio Segmentation Analysis")
    print("=" * 80)
    print(f"\nAnalyzing: {audio_path}")
    print(f"Output directory: {output_dir}\n")
    
    # Perform comprehensive analysis with all features enabled
    result = analyze_audio_comprehensive(
        audio_path=audio_path,
        output_dir=output_dir,
        enable_madmom=True,        # Enable Madmom for downbeat detection
        enable_discogs=True,       # Enable genre classification per segment
        enable_segmentation=True,  # Enable segmentation and clustering
        segmentation_method='hierarchical',  # Use hierarchical clustering
        n_clusters=4,              # Target number of clusters
        genre_type='sub'           # Use sub-genre for classification
    )
    
    print("\n" + "=" * 80)
    print("Analysis Results Summary")
    print("=" * 80)
    
    # Madmom results
    if result.madmom_features:
        print(f"\n1. Madmom Beat & Downbeat Detection:")
        print(f"   - Detected {len(result.madmom_features.beats)} beats")
        print(f"   - Detected {len(result.madmom_features.downbeats)} downbeats")
        print(f"   - Tempo: {result.madmom_features.tempo:.2f} BPM")
        print(f"   - Beat consistency: {result.madmom_features.beat_consistency:.2%}")
        
        # Show first few downbeats
        if len(result.madmom_features.downbeats) > 0:
            print(f"\n   First 10 downbeats (seconds):")
            for i, db in enumerate(result.madmom_features.downbeats[:10]):
                print(f"      {i+1:2d}. {db:7.3f}s")
    
    # Essentia features
    print(f"\n2. Essentia Features (Overall):")
    ef = result.essentia_features
    print(f"   - Danceability: {ef.danceability:.2f}")
    print(f"   - Energy: {ef.energy:.2f}")
    print(f"   - Valence: {ef.valence:.2f}")
    print(f"   - Key: {ef.key}, Mode: {'Major' if ef.mode else 'Minor'}")
    print(f"   - Time Signature: {ef.time_signature}/4")
    
    # Genre information
    if result.discogs_info:
        print(f"\n3. Genre Detection (Overall):")
        print(f"   - Top genres: {', '.join(result.discogs_info.genres[:5])}")
    
    # Segmentation results
    if result.segmentation_result:
        print(f"\n4. Track Segmentation:")
        seg = result.segmentation_result
        print(f"   - Created {len(seg.segments)} segments based on downbeats")
        print(f"   - Clustered into {seg.num_clusters} clusters")
        if seg.silhouette_score:
            print(f"   - Silhouette score: {seg.silhouette_score:.3f}")
        
        print(f"\n   Sample segments (first 10):")
        for segment in seg.segments[:10]:
            duration = segment.end_time - segment.start_time
            genre_str = f" | Genre: {segment.dominant_genre}" if segment.dominant_genre else ""
            print(f"      Seg {segment.segment_id:2d} [{segment.start_time:6.2f}s - {segment.end_time:6.2f}s] "
                  f"({duration:4.2f}s) | Cluster: {segment.cluster_id} | "
                  f"Energy: {segment.features.get('energy', 0):.2f}{genre_str}")
    
    # Grouped segmentation results
    if result.grouped_segmentation:
        print(f"\n5. Segment Grouping:")
        grouped = result.grouped_segmentation
        print(f"   - {len(seg.segments)} segments grouped into {grouped.num_groups} groups")
        print(f"   - Groups based on feature similarity (threshold: 0.15)")
        
        print(f"\n   Segment groups:")
        for group in grouped.segment_groups:
            duration = group.end_time - group.start_time
            num_segs = len(group.segment_ids)
            genre_str = f" | Genre: {group.dominant_genre}" if group.dominant_genre else ""
            
            print(f"\n      Group {group.group_id} [{group.start_time:6.2f}s - {group.end_time:6.2f}s] ({duration:5.1f}s)")
            print(f"         - Contains {num_segs} segment(s): {group.segment_ids}")
            print(f"         - Avg Energy: {group.avg_features.get('energy', 0):.2f}")
            print(f"         - Avg Danceability: {group.avg_features.get('danceability', 0):.2f}")
            print(f"         - Avg Valence: {group.avg_features.get('valence', 0):.2f}{genre_str}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python example_downbeat_segmentation.py <audio_file> [output_dir]")
        print("\nExample:")
        print("  python example_downbeat_segmentation.py /path/to/song.mp3 output/")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    
    if not Path(audio_path).exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    analyze_with_downbeat_segmentation(audio_path, output_dir)
