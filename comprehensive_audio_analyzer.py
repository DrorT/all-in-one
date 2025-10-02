#!/usr/bin/env python3
"""
Comprehensive Audio Analyzer

This script provides a command-line interface for comprehensive audio analysis
using multiple libraries including Essentia, Madmom for beat and downbeat tracking,
and Discogs genre classification.

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
from allin1.comprehensive_analysis import analyze_audio_comprehensive, segment_by_downbeats, segment_by_similarity, segment_by_consecutive_similarity


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
  
  # With track segmentation
  python comprehensive_audio_analyzer.py --input song.wav --output results/ --enable-segmentation --num-clusters 4
  
  # With hierarchical clustering segmentation
  python comprehensive_audio_analyzer.py --input song.wav --output results/ --enable-segmentation --segmentation-method hierarchical --num-clusters 5
  
  # With primary genre segmentation
  python comprehensive_audio_analyzer.py --input song.wav --output results/ --enable-segmentation --genre-type primary --num-clusters 4
  
  # With Madmom beat and downbeat analysis (enabled by default)
  python comprehensive_audio_analyzer.py --input song.wav --output results/ --enable-madmom
  
  # Without Madmom beat and downbeat analysis
  python comprehensive_audio_analyzer.py --input song.wav --output results/ --disable-madmom
  
  # Complete analysis with all features
  python comprehensive_audio_analyzer.py --input song.wav --output results/ --discogs-model path/to/discogs-effnet-bs64-1.pb --include-original --enable-segmentation --segmentation-method kmeans --num-clusters 4 --genre-type sub --verbose
  
  # With downbeat-based segmentation
  python comprehensive_audio_analyzer.py --input song.wav --output results/ --enable-downbeat-segmentation --segment-group-sizes 1,2,4,8
  
  # With similarity-based segmentation
  python comprehensive_audio_analyzer.py --input song.wav --output results/ --enable-similarity-segmentation --similarity-threshold 0.85
  
  # With consecutive similarity-based segmentation
  python comprehensive_audio_analyzer.py --input song.wav --output results/ --enable-consecutive-segmentation --consecutive-threshold 0.85
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
        "--enable-segmentation", "-es",
        action="store_true",
        help="Enable track segmentation based on feature similarity"
    )
    
    parser.add_argument(
        "--segmentation-method", "-sm",
        type=str,
        choices=['kmeans', 'dbscan', 'hierarchical'],
        default='kmeans',
        help="Clustering method for track segmentation (default: kmeans)"
    )
    
    parser.add_argument(
        "--num-clusters", "-nc",
        type=int,
        default=4,
        help="Number of clusters for segmentation methods that require it (default: 4)"
    )
    
    parser.add_argument(
        "--genre-type", "-gt",
        type=str,
        choices=['primary', 'sub', 'full'],
        default='sub',
        help="Which part of the genre label to use for segmentation (primary, sub, full) (default: sub)"
    )
    
    parser.add_argument(
        "--enable-madmom",
        action="store_true",
        default=True,
        help="Enable Madmom beat and downbeat analysis (default: True)"
    )
    
    parser.add_argument(
        "--disable-madmom",
        action="store_true",
        help="Disable Madmom beat and downbeat analysis"
    )
    
    parser.add_argument(
        "--enable-downbeat-segmentation",
        action="store_true",
        help="Enable downbeat-based segmentation"
    )
    
    parser.add_argument(
        "--segment-group-sizes",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list of segment group sizes for downbeat segmentation (default: 1,2,4,8)"
    )
    
    parser.add_argument(
        "--enable-similarity-segmentation",
        action="store_true",
        help="Enable similarity-based segmentation"
    )
    
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.85,
        help="Minimum similarity score for segments to be grouped (default: 0.85)"
    )
    
    parser.add_argument(
        "--enable-consecutive-segmentation",
        action="store_true",
        help="Enable consecutive similarity-based segmentation"
    )
    
    parser.add_argument(
        "--consecutive-threshold",
        type=float,
        default=0.85,
        help="Minimum similarity score for consecutive segments to be grouped (default: 0.85)"
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
    
    # Determine if Madmom should be enabled
    enable_madmom = args.enable_madmom and not args.disable_madmom
    
    # Perform comprehensive analysis
    print("\n=== Running Comprehensive Analysis ===")
    try:
        comprehensive_result = analyze_audio_comprehensive(
            audio_path=input_path,
            output_dir=output_path,
            original_analysis=original_analysis,
            discogs_model_path=args.discogs_model,
            enable_madmom=enable_madmom,
            enable_segmentation=args.enable_segmentation,
            segmentation_method=args.segmentation_method,
            n_clusters=args.num_clusters,
            genre_type=args.genre_type
        )
        
        if args.verbose:
            print("Comprehensive analysis completed successfully")
            
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
        sys.exit(1)
    
    # Perform downbeat-based segmentation if requested
    downbeat_segmentation_result = None
    if args.enable_downbeat_segmentation:
        print("\n=== Running Downbeat-Based Segmentation ===")
        try:
            # Check if we have madmom features from the comprehensive analysis
            if comprehensive_result.madmom_features:
                print("Reusing Madmom features from comprehensive analysis...")
                madmom_features = comprehensive_result.madmom_features
                
                # Parse segment group sizes
                segment_group_sizes = [int(x.strip()) for x in args.segment_group_sizes.split(',')]
                
                # Initialize segmenter
                from allin1.comprehensive_analysis import TrackSegmenter
                segmenter = TrackSegmenter()
                
                # Perform downbeat-based segmentation using existing data
                downbeat_segmentation_result = segmenter.segment_by_downbeats(
                    input_path, madmom_features, segment_group_sizes
                )
                
                if "error" in downbeat_segmentation_result:
                    print(f"Error in downbeat segmentation: {downbeat_segmentation_result['error']}")
                    downbeat_segmentation_result = None
                else:
                    # Save downbeat segmentation results to JSON
                    downbeat_file = output_path / f"{input_path.stem}_downbeat_segments.json"
                    
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_result = {}
                    for key, value in downbeat_segmentation_result.items():
                        if key == 'segments':
                            serializable_result[key] = []
                            for segment in value:
                                serializable_segment = segment.copy()
                                # Convert any numpy values in features
                                if 'features' in serializable_segment:
                                    for feature_name, feature_value in serializable_segment['features'].items():
                                        if hasattr(feature_value, 'item'):
                                            serializable_segment['features'][feature_name] = feature_value.item()
                                serializable_result[key].append(serializable_segment)
                        elif key == 'segment_groups':
                            serializable_result[key] = {}
                            for group_name, groups in value.items():
                                serializable_result[key][group_name] = []
                                for group in groups:
                                    serializable_group = group.copy()
                                    # Convert any numpy values in avg_features
                                    if 'avg_features' in serializable_group:
                                        for feature_name, feature_value in serializable_group['avg_features'].items():
                                            if hasattr(feature_value, 'item'):
                                                serializable_group['avg_features'][feature_name] = feature_value.item()
                                    serializable_result[key][group_name].append(serializable_group)
                        else:
                            serializable_result[key] = value
                    
                    with open(downbeat_file, 'w') as f:
                        json.dump(serializable_result, f, indent=2)
                    
                    print(f"Downbeat segmentation results saved to: {downbeat_file}")
                    
                    if args.verbose:
                        print(f"Created {len(downbeat_segmentation_result['segments'])} segments based on downbeats")
                        print(f"Analyzed {len(downbeat_segmentation_result['segment_groups'])} group sizes")
            else:
                print("No Madmom features available from comprehensive analysis. Skipping downbeat segmentation.")
                    
        except Exception as e:
            print(f"Error in downbeat segmentation: {e}")
            downbeat_segmentation_result = None
    
    # Perform similarity-based segmentation if requested
    similarity_segmentation_result = None
    if args.enable_similarity_segmentation:
        print("\n=== Running Similarity-Based Segmentation ===")
        try:
            # Check if we have madmom features from the comprehensive analysis
            if comprehensive_result.madmom_features:
                print("Reusing Madmom features from comprehensive analysis...")
                madmom_features = comprehensive_result.madmom_features
                
                # Initialize segmenter
                from allin1.comprehensive_analysis import TrackSegmenter
                segmenter = TrackSegmenter()
                
                # Perform similarity-based segmentation using existing data
                similarity_segmentation_result = segmenter.segment_by_similarity(
                    input_path, madmom_features, args.similarity_threshold
                )
                
                if "error" in similarity_segmentation_result:
                    print(f"Error in similarity segmentation: {similarity_segmentation_result['error']}")
                    similarity_segmentation_result = None
                else:
                    # Save similarity segmentation results to JSON
                    similarity_file = output_path / f"{input_path.stem}_similarity_segments.json"
                    
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_result = {}
                    for key, value in similarity_segmentation_result.items():
                        if key == 'segments':
                            serializable_result[key] = []
                            for segment in value:
                                serializable_segment = segment.copy()
                                # Convert any numpy values in features
                                if 'features' in serializable_segment:
                                    for feature_name, feature_value in serializable_segment['features'].items():
                                        if hasattr(feature_value, 'item'):
                                            serializable_segment['features'][feature_name] = feature_value.item()
                                serializable_result[key].append(serializable_segment)
                        elif key == 'similarity_groups':
                            serializable_result[key] = []
                            for group in value:
                                serializable_group = group.copy()
                                # Convert any numpy values in avg_features
                                if 'avg_features' in serializable_group:
                                    for feature_name, feature_value in serializable_group['avg_features'].items():
                                        if hasattr(feature_value, 'item'):
                                            serializable_group['avg_features'][feature_name] = feature_value.item()
                                serializable_result[key].append(serializable_group)
                        else:
                            serializable_result[key] = value
                    
                    with open(similarity_file, 'w') as f:
                        json.dump(serializable_result, f, indent=2)
                    
                    print(f"Similarity segmentation results saved to: {similarity_file}")
                    
                    if args.verbose:
                        print(f"Created {len(similarity_segmentation_result['segments'])} segments based on downbeats")
                        print(f"Found {len(similarity_segmentation_result['similarity_groups'])} similarity-based groups")
            else:
                print("No Madmom features available from comprehensive analysis. Skipping similarity segmentation.")
                
        except Exception as e:
            print(f"Error in similarity segmentation: {e}")
            similarity_segmentation_result = None
    
    # Perform consecutive similarity-based segmentation if requested
    consecutive_segmentation_result = None
    if args.enable_consecutive_segmentation:
        print("\n=== Running Consecutive Similarity-Based Segmentation ===")
        try:
            # Check if we have madmom features from the comprehensive analysis
            if comprehensive_result.madmom_features:
                print("Reusing Madmom features from comprehensive analysis...")
                madmom_features = comprehensive_result.madmom_features
                
                # Initialize segmenter
                from allin1.comprehensive_analysis import TrackSegmenter
                segmenter = TrackSegmenter()
                
                # Perform consecutive similarity-based segmentation using existing data
                consecutive_segmentation_result = segmenter.segment_by_consecutive_similarity(
                    input_path, madmom_features, args.consecutive_threshold
                )
                
                if "error" in consecutive_segmentation_result:
                    print(f"Error in consecutive segmentation: {consecutive_segmentation_result['error']}")
                    consecutive_segmentation_result = None
                else:
                    # Save consecutive segmentation results to JSON
                    consecutive_file = output_path / f"{input_path.stem}_consecutive_segments.json"
                    
                    # Convert numpy arrays to lists for JSON serialization
                    serializable_result = {}
                    for key, value in consecutive_segmentation_result.items():
                        if key == 'segments':
                            serializable_result[key] = []
                            for segment in value:
                                serializable_segment = segment.copy()
                                # Convert any numpy values in features
                                if 'features' in serializable_segment:
                                    for feature_name, feature_value in serializable_segment['features'].items():
                                        if hasattr(feature_value, 'item'):
                                            serializable_segment['features'][feature_name] = feature_value.item()
                                serializable_result[key].append(serializable_segment)
                        elif key == 'consecutive_groups':
                            serializable_result[key] = []
                            for group in value:
                                serializable_group = group.copy()
                                # Convert any numpy values in avg_features
                                if 'avg_features' in serializable_group:
                                    for feature_name, feature_value in serializable_group['avg_features'].items():
                                        if hasattr(feature_value, 'item'):
                                            serializable_group['avg_features'][feature_name] = feature_value.item()
                                serializable_result[key].append(serializable_group)
                        else:
                            serializable_result[key] = value
                    
                    with open(consecutive_file, 'w') as f:
                        json.dump(serializable_result, f, indent=2)
                    
                    print(f"Consecutive segmentation results saved to: {consecutive_file}")
                    
                    if args.verbose:
                        print(f"Created {len(consecutive_segmentation_result['segments'])} segments based on downbeats")
                        print(f"Found {len(consecutive_segmentation_result['consecutive_groups'])} consecutive similarity-based groups")
            else:
                print("No Madmom features available from comprehensive analysis. Skipping consecutive segmentation.")
                
        except Exception as e:
            print(f"Error in consecutive segmentation: {e}")
            consecutive_segmentation_result = None
    
    # Print summary of results
    print("\n=== Analysis Summary ===")
    print(f"Audio file: {comprehensive_result.path.name}")
    
    print("\n--- Essentia Features ---")
    print(f"Danceability: {comprehensive_result.essentia_features.danceability:.3f}")
    print(f"Energy: {comprehensive_result.essentia_features.energy:.3f}")
    print(f"Valence: {comprehensive_result.essentia_features.valence:.3f}")
    print(f"Acousticness: {comprehensive_result.essentia_features.acousticness:.3f}")
    print(f"Key: {comprehensive_result.essentia_features.key} ({'Major' if comprehensive_result.essentia_features.mode == 1 else 'Minor'})")
    
    # Print Madmom features if available
    if comprehensive_result.madmom_features:
        print("\n--- Madmom Beat and Downbeat Analysis ---")
        print(f"Tempo: {comprehensive_result.madmom_features.tempo:.1f} BPM")
        print(f"Beats detected: {len(comprehensive_result.madmom_features.beats)}")
        print(f"Downbeats detected: {len(comprehensive_result.madmom_features.downbeats)}")
        print(f"Beat consistency: {comprehensive_result.madmom_features.beat_consistency:.3f}")
    
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
    
    # Print segmentation results if available
    if comprehensive_result.segmentation_result:
        print("\n--- Track Segmentation ---")
        seg_result = comprehensive_result.segmentation_result
        print(f"Clustering method: {seg_result.clustering_method}")
        print(f"Number of clusters: {seg_result.num_clusters}")
        
        if seg_result.silhouette_score is not None:
            print(f"Silhouette score: {seg_result.silhouette_score:.3f}")
        
        # Print cluster information
        cluster_info = {}
        for segment in seg_result.segments:
            cluster_id = segment.cluster_id
            if cluster_id not in cluster_info:
                cluster_info[cluster_id] = {
                    'count': 0,
                    'duration': 0,
                    'genres': {}
                }
            
            cluster_info[cluster_id]['count'] += 1
            cluster_info[cluster_id]['duration'] += segment.end_time - segment.start_time
            
            # Track dominant genres
            if segment.dominant_genre:
                genre = segment.dominant_genre
                if '---' in genre:
                    genre = genre.split('---')[1]
                if genre not in cluster_info[cluster_id]['genres']:
                    cluster_info[cluster_id]['genres'][genre] = 0
                cluster_info[cluster_id]['genres'][genre] += 1
        
        for cluster_id, info in sorted(cluster_info.items()):
            print(f"  Cluster {cluster_id}: {info['count']} segments, {info['duration']:.1f}s total")
            
            if info['genres']:
                top_genre = max(info['genres'].items(), key=lambda x: x[1])
                print(f"    Dominant genre: {top_genre[0]} ({top_genre[1]} segments)")
    
    # Print downbeat segmentation results if available
    if downbeat_segmentation_result:
        print("\n--- Downbeat-Based Segmentation ---")
        print(f"Total duration: {downbeat_segmentation_result['total_duration']:.2f} seconds")
        print(f"Number of downbeats: {downbeat_segmentation_result['num_downbeats']}")
        print(f"Number of segments: {len(downbeat_segmentation_result['segments'])}")
        
        # Print segment group information
        for group_name, groups in downbeat_segmentation_result['segment_groups'].items():
            print(f"  {group_name}: {len(groups)} groups")
        
        # Print most similar pairs for each group size
        for group_size, sim_data in downbeat_segmentation_result['similarities'].items():
            print(f"\n  Most similar {group_size}:")
            for pair in sim_data['most_similar_pairs'][:2]:  # Show top 2 pairs
                print(f"    Groups {pair['group1_id']} and {pair['group2_id']}: "
                      f"similarity={pair['similarity']:.3f}")
    
    # Print similarity segmentation results if available
    if similarity_segmentation_result:
        print("\n--- Similarity-Based Segmentation ---")
        print(f"Total duration: {similarity_segmentation_result['total_duration']:.2f} seconds")
        print(f"Number of downbeats: {similarity_segmentation_result['num_downbeats']}")
        print(f"Number of segments: {len(similarity_segmentation_result['segments'])}")
        print(f"Similarity threshold: {similarity_segmentation_result['similarity_threshold']}")
        print(f"Number of similarity groups: {len(similarity_segmentation_result['similarity_groups'])}")
        
        # Print group information
        for group in similarity_segmentation_result['similarity_groups']:
            print(f"  Group {group['group_id']}: {group['num_segments']} segments, "
                  f"similarity={group['similarities']['avg_similarity']:.3f}")
    
    # Print consecutive segmentation results if available
    if consecutive_segmentation_result:
        print("\n--- Consecutive Similarity-Based Segmentation ---")
        print(f"Total duration: {consecutive_segmentation_result['total_duration']:.2f} seconds")
        print(f"Number of downbeats: {consecutive_segmentation_result['num_downbeats']}")
        print(f"Number of segments: {len(consecutive_segmentation_result['segments'])}")
        print(f"Similarity threshold: {consecutive_segmentation_result['similarity_threshold']}")
        print(f"Number of consecutive groups: {len(consecutive_segmentation_result['consecutive_groups'])}")
        
        # Print group information
        for group in consecutive_segmentation_result['consecutive_groups']:
            print(f"  Group {group['group_id']}: {group['num_segments']} segments, "
                  f"{group['start_time']:.2f}s - {group['end_time']:.2f}s")
        
        # Print most similar groups
        if consecutive_segmentation_result['group_similarities']['most_similar_pairs']:
            print("\nMost similar groups:")
            for pair in consecutive_segmentation_result['group_similarities']['most_similar_pairs'][:3]:
                print(f"  Groups {pair['group1_id']} and {pair['group2_id']}: "
                      f"similarity={pair['similarity']:.3f}")
    
    print(f"\nResults saved to: {output_path}")
    print(f"  - Comprehensive analysis: {output_path / (input_path.stem + '_comprehensive_analysis.json')}")
    print(f"  - Heatmap visualization: {output_path / (input_path.stem + '_heatmap.png')}")
    print(f"  - Timeline visualization: {output_path / (input_path.stem + '_timeline.png')}")
    
    if comprehensive_result.madmom_features:
        print(f"  - Beat and downbeat visualization: {output_path / (input_path.stem + '_beats_downbeats.png')}")
        print(f"  - Beat and downbeat visualization (SVG): {output_path / (input_path.stem + '_beats_downbeats.svg')}")
    
    if comprehensive_result.segmentation_result:
        print(f"  - Segmentation visualization: {output_path / (input_path.stem + '_segmentation.png')}")
    
    if args.include_original:
        print(f"  - Original analysis: {output_path / 'original_analysis'}")
    
    if downbeat_segmentation_result:
        print(f"  - Downbeat segmentation: {output_path / (input_path.stem + '_downbeat_segments.json')}")
    
    if similarity_segmentation_result:
        print(f"  - Similarity segmentation: {output_path / (input_path.stem + '_similarity_segments.json')}")
    
    if consecutive_segmentation_result:
        print(f"  - Consecutive segmentation: {output_path / (input_path.stem + '_consecutive_segments.json')}")
    
    print("\n=== Analysis Complete ===")


if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues with imports above
    main()