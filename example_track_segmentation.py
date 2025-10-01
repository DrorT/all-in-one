"""
Example script demonstrating track segmentation using time-based features and genre predictions.

This script shows how to:
1. Extract time-based features from an audio file
2. Perform genre analysis over time
3. Segment the track based on feature similarity
4. Visualize the segmentation results
"""

import os
import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from allin1.comprehensive_analysis import (
    ComprehensiveAnalyzer, 
    TrackSegmenter,
    analyze_audio_comprehensive,
    segment_audio_track
)


def main():
    # Example audio file path - replace with your own audio file
    audio_file = "path/to/your/audio/file.mp3"
    
    # Check if the file exists
    if not os.path.exists(audio_file):
        print(f"Audio file not found: {audio_file}")
        print("Please update the audio_file variable with a valid path to an audio file.")
        return
    
    # Output directory for results
    output_dir = Path("segmentation_results")
    output_dir.mkdir(exist_ok=True)
    
    # Path to Discogs model (if not in Essentia's default location)
    # You can download the model from: https://essentia.upf.edu/models/
    discogs_model_path = "autotagging/discogs-effnet-bs64-1.pb"
    
    print("=" * 60)
    print("Track Segmentation Example")
    print("=" * 60)
    
    # Method 1: Using the convenience function with segmentation enabled
    print("\n1. Performing comprehensive analysis with segmentation...")
    try:
        result = analyze_audio_comprehensive(
            audio_path=audio_file,
            output_dir=output_dir / "comprehensive",
            enable_discogs=True,
            discogs_model_path=discogs_model_path if os.path.exists(discogs_model_path) else None,
            enable_segmentation=True,
            segmentation_method='kmeans',
            n_clusters=4
        )
        
        print(f"Analysis complete! Results saved to {output_dir / 'comprehensive'}")
        
        # Print segmentation summary
        if result.segmentation_result:
            print("\nSegmentation Summary:")
            print(f"  Method: {result.segmentation_result.clustering_method}")
            print(f"  Number of clusters: {result.segmentation_result.num_clusters}")
            if result.segmentation_result.silhouette_score is not None:
                print(f"  Silhouette score: {result.segmentation_result.silhouette_score:.3f}")
            
            # Print cluster information
            for cluster_id in range(result.segmentation_result.num_clusters):
                cluster_segments = [s for s in result.segmentation_result.segments if s.cluster_id == cluster_id]
                total_duration = sum(s.end_time - s.start_time for s in cluster_segments)
                
                # Get dominant genre for this cluster
                genres = {}
                for segment in cluster_segments:
                    if segment.dominant_genre:
                        genre = segment.dominant_genre
                        if '---' in genre:
                            genre = genre.split('---')[1]
                        if genre not in genres:
                            genres[genre] = 0
                        genres[genre] += 1
                
                dominant_genre = max(genres.items(), key=lambda x: x[1])[0] if genres else "Unknown"
                
                print(f"  Cluster {cluster_id}: {len(cluster_segments)} segments, "
                      f"{total_duration:.1f}s total, dominant genre: {dominant_genre}")
    
    except Exception as e:
        print(f"Error in comprehensive analysis: {e}")
    
    # Method 2: Using the dedicated segmentation function
    print("\n2. Performing dedicated track segmentation...")
    try:
        segmentation_result = segment_audio_track(
            audio_path=audio_file,
            method='kmeans',
            n_clusters=4,
            segment_duration=5.0,
            discogs_model_path=discogs_model_path if os.path.exists(discogs_model_path) else None
        )
        
        print(f"Segmentation complete!")
        print(f"  Method: {segmentation_result.clustering_method}")
        print(f"  Number of clusters: {segmentation_result.num_clusters}")
        if segmentation_result.silhouette_score is not None:
            print(f"  Silhouette score: {segmentation_result.silhouette_score:.3f}")
    
    except Exception as e:
        print(f"Error in dedicated segmentation: {e}")
    
    # Method 3: Using the analyzer class directly with different clustering methods
    print("\n3. Comparing different clustering methods...")
    analyzer = ComprehensiveAnalyzer(
        enable_discogs=True,
        discogs_model_path=discogs_model_path if os.path.exists(discogs_model_path) else None
    )
    
    methods = ['kmeans', 'hierarchical']
    for method in methods:
        try:
            print(f"\n  Testing {method} clustering...")
            segmentation_result = analyzer.segment_track(
                audio_path=audio_file,
                method=method,
                n_clusters=4,
                segment_duration=5.0
            )
            
            print(f"    Number of clusters: {segmentation_result.num_clusters}")
            if segmentation_result.silhouette_score is not None:
                print(f"    Silhouette score: {segmentation_result.silhouette_score:.3f}")
            
            # Create visualization
            viz_path = output_dir / f"segmentation_{method}.png"
            analyzer.visualizer.create_segmentation_visualization(segmentation_result, viz_path)
            print(f"    Visualization saved to {viz_path}")
        
        except Exception as e:
            print(f"    Error with {method} clustering: {e}")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("Check the output directory for segmentation results and visualizations.")
    print("=" * 60)


if __name__ == "__main__":
    main()