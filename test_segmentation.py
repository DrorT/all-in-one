"""
Test script for track segmentation functionality.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from allin1.comprehensive_analysis import (
    TimeBasedFeatures,
    TrackSegmenter,
    HeatmapVisualizer,
    ComprehensiveAnalyzer
)


def create_mock_time_features():
    """Create mock time-based features for testing"""
    # Create mock data
    num_segments = 20
    time_stamps = np.linspace(0, 100, num_segments)
    
    # Create mock features with some patterns
    features = {
        'danceability': np.random.rand(num_segments) * 0.5 + 0.25,
        'energy': np.random.rand(num_segments) * 0.5 + 0.25,
        'valence': np.random.rand(num_segments) * 0.5 + 0.25,
        'tempo': np.random.rand(num_segments) * 40 + 100,
        'spectral_centroid': np.random.rand(num_segments) * 2000 + 2000,
        'spectral_rolloff': np.random.rand(num_segments) * 2000 + 4000,
        'spectral_bandwidth': np.random.rand(num_segments) * 1000 + 1000,
        'zero_crossing_rate': np.random.rand(num_segments) * 0.1,
        'mfcc_mean': np.random.rand(num_segments, 13),
        'chroma_mean': np.random.rand(num_segments, 12)
    }
    
    # Create some patterns in the features
    # First 5 segments: high energy, high danceability
    features['energy'][:5] = np.random.rand(5) * 0.3 + 0.7
    features['danceability'][:5] = np.random.rand(5) * 0.3 + 0.7
    
    # Next 5 segments: low energy, low danceability
    features['energy'][5:10] = np.random.rand(5) * 0.3 + 0.2
    features['danceability'][5:10] = np.random.rand(5) * 0.3 + 0.2
    
    # Next 5 segments: high valence, moderate energy
    features['valence'][10:15] = np.random.rand(5) * 0.3 + 0.7
    features['energy'][10:15] = np.random.rand(5) * 0.3 + 0.5
    
    # Last 5 segments: low valence, moderate energy
    features['valence'][15:] = np.random.rand(5) * 0.3 + 0.2
    features['energy'][15:] = np.random.rand(5) * 0.3 + 0.5
    
    # Create mock genre predictions
    genre_predictions = np.random.rand(num_segments, 10)
    genre_predictions = genre_predictions / np.sum(genre_predictions, axis=1, keepdims=True)
    
    # Create genre labels
    genre_labels = [
        'Rock---Classic Rock', 'Pop---Dance Pop', 'Electronic---House',
        'Jazz---Smooth Jazz', 'Classical---Baroque', 'Hip-Hop---Old School',
        'Country---Modern Country', 'Blues---Chicago Blues', 'Folk---Indie Folk',
        'Metal---Heavy Metal'
    ]
    
    # Add some patterns to genre predictions
    # First 5 segments: more likely to be Rock
    genre_predictions[:5, 0] = np.random.rand(5) * 0.3 + 0.7
    
    # Next 5 segments: more likely to be Jazz
    genre_predictions[5:10, 3] = np.random.rand(5) * 0.3 + 0.7
    
    # Next 5 segments: more likely to be Pop
    genre_predictions[10:15, 1] = np.random.rand(5) * 0.3 + 0.7
    
    # Last 5 segments: more likely to be Classical
    genre_predictions[15:, 4] = np.random.rand(5) * 0.3 + 0.7
    
    # Normalize again
    genre_predictions = genre_predictions / np.sum(genre_predictions, axis=1, keepdims=True)
    
    return TimeBasedFeatures(
        time_stamps=time_stamps,
        features=features,
        feature_names=list(features.keys()),
        genre_predictions=genre_predictions,
        genre_labels=genre_labels
    )


def test_track_segmenter():
    """Test the TrackSegmenter class"""
    print("Testing TrackSegmenter...")
    
    # Create mock time-based features
    time_features = create_mock_time_features()
    
    # Test with K-means
    print("  Testing K-means clustering...")
    segmenter = TrackSegmenter(method='kmeans', n_clusters=4, include_genre=True)
    result = segmenter.segment_track(time_features)
    
    print(f"    Number of clusters: {result.num_clusters}")
    print(f"    Number of segments: {len(result.segments)}")
    print(f"    Clustering method: {result.clustering_method}")
    if result.silhouette_score is not None:
        print(f"    Silhouette score: {result.silhouette_score:.3f}")
    
    # Test with Hierarchical clustering
    print("  Testing Hierarchical clustering...")
    segmenter = TrackSegmenter(method='hierarchical', n_clusters=4, include_genre=True)
    result = segmenter.segment_track(time_features)
    
    print(f"    Number of clusters: {result.num_clusters}")
    if result.silhouette_score is not None:
        print(f"    Silhouette score: {result.silhouette_score:.3f}")
    
    # Test with DBSCAN
    print("  Testing DBSCAN clustering...")
    segmenter = TrackSegmenter(method='dbscan', include_genre=True)
    result = segmenter.segment_track(time_features)
    
    print(f"    Number of clusters: {result.num_clusters}")
    if result.silhouette_score is not None:
        print(f"    Silhouette score: {result.silhouette_score:.3f}")
    
    print("  TrackSegmenter test complete!")
    return result


def test_visualization():
    """Test the visualization functionality"""
    print("Testing visualization...")
    
    # Create mock time-based features
    time_features = create_mock_time_features()
    
    # Create segmenter and perform segmentation
    segmenter = TrackSegmenter(method='kmeans', n_clusters=4, include_genre=True)
    result = segmenter.segment_track(time_features)
    
    # Create visualizer
    visualizer = HeatmapVisualizer()
    
    # Test segmentation visualization
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    viz_path = output_dir / "test_segmentation.png"
    fig = visualizer.create_segmentation_visualization(result, viz_path)
    
    if fig is not None:
        print(f"  Segmentation visualization saved to {viz_path}")
        print("  Visualization test complete!")
        return True
    else:
        print("  Visualization test failed!")
        return False


def test_comprehensive_analyzer():
    """Test the ComprehensiveAnalyzer class with segmentation"""
    print("Testing ComprehensiveAnalyzer with segmentation...")
    
    # Create analyzer
    analyzer = ComprehensiveAnalyzer(enable_discogs=False)
    
    # Create mock time-based features
    time_features = create_mock_time_features()
    
    # Create segmenter
    analyzer.segmenter = TrackSegmenter(method='kmeans', n_clusters=4, include_genre=False)
    
    # Perform segmentation
    result = analyzer.segmenter.segment_track(time_features)
    
    print(f"    Number of clusters: {result.num_clusters}")
    if result.silhouette_score is not None:
        print(f"    Silhouette score: {result.silhouette_score:.3f}")
    
    print("  ComprehensiveAnalyzer test complete!")
    return result


def main():
    """Run all tests"""
    print("=" * 60)
    print("Running Track Segmentation Tests")
    print("=" * 60)
    
    try:
        # Test TrackSegmenter
        segmentation_result = test_track_segmenter()
        
        # Test visualization
        viz_success = test_visualization()
        
        # Test ComprehensiveAnalyzer
        analyzer_result = test_comprehensive_analyzer()
        
        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("=" * 60)
        
        return True
    
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)