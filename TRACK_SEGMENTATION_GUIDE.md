# Track Segmentation Guide

This guide explains how to use the track segmentation functionality in the comprehensive analysis module to divide a track into segments with similar features.

## Overview

Track segmentation uses clustering algorithms to group time-based segments of an audio file based on their feature similarity. This can be useful for:

- Identifying different sections of a song (verse, chorus, bridge, etc.)
- Finding structural changes in the music
- Creating playlists based on similar segments
- Analyzing how musical characteristics change over time

## Features

The track segmentation functionality includes:

1. **Multiple Clustering Methods**:
   - K-means clustering
   - DBSCAN clustering
   - Hierarchical clustering

2. **Feature Types**:
   - Basic musical features (danceability, energy, valence, tempo)
   - Spectral features (spectral centroid, rolloff, bandwidth, zero crossing rate)
   - Genre predictions (if Discogs analyzer is enabled)

3. **Visualization**:
   - Timeline visualization of segments
   - Feature heatmap by cluster
   - Cluster statistics and information

## Basic Usage

### Method 1: Using the Convenience Function

```python
from allin1.comprehensive_analysis import analyze_audio_comprehensive

# Perform comprehensive analysis with segmentation enabled
result = analyze_audio_comprehensive(
    audio_path="path/to/your/audio.mp3",
    output_dir="results",
    enable_discogs=True,
    enable_segmentation=True,
    segmentation_method='kmeans',
    n_clusters=4
)

# Access segmentation results
if result.segmentation_result:
    print(f"Track segmented into {result.segmentation_result.num_clusters} clusters")
    for segment in result.segmentation_result.segments:
        print(f"Segment {segment.segment_id}: {segment.start_time:.1f}s - {segment.end_time:.1f}s, "
              f"Cluster {segment.cluster_id}, Genre: {segment.dominant_genre}")
```

### Method 2: Using the Dedicated Segmentation Function

```python
from allin1.comprehensive_analysis import segment_audio_track

# Perform dedicated track segmentation
segmentation_result = segment_audio_track(
    audio_path="path/to/your/audio.mp3",
    method='kmeans',
    n_clusters=4,
    segment_duration=5.0
)

print(f"Track segmented into {segmentation_result.num_clusters} clusters")
```

### Method 3: Using the Analyzer Class Directly

```python
from allin1.comprehensive_analysis import ComprehensiveAnalyzer

# Initialize analyzer
analyzer = ComprehensiveAnalyzer(enable_discogs=True)

# Perform segmentation
segmentation_result = analyzer.segment_track(
    audio_path="path/to/your/audio.mp3",
    method='kmeans',
    n_clusters=4,
    segment_duration=5.0
)

# Create visualization
viz_path = "segmentation_visualization.png"
analyzer.visualizer.create_segmentation_visualization(segmentation_result, viz_path)
```

## Advanced Usage

### Customizing Clustering Parameters

```python
from allin1.comprehensive_analysis import TrackSegmenter, TimeBasedAnalyzer, ComprehensiveAnalyzer

# Initialize analyzer
analyzer = ComprehensiveAnalyzer(enable_discogs=True)

# Extract time-based features
time_features = analyzer.time_analyzer.extract_time_features_with_genre(
    audio_path="path/to/your/audio.mp3",
    discogs_analyzer=analyzer.discogs_analyzer,
    segment_duration=5.0
)

# Initialize segmenter with custom parameters
segmenter = TrackSegmenter(
    method='kmeans',
    n_clusters=5,
    include_genre=True,
    scaler='standard'  # Options: 'standard', 'minmax', 'none'
)

# Perform segmentation
segmentation_result = segmenter.segment_track(time_features)
```

### Comparing Different Clustering Methods

```python
from allin1.comprehensive_analysis import ComprehensiveAnalyzer

# Initialize analyzer
analyzer = ComprehensiveAnalyzer(enable_discogs=True)

# Test different clustering methods
methods = ['kmeans', 'hierarchical', 'dbscan']
results = {}

for method in methods:
    try:
        segmentation_result = analyzer.segment_track(
            audio_path="path/to/your/audio.mp3",
            method=method,
            n_clusters=4,
            segment_duration=5.0
        )
        
        results[method] = segmentation_result
        print(f"{method}: {segmentation_result.num_clusters} clusters, "
              f"Silhouette score: {segmentation_result.silhouette_score:.3f}")
        
        # Create visualization
        viz_path = f"segmentation_{method}.png"
        analyzer.visualizer.create_segmentation_visualization(segmentation_result, viz_path)
        
    except Exception as e:
        print(f"Error with {method}: {e}")
```

## Understanding the Results

### SegmentationResult

The `SegmentationResult` object contains:

- `segments`: List of `SegmentInfo` objects, one for each time segment
- `num_clusters`: Number of clusters identified
- `cluster_labels`: Array of cluster IDs for each segment
- `feature_names`: Names of features used for clustering
- `clustering_method`: The clustering method used
- `silhouette_score`: Quality measure of the clustering (higher is better)

### SegmentInfo

Each `SegmentInfo` object contains:

- `start_time`: Start time of the segment in seconds
- `end_time`: End time of the segment in seconds
- `segment_id`: Unique ID for the segment
- `cluster_id`: ID of the cluster the segment belongs to
- `features`: Dictionary of feature values for this segment
- `dominant_genre`: Most likely genre for this segment (if genre analysis is enabled)
- `genre_confidence`: Confidence in the dominant genre prediction

## Choosing the Right Clustering Method

### K-means
- **Best for**: When you know the number of clusters you want
- **Pros**: Fast, simple, works well with globular clusters
- **Cons**: Requires specifying the number of clusters, sensitive to initialization

### DBSCAN
- **Best for**: When you don't know the number of clusters and want to find noise
- **Pros**: Can find arbitrarily shaped clusters, identifies noise points
- **Cons**: Sensitive to parameters (eps, min_samples), may struggle with varying density

### Hierarchical
- **Best for**: When you want a hierarchy of clusters
- **Pros**: Doesn't require specifying the number of clusters, provides dendrogram
- **Cons**: Computationally expensive for large datasets, once merged can't be undone

## Tips for Better Segmentation

1. **Adjust Segment Duration**:
   - Shorter segments (2-3 seconds) capture more detail but may be noisy
   - Longer segments (8-10 seconds) are more stable but may miss transitions

2. **Choose the Right Number of Clusters**:
   - For pop songs, 3-4 clusters often work well (verse, chorus, bridge, outro)
   - For classical music, you might need more clusters for different movements

3. **Experiment with Different Features**:
   - Including genre information can improve segmentation for tracks with genre changes
   - For tracks with subtle changes, you might want to focus on spectral features

4. **Post-process the Results**:
   - Merge consecutive segments with the same cluster ID
   - Filter out very short segments that might be noise

## Example Output

When you run the segmentation, you'll get output like this:

```
Track segmented into 4 clusters
Silhouette score: 0.256

Cluster information:
Cluster 0: 5 segments, 25.0s total, dominant genre: Rock (3 segments)
Cluster 1: 4 segments, 20.0s total, dominant genre: Electronic (2 segments)
Cluster 2: 3 segments, 15.0s total, dominant genre: Jazz (2 segments)
Cluster 3: 2 segments, 10.0s total, dominant genre: Pop (1 segment)
```

And you'll get visualizations like:

1. **Segment Timeline**: Shows the track divided into colored segments by cluster
2. **Feature Heatmap**: Shows feature values for each segment, sorted by cluster
3. **Cluster Information**: Text summary of each cluster with statistics

## Troubleshooting

### Issue: All segments in one cluster
- **Possible cause**: Features are too similar or clustering parameters need adjustment
- **Solution**: Try fewer clusters, different clustering method, or check if features are normalized properly

### Issue: Too many clusters
- **Possible cause**: DBSCAN is detecting noise or parameters are too sensitive
- **Solution**: Try K-means or hierarchical clustering with a fixed number of clusters

### Issue: Segmentation doesn't match musical sections
- **Possible cause**: Features don't capture the musical characteristics that define sections
- **Solution**: Try different segment durations or focus on specific features like tempo or energy

## References

For more information about the algorithms used:

- [K-means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [DBSCAN Clustering](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- [Hierarchical Clustering](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- [Silhouette Score](https://scikit-learn.org/stable/modules/clustering.html#silhouette-coefficient)