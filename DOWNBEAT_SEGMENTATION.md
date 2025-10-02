# Downbeat-Based Segmentation Implementation

## Overview

This document describes the implementation of downbeat-based segmentation for audio analysis, replacing the previous fixed-duration segmentation approach.

## Key Features

### 1. Downbeat-Based Segment Boundaries

Each segment is now calculated based on downbeats detected by Madmom:

- **Regular segments**: Time between consecutive downbeats (N to N+1)
- **First segment**:
  - If first downbeat < 0.3s: Segment from 0 to 2nd downbeat
  - Otherwise: Segment from 0 to 1st downbeat
- **Last segment**:
  - If last downbeat is within 0.3s of track end: Extend previous segment to end
  - Otherwise: Segment from last downbeat to end

### 2. Per-Segment Feature Extraction

For each segment, the following features are extracted:

- Danceability
- Energy
- Valence
- Spectral centroid
- Spectral rolloff
- Spectral bandwidth
- Zero crossing rate
- MFCC (mean)
- Chroma (mean)

### 3. Per-Segment Genre Analysis

If Discogs genre classifier is available, each segment gets:

- Genre predictions (probability distribution)
- Dominant genre label
- Confidence score

### 4. Segment Grouping

Consecutive segments with similar features are automatically grouped together:

- Uses normalized Euclidean distance between feature vectors
- Default similarity threshold: 0.15
- Creates segment groups with:
  - Start/end times
  - List of contained segment IDs
  - Averaged features
  - Dominant genre (most common across group)
  - Average genre confidence

## API Changes

### TimeBasedAnalyzer Methods

```python
# Now accepts madmom_features parameter
def extract_time_features(
    self,
    audio_path: PathLike,
    segment_duration: float = 5.0,
    madmom_features: Optional[MadmomFeatures] = None
) -> TimeBasedFeatures
```

```python
# Now accepts madmom_features parameter
def extract_time_features_with_genre(
    self,
    audio_path: PathLike,
    segment_duration: float = 5.0,
    discogs_analyzer=None,
    madmom_features: Optional[MadmomFeatures] = None
) -> TimeBasedFeatures
```

### New Data Classes

#### TimeBasedFeatures (Updated)

```python
@dataclass
class TimeBasedFeatures:
    time_stamps: np.ndarray
    features: Dict[str, np.ndarray]
    feature_names: List[str]
    genre_predictions: Optional[np.ndarray] = None
    genre_labels: Optional[List[str]] = None
    segment_boundaries: Optional[List[Tuple[float, float]]] = None  # NEW
```

#### SegmentGroup (New)

```python
@dataclass
class SegmentGroup:
    group_id: int
    start_time: float
    end_time: float
    segment_ids: List[int]
    avg_features: Dict[str, float]
    dominant_genre: Optional[str] = None
    genre_confidence: Optional[float] = None
```

#### GroupedSegmentationResult (New)

```python
@dataclass
class GroupedSegmentationResult:
    segment_groups: List[SegmentGroup]
    original_segments: List[SegmentInfo]
    num_groups: int
    feature_names: List[str]
```

#### ComprehensiveAnalysisResult (Updated)

```python
@dataclass
class ComprehensiveAnalysisResult:
    path: Path
    essentia_features: EssentiaFeatures
    madmom_features: Optional[MadmomFeatures]
    discogs_info: Optional[DiscogsInfo]
    time_based_features: TimeBasedFeatures
    original_analysis: Optional[AnalysisResult] = None
    segmentation_result: Optional[SegmentationResult] = None
    grouped_segmentation: Optional[GroupedSegmentationResult] = None  # NEW
```

### New Functions

#### calculate_downbeat_segments()

```python
def calculate_downbeat_segments(
    downbeats: np.ndarray,
    audio_duration: float,
    min_edge_duration: float = 0.3
) -> List[Tuple[float, float]]
```

Calculates segment boundaries from downbeat times with special handling for edge cases.

#### group_similar_segments()

```python
def group_similar_segments(
    segments: List[SegmentInfo],
    feature_names: List[str],
    similarity_threshold: float = 0.15
) -> GroupedSegmentationResult
```

Groups consecutive segments with similar features.

## Usage Example

```python
from src.allin1.comprehensive_analysis import analyze_audio_comprehensive

# Perform comprehensive analysis with downbeat-based segmentation
result = analyze_audio_comprehensive(
    audio_path='path/to/audio.mp3',
    output_dir='output/',
    enable_madmom=True,        # Enable Madmom for downbeat detection
    enable_discogs=True,       # Enable genre classification per segment
    enable_segmentation=True,  # Enable segmentation
    segmentation_method='kmeans',
    n_clusters=4
)

# Access downbeat information
print(f"Detected {len(result.madmom_features.downbeats)} downbeats")
print(f"Tempo: {result.madmom_features.tempo:.2f} BPM")

# Access segments
print(f"\nCreated {len(result.segmentation_result.segments)} segments")
for seg in result.segmentation_result.segments[:5]:  # First 5 segments
    print(f"Segment {seg.segment_id}: {seg.start_time:.2f}s - {seg.end_time:.2f}s")
    print(f"  Energy: {seg.features.get('energy', 0):.2f}")
    print(f"  Genre: {seg.dominant_genre} ({seg.genre_confidence:.2%})")

# Access segment groups
print(f"\nGrouped into {result.grouped_segmentation.num_groups} groups")
for group in result.grouped_segmentation.segment_groups:
    duration = group.end_time - group.start_time
    print(f"\nGroup {group.group_id}: {group.start_time:.2f}s - {group.end_time:.2f}s ({duration:.1f}s)")
    print(f"  Contains segments: {group.segment_ids}")
    print(f"  Dominant genre: {group.dominant_genre}")
    print(f"  Avg energy: {group.avg_features.get('energy', 0):.2f}")
    print(f"  Avg danceability: {group.avg_features.get('danceability', 0):.2f}")
```

## Benefits

1. **Musically Meaningful Segments**: Segments align with the musical structure (bars/measures)
2. **Variable Length**: Segments adapt to the tempo and time signature of the music
3. **Feature Consistency**: Features within downbeat-based segments are more homogeneous
4. **Better Genre Detection**: Genre changes typically happen at downbeats
5. **Hierarchical Organization**: Individual segments can be grouped into larger sections

## Fallback Behavior

If Madmom is not available or downbeat detection fails:

- System falls back to fixed-duration segmentation
- Default segment duration: 5.0 seconds
- All other functionality remains the same

## Performance Considerations

- Downbeat detection adds ~2-5 seconds per track (depending on length)
- Per-segment genre analysis scales with number of segments
- More segments = better temporal resolution but slower processing
- Typical track (3-5 minutes): 30-50 segments with 4/4 time signature

## Testing

Run the test script to verify the implementation:

```bash
~/venvs/pydemucs/bin/python test_downbeat_segmentation.py
```

This tests:

- Edge case handling (early first downbeat, late last downbeat)
- Normal downbeat segmentation
- Fallback behavior (no downbeats)
- Single downbeat case
