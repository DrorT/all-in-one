# Quick Reference: Downbeat-Based Segmentation

## How Segments Are Created

```
Audio Track (e.g., 180 seconds)
   ↓
Madmom Downbeat Detection
   ↓
Downbeats at: [2.0s, 4.0s, 6.0s, ..., 178.0s]
   ↓
Segment Boundaries Calculated:
   - Segment 0: 0.0s → 2.0s     (before first downbeat)
   - Segment 1: 2.0s → 4.0s     (downbeat 1 → 2)
   - Segment 2: 4.0s → 6.0s     (downbeat 2 → 3)
   - ...
   - Segment N: 178.0s → 180.0s (after last downbeat)
   ↓
Features Extracted Per Segment
   ↓
Genre Analyzed Per Segment (optional)
   ↓
Segments Clustered (optional)
   ↓
Similar Consecutive Segments Grouped
```

## Edge Cases

### First Downbeat < 0.3s
```python
# Input
downbeats = [0.15, 2.0, 4.0, 6.0]
audio_duration = 8.0

# Output: First segment from 0 to 2nd downbeat
segments = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]
```

### Last Downbeat Close to End (< 0.3s remaining)
```python
# Input
downbeats = [2.0, 4.0, 6.0, 7.85]
audio_duration = 8.0

# Output: Last segment extended to end
segments = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0)]
```

## Code Examples

### Basic Analysis
```python
from src.allin1.comprehensive_analysis import analyze_audio_comprehensive

result = analyze_audio_comprehensive(
    audio_path='song.mp3',
    enable_madmom=True,
    enable_segmentation=True
)

# Access segments
for seg in result.segmentation_result.segments:
    print(f"Segment {seg.segment_id}: "
          f"{seg.start_time:.2f}s - {seg.end_time:.2f}s, "
          f"Energy: {seg.features['energy']:.2f}")
```

### Access Segment Groups
```python
# Groups are automatically created
for group in result.grouped_segmentation.segment_groups:
    duration = group.end_time - group.start_time
    print(f"Group {group.group_id}: {duration:.1f}s, "
          f"{len(group.segment_ids)} segments, "
          f"Genre: {group.dominant_genre}")
```

### Custom Similarity Threshold
```python
from src.allin1.comprehensive_analysis import group_similar_segments

# Re-group with different threshold
grouped = group_similar_segments(
    segments=result.segmentation_result.segments,
    feature_names=result.segmentation_result.feature_names,
    similarity_threshold=0.10  # Stricter (lower = more groups)
)
```

## Available Features Per Segment

### Audio Features
- `danceability` - How suitable for dancing (0-1)
- `energy` - Intensity and activity (0-1)
- `valence` - Musical positiveness (0-1)
- `spectral_centroid` - Brightness of sound
- `spectral_rolloff` - Frequency rolloff point
- `spectral_bandwidth` - Spectral width
- `zero_crossing_rate` - Noisiness indicator
- `mfcc_mean` - Timbre characteristics
- `chroma_mean` - Pitch class profile

### Genre Features (if enabled)
- `dominant_genre` - Most likely genre
- `genre_confidence` - Confidence score (0-1)

## Command Line Usage

### Run Example Script
```bash
~/venvs/pydemucs/bin/python example_downbeat_segmentation.py song.mp3 output/
```

### Run Tests
```bash
~/venvs/pydemucs/bin/python test_downbeat_segmentation.py
```

## Output Structure

```python
ComprehensiveAnalysisResult
├── madmom_features
│   ├── beats (array)
│   ├── downbeats (array)
│   ├── tempo (float)
│   └── beat_consistency (float)
├── segmentation_result
│   ├── segments (List[SegmentInfo])
│   │   ├── segment_id
│   │   ├── start_time, end_time
│   │   ├── cluster_id
│   │   ├── features (dict)
│   │   ├── dominant_genre
│   │   └── genre_confidence
│   ├── num_clusters
│   └── silhouette_score
└── grouped_segmentation
    ├── segment_groups (List[SegmentGroup])
    │   ├── group_id
    │   ├── start_time, end_time
    │   ├── segment_ids (list)
    │   ├── avg_features (dict)
    │   ├── dominant_genre
    │   └── genre_confidence
    └── num_groups
```

## Tips & Best Practices

1. **Enable Madmom** for best results
2. **Use hierarchical clustering** for most musical data
3. **Adjust similarity threshold** (0.10-0.20) based on needs:
   - Lower = more groups (stricter)
   - Higher = fewer groups (looser)
4. **Check beat_consistency**: < 0.5 may indicate variable tempo
5. **Enable genre analysis** for richer segment descriptions

## Troubleshooting

### Madmom Not Available
```
Solution: Install with pip install madmom
Fallback: System uses fixed-duration segments
```

### Too Many/Few Segments
```
Check: Downbeat detection accuracy
Adjust: Use different time signature in DBNDownBeatTrackingProcessor
```

### Groups Don't Make Sense
```
Adjust: similarity_threshold parameter
Lower (e.g., 0.10): More, smaller groups
Higher (e.g., 0.25): Fewer, larger groups
```

## Performance Notes

- Typical track (3 min): ~5-10 seconds total
- Downbeat detection: Most expensive step
- Per-segment genre: Adds ~0.1s per segment
- Grouping: Very fast (< 0.1s)

## Related Files

- `DOWNBEAT_SEGMENTATION.md` - Full API documentation
- `DOWNBEAT_IMPLEMENTATION_SUMMARY.md` - Technical details
- `IMPLEMENTATION_COMPLETE.md` - Overview and verification
