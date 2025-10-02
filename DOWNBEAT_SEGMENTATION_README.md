# Downbeat-Based Segmentation

This document describes the downbeat-based segmentation functionality added to the comprehensive audio analysis module.

## Overview

The downbeat-based segmentation feature divides an audio track into segments based on the downbeats detected by Madmom. Each segment represents the audio between two consecutive downbeats, with special handling for the first and last segments. The feature also groups segments into chunks of 1, 2, 4, or 8 segments (representing musical bars) and calculates feature similarities between these groups.

## Key Features

1. **Segmentation by Downbeats**: Creates segments based on the downbeats detected by Madmom
2. **Special Segment Handling**: Handles the first and last segments according to specific rules
3. **Feature Extraction**: Extracts comprehensive features for each segment using Essentia
4. **Segment Grouping**: Groups segments into chunks of 1, 2, 4, or 8 segments
5. **Similarity Analysis**: Calculates feature similarities between segment groups of the same size

## Special Segment Handling

### First Segment
- If the first downbeat is less than 0.3 seconds from the start, the first segment starts at time 0
- Otherwise, the first segment includes the audio before the first downbeat

### Last Segment
- If the last downbeat is less than 0.3 seconds from the end of the track, the final segment is connected to the previous segment
- Otherwise, a final segment is created from the last downbeat to the end of the track

## Usage

### Using the Convenience Function

```python
from allin1.comprehensive_analysis import segment_by_downbeats

# Segment an audio file
result = segment_by_downbeats(
    audio_path="path/to/audio.mp3",
    segment_group_sizes=[1, 2, 4, 8]  # Default group sizes
)

# Check for errors
if "error" in result:
    print(f"Error: {result['error']}")
else:
    print(f"Created {len(result['segments'])} segments")
```

### Using the Class Directly

```python
from allin1.comprehensive_analysis import ComprehensiveAnalyzer

# Initialize the analyzer
analyzer = ComprehensiveAnalyzer(enable_discogs=False, enable_madmom=True)

# Perform downbeat-based segmentation
result = analyzer.segment_by_downbeats(
    audio_path="path/to/audio.mp3",
    segment_group_sizes=[1, 2, 4, 8]  # Default group sizes
)
```

## Result Structure

The result is a dictionary containing the following keys:

- `segments`: List of segment dictionaries, each with:
  - `start_time`: Start time of the segment in seconds
  - `end_time`: End time of the segment in seconds
  - `segment_id`: Unique identifier for the segment
  - `is_special`: Boolean indicating if this is a special segment (first/last)
  - `features`: Dictionary of extracted features for the segment

- `segment_groups`: Dictionary of segment groups, with keys like:
  - `groups_of_1`: Groups of 1 segment each
  - `groups_of_2`: Groups of 2 segments each
  - `groups_of_4`: Groups of 4 segments each
  - `groups_of_8`: Groups of 8 segments each

  Each group contains:
  - `group_id`: Unique identifier for the group
  - `segment_indices`: List of segment IDs in this group
  - `start_time`: Start time of the group
  - `end_time`: End time of the group
  - `num_segments`: Number of segments in the group
  - `avg_features`: Average features for the group

- `similarities`: Dictionary of similarity results for each group size, with keys like:
  - `groups_of_1`: Similarity results for 1-segment groups
  - `groups_of_2`: Similarity results for 2-segment groups
  - etc.

  Each similarity result contains:
  - `most_similar_pairs`: List of most similar group pairs
  - `similarity_matrix`: Full similarity matrix

- `total_duration`: Total duration of the audio in seconds
- `num_downbeats`: Number of downbeats detected
- `downbeat_times`: List of downbeat times in seconds

## Example Output

```python
{
    "segments": [
        {
            "start_time": 0.0,
            "end_time": 2.08,
            "segment_id": 0,
            "is_special": False,
            "features": {
                "danceability": 0.65,
                "energy": 0.78,
                ...
            }
        },
        ...
    ],
    "segment_groups": {
        "groups_of_1": [
            {
                "group_id": 0,
                "segment_indices": [0],
                "start_time": 0.0,
                "end_time": 2.08,
                "num_segments": 1,
                "avg_features": {...}
            },
            ...
        ],
        "groups_of_2": [...],
        ...
    },
    "similarities": {
        "groups_of_1": {
            "most_similar_pairs": [
                {
                    "group1_id": 5,
                    "group2_id": 12,
                    "similarity": 0.987,
                    "group1_start": 10.4,
                    "group1_end": 12.48,
                    "group2_start": 24.96,
                    "group2_end": 27.04
                },
                ...
            ],
            "similarity_matrix": [...]
        },
        ...
    },
    "total_duration": 310.36,
    "num_downbeats": 157,
    "downbeat_times": [0.14, 2.08, 4.01, ...]
}
```

## Dependencies

This functionality requires:

- Madmom: For beat and downbeat detection
- Essentia: For feature extraction
- NumPy: For numerical operations
- scikit-learn: For similarity calculations

## Testing

Run the test script to verify the implementation:

```bash
~/venvs/pydemucs/bin/python test_downbeat_segmentation.py
```

## Example Script

See `example_downbeat_segmentation.py` for a complete example of how to use this functionality.