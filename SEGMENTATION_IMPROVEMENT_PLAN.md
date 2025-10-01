# Track Segmentation Improvement Plan

## Current Issues

1. **Segment Overlap**: There's a 2.5-second overlap between segments, which can cause redundant feature analysis.
2. **Fixed Time Intervals**: Segments are analyzed at fixed 5-second intervals (2.5, 7.5, 12.5, etc.) that don't align with musical features like beats or bars.

## Proposed Solution

### 1. Music-Aligned Segmentation

Instead of using fixed time intervals, we'll use musical features to create segments that align with the music structure:

1. **Extract Downbeats**: Use Essentia's RhythmExtractor to identify downbeats in the music.
2. **Determine Tempo and Time Signature**: Calculate the BPM and time signature to understand the musical structure.
3. **Create Music-Aligned Segments**: Generate segments based on musical bars (e.g., every 1, 2, 4, or 8 bars).

### 2. Eliminate Segment Overlap

We'll modify the segmentation logic to create non-overlapping segments that cover the entire track without redundancy.

## Implementation Details

### New TimeBasedAnalyzer Methods

1. `extract_downbeats(audio_path)`: Extract downbeat positions from the audio.
2. `calculate_musical_segments(audio_path, bars_per_segment=4)`: Create segments based on musical bars.
3. `extract_time_features_music_aligned(audio_path, bars_per_segment=4)`: Extract features using music-aligned segments.

### Modified TrackSegmenter

1. Update the segment_track method to handle music-aligned segments.
2. Ensure segments are non-overlapping.
3. Add options to control the number of bars per segment.

### Command-Line Options

1. `--segment-type`: Choose between 'time' (fixed intervals) or 'music' (music-aligned).
2. `--bars-per-segment`: Number of bars per segment when using music-aligned segmentation (default: 4).

## Code Changes Required

### 1. TimeBasedAnalyzer Class

```python
def extract_downbeats(self, audio_path: PathLike) -> Tuple[np.ndarray, float, int]:
    """
    Extract downbeat positions, tempo, and time signature
    
    Returns:
        downbeats: Array of downbeat times in seconds
        tempo: BPM
        time_signature: Time signature (e.g., 4 for 4/4)
    """

def create_music_segments(self, downbeats: np.ndarray, tempo: float, 
                         time_signature: int, bars_per_segment: int = 4) -> List[Tuple[float, float]]:
    """
    Create music-aligned segments based on downbeats
    
    Returns:
        segments: List of (start_time, end_time) tuples
    """

def extract_time_features_music_aligned(self, audio_path: PathLike, 
                                       bars_per_segment: int = 4) -> TimeBasedFeatures:
    """
    Extract time-based features using music-aligned segments
    """
```

### 2. TrackSegmenter Class

```python
def segment_track(self, time_based_features: TimeBasedFeatures,
                 segment_type: str = 'music') -> SegmentationResult:
    """
    Segment the track based on feature similarity
    
    Parameters:
        segment_type: 'time' for fixed intervals, 'music' for music-aligned
    """
```

### 3. Command-Line Interface

```python
parser.add_argument(
    "--segment-type", "-st",
    type=str,
    choices=['time', 'music'],
    default='music',
    help="Type of segmentation (time or music) (default: music)"
)

parser.add_argument(
    "--bars-per-segment", "-bps",
    type=int,
    default=4,
    help="Number of bars per segment for music-aligned segmentation (default: 4)"
)
```

## Benefits of Music-Aligned Segmentation

1. **Musically Relevant**: Segments align with the musical structure, making them more meaningful.
2. **Better Feature Analysis**: Features are extracted from complete musical phrases, providing more consistent results.
3. **Improved Segmentation**: Clustering based on musically relevant segments should produce more meaningful groupings.
4. **No Overlap**: Eliminates redundant feature analysis from overlapping segments.

## Implementation Steps

1. Implement the new methods in TimeBasedAnalyzer to extract downbeats and create music-aligned segments.
2. Modify the extract_time_features method to support music-aligned segmentation.
3. Update TrackSegmenter to handle music-aligned segments.
4. Add command-line options to control segmentation type.
5. Update documentation and examples.
6. Test with various musical genres to ensure robustness.

## Example Usage

```bash
# Use music-aligned segmentation with 4 bars per segment
python comprehensive_audio_analyzer.py --input song.wav --output results/ --segment-type music --bars-per-segment 4 --enable-segmentation

# Use fixed time intervals (current behavior)
python comprehensive_audio_analyzer.py --input song.wav --output results/ --segment-type time --segment-duration 5.0 --enable-segmentation
```

## Expected Output

With music-aligned segmentation, the time_stamps would look like:

```json
"time_stamps": [
  0.0,    # Start of first bar
  4.2,    # Start of 5th bar (assuming ~1.05s per bar at 57 BPM)
  8.4,    # Start of 9th bar
  12.6,   # Start of 13th bar
  ...
]
```

Instead of the current fixed intervals:

```json
"time_stamps": [
  2.5,
  7.5,
  12.5,
  17.5,
  ...
]
```

This approach ensures that each segment contains a complete musical phrase, which should result in more meaningful feature extraction and segmentation.