# Beat Detection Comparison: Madmom vs beat_this

This document describes the beat detection comparison functionality that compares two libraries:

- **Madmom**: Established beat tracking library using RNN-based models
- **beat_this**: Newer beat tracking library from CPJKU

## Installation

### Installing Madmom

```bash
pip install madmom
```

### Installing beat_this

beat_this has several dependencies that must be installed first:

1. **Install PyTorch 2.0 or later** (follow instructions for your platform at https://pytorch.org/)

2. **Install required packages:**

```bash
pip install tqdm einops soxr rotary-embedding-torch
```

3. **Install ffmpeg** (for audio format support beyond .wav):

```bash
# Using conda
conda install ffmpeg

# Or via your operating system (Ubuntu/Debian)
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

4. **Install beat_this:**

```bash
pip install https://github.com/CPJKU/beat_this/archive/main.zip
```

## Usage

### Running the Comparison Test

The `test_beat_comparison.py` script compares both methods and outputs:

- **Execution time** for each method
- **Number of beats and downbeats** detected
- **Tempo estimation** (BPM)
- **Beat consistency** score
- **Accuracy metrics** (if ground truth is provided)

#### Basic Usage (No Ground Truth)

```bash
python test_beat_comparison.py path/to/audio.wav
```

This will:

- Run both Madmom and beat_this on the audio file
- Measure execution time for each method
- Output beat and downbeat detections
- Save results to `beat_comparison_results.json`

#### With Ground Truth (For Accuracy Evaluation)

```bash
python test_beat_comparison.py path/to/audio.wav \
  --ground-truth-beats beats.txt \
  --ground-truth-downbeats downbeats.txt \
  --ground-truth-tempo 120.0
```

Ground truth files should be text files with one timestamp per line (in seconds).

#### Custom Output Path

```bash
python test_beat_comparison.py path/to/audio.wav -o my_results.json
```

#### Adjusting Beat Matching Tolerance

```bash
python test_beat_comparison.py path/to/audio.wav --tolerance 0.05
```

Default tolerance is 0.07 seconds (70ms). This is the time window used to match predicted beats with ground truth beats.

## Output Format

The comparison results are saved as JSON with the following structure:

```json
{
  "audio_file": "path/to/audio.wav",
  "madmom": {
    "execution_time": 2.34,
    "num_beats": 245,
    "num_downbeats": 61,
    "tempo": 120.5,
    "beat_consistency": 0.89,
    "beats": [0.12, 0.62, 1.12, ...],
    "downbeats": [0.12, 2.12, 4.12, ...],
    "beat_metrics": {
      "precision": 0.95,
      "recall": 0.93,
      "f1_score": 0.94,
      "num_predicted": 245,
      "num_reference": 240,
      "num_matched": 228
    },
    "downbeat_metrics": { ... },
    "tempo_error": {
      "absolute_error": 0.5,
      "relative_error": 0.42,
      "predicted_tempo": 120.5,
      "reference_tempo": 120.0
    }
  },
  "beat_this": {
    "execution_time": 1.87,
    "num_beats": 238,
    "num_downbeats": 59,
    "tempo": 119.8,
    "beat_consistency": 0.91,
    "beats": [0.14, 0.64, 1.14, ...],
    "downbeats": [0.14, 2.14, 4.14, ...],
    "beat_metrics": { ... },
    "downbeat_metrics": { ... },
    "tempo_error": { ... }
  },
  "comparison": {
    "time_comparison": {
      "madmom_time": 2.34,
      "beat_this_time": 1.87,
      "speedup_factor": 1.25,
      "faster_method": "beat_this"
    },
    "accuracy_comparison": {
      "madmom_beat_f1": 0.94,
      "beat_this_beat_f1": 0.92,
      "f1_difference": 0.02,
      "better_method": "madmom"
    }
  }
}
```

## Evaluation Metrics

### Accuracy Metrics

When ground truth is provided, the following metrics are calculated:

- **Precision**: Proportion of predicted beats that match ground truth (within tolerance)
- **Recall**: Proportion of ground truth beats that are detected
- **F1-Score**: Harmonic mean of precision and recall

### Tempo Error

- **Absolute Error**: Difference in BPM between predicted and ground truth
- **Relative Error**: Percentage difference

### Performance Metrics

- **Execution Time**: Wall-clock time to process the audio file
- **Speedup Factor**: Ratio of execution times (how much faster one method is)

## Integration with Comprehensive Analysis

The `BeatThisAnalyzer` is integrated into the comprehensive analysis module and can be used alongside or instead of Madmom:

```python
from allin1.comprehensive_analysis import BeatThisAnalyzer

# Initialize analyzer
analyzer = BeatThisAnalyzer()

# Extract beats and downbeats
features = analyzer.extract_beats_and_downbeats('audio.wav')

print(f"Beats: {len(features.beats)}")
print(f"Downbeats: {len(features.downbeats)}")
print(f"Tempo: {features.tempo:.2f} BPM")
print(f"Consistency: {features.beat_consistency:.3f}")
```

## Implementation Details

### Madmom Implementation

Madmom uses:

- **RNNBeatProcessor**: Recurrent Neural Network for beat activation detection
- **BeatTrackingProcessor**: Dynamic Bayesian Network for beat tracking
- **RNNDownBeatProcessor**: RNN for downbeat activation
- **DBNDownBeatTrackingProcessor**: DBN for downbeat tracking with time signature inference

### beat_this Implementation

beat_this uses:

- **BeatTracker**: State-of-the-art deep learning model for beat and downbeat tracking
- Processes audio at 22050 Hz sample rate
- Returns beat and downbeat timestamps directly

### Tempo Calculation

Both implementations calculate tempo from beat intervals using:

1. Median interval for robustness
2. Convert to BPM: `tempo = 60.0 / median_interval`

### Beat Consistency

Measures the regularity of beat intervals:

```python
beat_consistency = 1.0 - (std(intervals) / median(intervals))
```

Values range from 0 to 1, where 1 indicates perfectly regular beats.

## Typical Results

Based on testing, typical characteristics:

### Madmom

- **Pros**: Very accurate, well-tested, handles complex music
- **Cons**: Slower execution, requires more dependencies
- **Typical F1-Score**: 0.90-0.95 on well-recorded music
- **Typical Time**: 2-5 seconds for a 3-minute song

### beat_this

- **Pros**: Faster execution, modern architecture
- **Cons**: May be less accurate on very complex music
- **Typical F1-Score**: 0.88-0.94 on well-recorded music
- **Typical Time**: 1-3 seconds for a 3-minute song

## Troubleshooting

### beat_this not found

If you get an import error for beat_this, make sure all dependencies are installed:

```bash
# 1. Check PyTorch is installed (2.0 or later required)
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"

# 2. Install required packages
pip install tqdm einops soxr rotary-embedding-torch

# 3. Install beat_this from GitHub
pip install https://github.com/CPJKU/beat_this/archive/main.zip

# 4. Install ffmpeg for audio format support
# Using conda:
conda install ffmpeg

# Or using system package manager:
# Ubuntu/Debian:
sudo apt-get install ffmpeg
# macOS:
brew install ffmpeg
```

**Note:** If you see errors about missing `torch`, make sure you have PyTorch 2.0+ installed first from https://pytorch.org/

### Madmom installation issues

On some systems, Madmom may require additional dependencies:

```bash
# macOS
brew install libsndfile

# Ubuntu/Debian
sudo apt-get install libsndfile1

# Then install madmom
pip install madmom
```

### Different beat counts

It's normal for the two methods to detect slightly different numbers of beats, especially:

- At the beginning/end of the track
- During tempo changes
- In complex rhythmic passages

The F1-Score accounts for these differences and provides a fair comparison.

## References

- Madmom: https://github.com/CPJKU/madmom
- beat_this: https://github.com/CPJKU/beat_this
- Paper: "Learning to Groove with Inverse Sequence Transformations" (CPJKU)
