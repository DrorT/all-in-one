# Madmom Tempo Calculation Improvements

## Overview

The original Madmom BPM calculation in `comprehensive_analysis.py` used a simple mean-based approach that could be inaccurate for tracks with tempo variations, beat tracking errors, or outliers. This document explains the improvements made to provide more accurate and robust tempo estimation.

**Important Update**: With these improvements, Madmom is now the sole provider of rhythm features (tempo, beats, and downbeats) in the comprehensive analysis. Essentia's rhythm extraction has been removed to avoid conflicts and ensure consistency.

## Original Implementation Issues

The original implementation (lines 436-443) calculated BPM using:

```python
# Calculate beat intervals
beat_intervals = np.diff(beat_times)
avg_interval = np.mean(beat_intervals)

# Calculate tempo in BPM
tempo_value = 60.0 / avg_interval if avg_interval > 0 else 0.0
```

### Problems with this approach:

1. **Sensitive to outliers**: A few incorrect beat detections could significantly skew the average
2. **No tempo change handling**: Assumes constant tempo throughout the track
3. **Vulnerable to missing beats**: Missing or extra beats directly affect the calculation
4. **Not using Madmom's strengths**: Madmom has more sophisticated tempo estimation methods

## Improved Implementation

The new implementation uses multiple approaches and combines them intelligently:

### 1. Multiple Estimation Methods

#### Median-based Estimation
```python
median_interval = np.median(beat_intervals)
tempo_from_median = 60.0 / median_interval if median_interval > 0 else 0.0
```
- More robust against outliers than mean
- Provides a good baseline estimation

#### Histogram-based Estimation
```python
# Create histogram of intervals (focus on reasonable tempo range: 60-200 BPM)
valid_intervals = beat_intervals[(beat_intervals > 0.3) & (beat_intervals < 1.0)]
hist_counts, hist_bins = np.histogram(valid_intervals, bins=50)
most_common_bin_idx = np.argmax(hist_counts)
most_common_interval = (hist_bins[most_common_bin_idx] + hist_bins[most_common_bin_idx + 1]) / 2
tempo_from_histogram = 60.0 / most_common_interval
```
- Finds the most common beat interval
- Robust against outliers and tempo variations
- Focuses on musically reasonable tempo range (60-200 BPM)

#### Weighted Average Estimation
```python
# Calculate local variance for each interval
local_variances = []
for i in range(len(beat_intervals)):
    start_idx = max(0, i - 2)
    end_idx = min(len(beat_intervals), i + 3)
    local_intervals = beat_intervals[start_idx:end_idx]
    if len(local_intervals) > 1:
        local_variances.append(np.var(local_intervals))
    else:
        local_variances.append(0.0)

# Use inverse variance as weights
weights = 1.0 / (np.array(local_variances) + 1e-6)
weighted_interval = np.average(beat_intervals, weights=weights)
tempo_weighted = 60.0 / weighted_interval
```
- Gives more weight to consistent beat regions
- Reduces influence of tempo changes or beat tracking errors

### 2. Intelligent Method Combination

```python
# Check histogram reliability by looking at the peak-to-average ratio
if len(valid_intervals) > 0:
    peak_to_avg = hist_counts[most_common_bin_idx] / (np.mean(hist_counts) + 1e-6)
    if peak_to_avg > 2.0:  # Strong peak in histogram
        tempo_value = tempo_from_histogram
    else:
        # Use weighted average of median and weighted tempo
        tempo_value = 0.6 * tempo_from_median + 0.4 * tempo_weighted
```

- Uses histogram method if there's a clear peak in beat intervals
- Otherwise combines median and weighted methods
- Falls back to median if tempo seems unreasonable

### 3. Integration with Madmom's Native Tempo Estimation

The improved implementation also integrates Madmom's dedicated tempo estimation processors:

```python
# Use RNNTempoProcessor for more accurate tempo estimation
tempo_processor = RNNTempoProcessor(fps=100)
tempo_acts = tempo_processor(signal)

# Use TempoEstimationProcessor to extract BPM from activations
tempo_estimator = TempoEstimationProcessor(fps=100)
tempo_values = tempo_estimator(tempo_acts)

if len(tempo_values) > 0:
    madmom_tempo = float(tempo_values[0])
```

And includes octave error correction:

```python
# Check for octave error (tempo is double or half)
if abs(tempo_value - madmom_tempo * 2) < abs(tempo_value - madmom_tempo):
    tempo_value = madmom_tempo * 2
    print(f"Corrected octave error (tempo doubled): {tempo_value:.2f} BPM")
elif abs(tempo_value - madmom_tempo / 2) < abs(tempo_value - madmom_tempo):
    tempo_value = madmom_tempo / 2
    print(f"Corrected octave error (tempo halved): {tempo_value:.2f} BPM")
```

## Benefits of the Improved Implementation

1. **More accurate**: Multiple methods provide better accuracy across different music types
2. **Robust to outliers**: Median and histogram methods are less affected by beat tracking errors
3. **Handles tempo variations**: Weighted method accounts for local consistency
4. **Octave error correction**: Detects and corrects common octave errors in tempo estimation
5. **Utilizes Madmom's strengths**: Incorporates Madmom's dedicated tempo estimation when available

## Usage

The improved tempo calculation is used automatically when calling `extract_beats_and_downbeats()`. You can also control whether to use Madmom's native tempo estimation:

```python
analyzer = MadmomAnalyzer()
features = analyzer.extract_beats_and_downbeats(audio_path, use_madmom_tempo=True)
```

## Testing

Run the test script to compare the old and new methods:

```bash
python test_madmom_tempo_improvement.py
```

This will:
1. Test the improved method on available audio files
2. Compare results with the original simple method
3. Generate visualizations showing the differences

## Expected Improvements

The improved implementation should provide:
- More accurate tempo estimates, especially for tracks with tempo variations
- Better handling of beat tracking errors
- Reduced influence of outliers
- More consistent results across different music genres
- Elimination of conflicts between Essentia and Madmom rhythm features

For tracks with very irregular rhythms or significant tempo changes, the histogram method may not always be optimal, but the fallback to median-based estimation ensures reasonable results.

## Integration with Essentia Features

While Essentia no longer extracts rhythm features, Madmom's superior rhythm information is now integrated into the EssentiaFeatures dataclass:

1. **Tempo**: Madmom's accurate tempo estimation replaces Essentia's tempo
2. **Beats**: Madmom's beat positions replace Essentia's beat positions
3. **Time Signature**: Recalculated using Madmom's beats and tempo

This ensures that all existing code that references `essentia_features.tempo`, `essentia_features.beats`, or `essentia_features.time_signature` will automatically use Madmom's superior rhythm information.