# Discogs Genre Classification in Comprehensive Analysis

This document describes the Discogs genre classification integration in the comprehensive audio analysis module.

## Overview

The comprehensive analysis module now includes Discogs genre classification using Essentia's `TensorflowPredictEffnetDiscogs` model. This feature analyzes the audio content directly to determine genre information, rather than relying on metadata or external databases.

## Installation

The Discogs genre classification functionality is part of Essentia, but requires an additional model download:

1. Install the required packages (already needed for comprehensive analysis):
```bash
pip install essentia tensorflow
```

2. Download the Discogs model:
   - Go to [https://essentia.upf.edu/models/](https://essentia.upf.edu/models/)
   - Navigate to the `autotagging/` folder
   - Download the 'discogs-effnet-bs64-1.pb' file (this is the standard genre classification model)
   - Download the 'discogs-effnet-bs64-1.json' file (contains the genre labels)
   - Save both files to a location of your choice

The model file is required for genre classification but the comprehensive analysis will still work without it.

## Usage

### Basic Usage

You can use the Discogs genre classification with just a single line:

```python
from allin1.comprehensive_analysis import analyze_audio_comprehensive

# Analyze audio file with Discogs genre classification enabled
result = analyze_audio_comprehensive(
    "path/to/your/audio.wav",
    output_dir="analysis_output",
    enable_discogs=True
)

# Check if genre information was found
if result.discogs_info:
    print(f"Detected genres: {result.discogs_info.genres}")
```

### Advanced Usage

For more control, use the `ComprehensiveAnalyzer` class directly:

```python
from allin1.comprehensive_analysis import ComprehensiveAnalyzer

# Initialize analyzer with Discogs enabled
analyzer = ComprehensiveAnalyzer(enable_discogs=True)

# Analyze audio file
result = analyzer.analyze(
    "path/to/your/audio.wav",
    output_dir="analysis_output"
)

# Get genre information
if result.discogs_info:
    print(f"Top genres: {result.discogs_info.genres}")
```

### Disabling Discogs

If you don't need genre information, you can disable Discogs:

```python
result = analyze_audio_comprehensive(
    "path/to/your/audio.wav",
    output_dir="analysis_output",
    enable_discogs=False
)
```

## Genre Over Time Analysis

The Discogs genre classification now supports analyzing genre changes over time, which can be used as indicators for segmentation:

### Analyzing Genre Over Time

```python
from allin1.comprehensive_analysis import DiscogsAnalyzer

# Initialize the analyzer
analyzer = DiscogsAnalyzer(model_path="/path/to/discogs-effnet-bs64-1.pb")

# Analyze genre over time
overall_info, time_stamps, genre_predictions = analyzer.analyze_genre_over_time(
    "audio.wav",
    segment_duration=5.0  # 5-second segments
)

# Get the top genre for each segment
for i, timestamp in enumerate(time_stamps):
    sorted_indices = np.argsort(genre_predictions[i])[::-1]
    top_genre = analyzer.genre_labels[sorted_indices[0]]
    print(f"Time {timestamp:.1f}s: {top_genre}")
```

### Integration with Time-Based Analysis

```python
from allin1.comprehensive_analysis import ComprehensiveAnalyzer

# Initialize the analyzer
analyzer = ComprehensiveAnalyzer(
    enable_discogs=True,
    discogs_model_path="/path/to/discogs-effnet-bs64-1.pb"
)

# Analyze with genre over time
result = analyzer.analyze("audio.wav", output_dir="results/")

# Access genre predictions over time
if result.time_based_features.genre_predictions is not None:
    genre_preds = result.time_based_features.genre_predictions
    genre_labels = result.time_based_features.genre_labels
    time_stamps = result.time_based_features.time_stamps
```

### Visualizations

The comprehensive analysis now includes genre information in the visualizations:

1. **Heatmap**: Shows genre changes over time alongside other features
2. **Genre Timeline**: A dedicated visualization showing genre probability changes over time

## Features

### DiscogsAnalyzer Class

The `DiscogsAnalyzer` class provides the following features:
- Analyze audio content using Essentia's `TensorflowPredictEffnetDiscogs` model
- Extract genre probabilities from the audio
- Return top 5 genres by confidence
- Handle model initialization errors gracefully

### Supported Genres

The Discogs genre classifier can identify the following genres:
- Blues
- Classical
- Country
- Disco
- Hip-Hop
- Jazz
- Metal
- Pop
- Reggae
- Rock
- Electronic
- Folk

### Error Handling

The Discogs genre classification includes robust error handling:
- Graceful fallback when the model is unavailable
- Continues analysis even if genre classification fails
- Informative messages about model status

## Technical Details

### Model Architecture

The implementation uses Essentia's `TensorflowPredictEffnetDiscogs` model, which is based on a neural network trained on the Discogs dataset. The model analyzes mel-spectrograms extracted from the audio to determine genre probabilities.

### Audio Requirements

- Sample rate: 16kHz (handled automatically)
- Mono channel (handled automatically)
- Recommended minimum duration: 3 seconds for reliable classification

## Limitations

- The model is trained on Western music genres and may not perform well on other music traditions
- Short audio clips (< 3 seconds) may produce less reliable results
- The model provides genre classifications but not sub-genres or styles

## Examples

See `example_discogs_usage.py` for complete examples of how to use the Discogs genre classification functionality.

See `example_genre_over_time.py` for an example of how to use the genre over time analysis functionality.

## Testing

Run the test script to verify Discogs functionality:

```bash
python test_discogs.py
```

This will test the Discogs genre classification and provide feedback on any issues.