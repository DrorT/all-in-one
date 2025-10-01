# Comprehensive Audio Analysis JSON Structure Explanation

This document provides a detailed explanation of all values in the comprehensive analysis JSON output from the all-in-one audio analysis tool.

## Overview

The comprehensive analysis combines multiple audio analysis techniques to provide a complete picture of a music track's characteristics. It uses three main approaches:

1. **Essentia Features** - Extracts musical characteristics using the Essentia library
2. **Discogs Genre Classification** - Identifies genres using a neural network trained on Discogs metadata
3. **Time-Based Analysis** - Tracks how features change over time by segmenting the audio

## JSON Structure

```json
{
  "path": "string",
  "essentia_features": { ... },
  "discogs_info": { ... },
  "time_based_features": { ... },
  "original_analysis": null
}
```

---

## 1. Path Information

```json
"path": "/path/to/audio/file.m4a"
```

- **Meaning**: The file path of the analyzed audio
- **Type**: String
- **Usage**: Reference to the source audio file

---

## 2. Essentia Features

Essentia is a comprehensive audio analysis library that extracts musical features. These features provide insights into various aspects of the music.

### 2.1 Basic Musical Features

| Feature | Type | Range | Meaning |
|---------|------|-------|---------|
| `danceability` | Float | 0.0-1.0+ | How suitable the track is for dancing. Higher values indicate more danceable rhythm |
| `energy` | Float | 0.0+ | Perceptual measure of intensity and activity. Energetic tracks feel fast, loud, and noisy |
| `loudness` | Float | 0.0+ | Overall loudness of the track in decibels (dB) |
| `valence` | Float | 0.0-1.0 | Musical positiveness/happiness. High valence sounds more positive (happy, cheerful), low valence sounds more negative (sad, angry) |
| `acousticness` | Float | 0.0-1.0 | Confidence that the track is acoustic. 1.0 indicates high confidence the track is acoustic |
| `instrumentalness` | Float | 0.0-1.0 | Confidence that the track has no vocals. Values closer to 1.0 indicate instrumental music |

### 2.2 Rhythm and Tempo Features

| Feature | Type | Meaning |
|---------|------|---------|
| `tempo` | Float | The speed of the music in beats per minute (BPM) |
| `time_signature` | Integer | Number of beats per measure (e.g., 4 for 4/4 time, 3 for 3/4 waltz) |
| `beats` | Array | Array of beat positions in seconds. Represents the timing of detected beats throughout the track |

### 2.3 Key and Mode

| Feature | Type | Meaning |
|---------|------|---------|
| `key` | Integer (0-11) | Musical key: 0=C, 1=C#/Db, 2=D, 3=D#/Eb, 4=E, 5=F, 6=F#/Gb, 7=G, 8=G#/Ab, 9=A, 10=A#/Bb, 11=B |
| `mode` | Integer (0-1) | Musical scale: 0=minor, 1=major |

### 2.4 Spectral Features

| Feature | Type | Meaning |
|---------|------|---------|
| `spectral_centroid` | Float | Center of mass of the spectrum. Higher values indicate brighter, more treble-heavy sound |
| `spectral_rolloff` | Float | Frequency below which 85% of the spectral energy is contained |
| `spectral_bandwidth` | Float | Standard deviation of the spectral centroid. Indicates the width of the frequency spectrum |
| `zero_crossing_rate` | Float | Rate at which the signal changes from positive to negative. Higher values indicate noisier sound |

### 2.5 Advanced Features

#### MFCC (Mel-Frequency Cepstral Coefficients)
```json
"mfcc": [13 coefficients]
```
- **Shape**: Array of 13 values (or matrix in time-based analysis)
- **Meaning**: Represents timbral qualities of the sound. Each coefficient captures different aspects of the spectral envelope
- **Usage**: Commonly used in speech and music recognition to characterize sound quality

#### Chroma Features
```json
"chroma": [[12 pitch classes], [time frames]]
```
- **Shape**: 12 rows × N columns matrix
- **Meaning**: Represents the energy of each of the 12 pitch classes (C, C#, D, ..., B) over time
- **Usage**: Useful for harmony analysis, chord recognition, and key detection

#### Tonal Features
```json
"tonal": {
  "chords_changes_rate": Float,
  "chords_histogram": Array[24],
  "chords_key": String,
  "chords_number_rate": Float,
  "chords_progression": Array,
  "chords_scale": String,
  "chords_strength": Array,
  "hpcp": Array,
  "hpcp_highres": Array,
  "key_key": String,
  "key_scale": String,
  "key_strength": Float,
  "rhythm_confidence": Float
}
```

| Feature | Meaning |
|---------|---------|
| `chords_changes_rate` | Rate of chord changes throughout the track |
| `chords_histogram` | Distribution of different chords detected |
| `chords_key` | Detected key based on chord analysis |
| `chords_number_rate` | Density of chord changes |
| `chords_progression` | Sequence of detected chords over time |
| `chords_scale` | Detected scale (major/minor) from chords |
| `chords_strength` | Confidence/strength of chord detections |
| `hpcp` | Harmonic Pitch Class Profile - chroma-like representation |
| `hpcp_highres` | High-resolution harmonic representation |
| `key_key` | Detected musical key from tonal analysis |
| `key_scale` | Detected scale from tonal analysis |
| `key_strength` | Confidence in key detection |
| `rhythm_confidence` | Confidence in rhythm analysis |

---

## 3. Discogs Genre Information

Discogs is a music database and marketplace. The genre classification uses a neural network trained on Discogs metadata to predict genres from audio content.

```json
"discogs_info": {
  "genres": ["Rock---Hard Rock", "Rock---Arena Rock", ...],
  "styles": [],
  "year": null,
  "title": "Track Name",
  "artist": null
}
```

| Feature | Type | Meaning |
|---------|------|---------|
| `genres` | Array of strings | Predicted music genres (up to 5 most likely). Format: "Main Genre---Subgenre" |
| `styles` | Array of strings | More specific style classifications (often empty) |
| `year` | Integer/null | Release year (not available from audio analysis) |
| `title` | String | Track title (extracted from filename) |
| `artist` | String/null | Artist name (not available from audio analysis) |

---

## 4. Time-Based Features

Time-based features track how musical characteristics change over time. The audio is segmented into chunks (typically 5 seconds each) and features are extracted for each segment.

### 4.1 Time Stamps
```json
"time_stamps": [2.5, 7.5, 12.5, ...]
```
- **Type**: Array of floats
- **Meaning**: Time points (in seconds) for each segment center
- **Usage**: Aligns feature values with specific times in the track

### 4.2 Segment Features

For each segment, the following features are extracted:

| Feature | Shape | Meaning |
|---------|-------|---------|
| `danceability` | Array[N] | How danceable each segment is |
| `energy` | Array[N] | Energy level of each segment |
| `valence` | Array[N] | Musical positiveness of each segment |
| `tempo` | Array[N] | Tempo (BPM) of each segment |
| `spectral_centroid` | Array[N] | Brightness of each segment |
| `spectral_rolloff` | Array[N] | Frequency distribution of each segment |
| `spectral_bandwidth` | Array[N] | Spectral width of each segment |
| `zero_crossing_rate` | Array[N] | Noisiness of each segment |
| `mfcc_mean` | Matrix[N×13] | Average MFCC coefficients for each segment |
| `chroma_mean` | Matrix[N×12] | Average chroma features for each segment |

Where N is the number of segments.

### 4.3 Genre Predictions Over Time

```json
"genre_predictions": [[prob1, prob2, ...], [prob1, prob2, ...], ...],
"genre_labels": ["Genre 1", "Genre 2", ...]
```

| Feature | Shape | Meaning |
|---------|-------|---------|
| `genre_predictions` | Matrix[N×M] | Genre classification probabilities for each time segment. N = number of segments, M = number of possible genres |
| `genre_labels` | Array[M] | Genre names corresponding to prediction columns |

Each row in `genre_predictions` contains the probability distribution across all possible genres for that time segment.

---

## 5. Original Analysis

```json
"original_analysis": null
```

This field would contain results from the basic allin1 analysis (beats, downbeats, segments, etc.) if provided during comprehensive analysis. It's the standard analysis output from the main allin1 library.

---

## Value Ranges and Interpretation

### Normalized Features (0.0-1.0)
- `danceability`, `valence`, `acousticness`, `instrumentalness`
  - 0.0: Low presence of the characteristic
  - 0.5: Moderate presence
  - 1.0: High presence

### Unbounded Features
- `energy`, `loudness`: Can exceed 1.0, represent physical measurements
- `tempo`: Typically 60-200 BPM for most music
- `spectral_*`: Frequency-related values in Hz

### Arrays and Matrices
- Arrays represent values over time or across categories
- Matrices represent 2D data (e.g., features over time)

---

## Practical Applications

1. **Music Recommendation**: Use features to find similar tracks
2. **Music Analysis**: Understand structure and characteristics
3. **DJ Mixing**: Match tracks by key, tempo, and energy
4. **Music Classification**: Automatically categorize music
5. **Trend Analysis**: Track how features change over time in a track

---

## Example Interpretation

For the analyzed track "Amazing":
- **High danceability (1.08)**: Very suitable for dancing
- **High energy (644504)**: Very energetic track
- **High valence (0.88)**: Very positive, happy mood
- **Key: C Major**: Bright, positive key
- **Tempo: 140 BPM**: Upbeat tempo
- **Genre: Hard Rock/Arena Rock**: Consistent with high energy and danceability
- **High instrumentalness (0.97)**: Likely an instrumental track

This combination suggests an upbeat, energetic instrumental rock track with positive mood and strong rhythmic qualities.