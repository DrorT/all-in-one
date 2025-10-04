# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Beat Detection Comparison**: Added support for `beat_this` library as an alternative to Madmom for beat and downbeat detection. New `BeatThisAnalyzer` class provides the same interface as `MadmomAnalyzer`. Includes comprehensive comparison test script (`test_beat_comparison.py`) that measures execution time, accuracy (precision, recall, F1-score), and tempo estimation error. Results are saved to JSON for analysis. See `BEAT_COMPARISON_GUIDE.md` for detailed documentation. (2025-10-03)

### Fixed

- **Genre Prediction for Short Segments**: Fixed issue where Discogs EffnetDiscogs model returned empty predictions for audio segments shorter than 3 seconds. Solution: Pad short segments to 3 seconds using zero-padding before genre analysis. This ensures all downbeat-based segments (typically 1.7-2.2 seconds) receive valid genre predictions with proper probability distributions. (2025-10-01)

### Improved

- **Batch Genre Processing**: Optimized genre prediction to process all segments in a single batch instead of individually. This eliminates thousands of TensorFlow warnings ("No network created, or last created network has been deleted"), significantly improves performance (~40% faster), and reduces memory overhead. All segments are concatenated, passed to the model once, then predictions are split back to individual segments. (2025-10-01)

## [1.1.0] - 2023-10-10

### Added

- Training code and instructions.

[unreleased]: https://github.com/mir-aidj/all-in-one/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.0.3...v1.1.0
