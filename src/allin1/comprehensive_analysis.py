"""
Comprehensive Audio Analysis Module

This module provides advanced audio analysis capabilities using multiple libraries:
- Essentia for musical feature extraction
- Discogs for genre information
- Time-based feature analysis with heatmap visualization
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Audio processing
import librosa
import essentia
import essentia.standard as es
from essentia import Pool

# Madmom for beat and downbeat tracking
try:
    import madmom
    from madmom.io.audio import load_audio_file
    from madmom.features.beats import RNNBeatProcessor, BeatTrackingProcessor
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    MADMOM_AVAILABLE = True
except ImportError:
    MADMOM_AVAILABLE = False
    print("Madmom not available. Install with: pip install madmom")

# Clustering and segmentation
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Reduce Essentia logging noise (e.g., TriangularBands filter-bank info messages)
essentia.log.infoActive = False

# Mapping from textual music keys to integer indices (C=0,...,B=11)
KEY_TO_INT = {
    'C': 0,
    'C#': 1, 'Db': 1,
    'D': 2,
    'D#': 3, 'Eb': 3,
    'E': 4,
    'F': 5,
    'F#': 6, 'Gb': 6,
    'G': 7,
    'G#': 8, 'Ab': 8,
    'A': 9,
    'A#': 10, 'Bb': 10,
    'B': 11
}

# Try to import optional dependencies

try:
    import discogs_client
    DISCOGS_AVAILABLE = True
except ImportError:
    DISCOGS_AVAILABLE = False
    print("Discogs client not available. Install with: pip install discogs-client")

from .typings import PathLike, AnalysisResult
from .utils import mkpath


def _to_serializable(value: Any) -> Any:
    """Recursively convert numpy types and Paths into JSON-serializable objects."""
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass
class MadmomFeatures:
    """Data class for Madmom extracted features"""
    beats: np.ndarray
    downbeats: np.ndarray
    tempo: float
    beat_consistency: float
    beat_intervals: np.ndarray


@dataclass
class EssentiaFeatures:
    """Data class for Essentia extracted features"""
    danceability: float
    energy: float
    loudness: float
    valence: float
    acousticness: float
    instrumentalness: float
    key: int
    mode: int
    time_signature: int
    spectral_centroid: float
    spectral_rolloff: float
    spectral_bandwidth: float
    zero_crossing_rate: float
    mfcc: np.ndarray
    chroma: np.ndarray
    tonal: Dict[str, float]


@dataclass
class DiscogsInfo:
    """Data class for Discogs genre information"""
    genres: List[str]
    styles: List[str]
    year: Optional[int]
    title: Optional[str]
    artist: Optional[str]


@dataclass
class TimeBasedFeatures:
    """Data class for time-based features"""
    time_stamps: np.ndarray
    features: Dict[str, np.ndarray]
    feature_names: List[str]
    genre_predictions: Optional[np.ndarray] = None
    genre_labels: Optional[List[str]] = None
    segment_boundaries: Optional[List[Tuple[float, float]]] = None  # List of (start, end) times


@dataclass
class SegmentInfo:
    """Data class for track segment information"""
    start_time: float
    end_time: float
    segment_id: int
    cluster_id: int
    features: Dict[str, float]
    dominant_genre: Optional[str] = None
    genre_confidence: Optional[float] = None


@dataclass
class SegmentationResult:
    """Data class for track segmentation results"""
    segments: List[SegmentInfo]
    num_clusters: int
    cluster_labels: np.ndarray
    feature_names: List[str]
    clustering_method: str
    silhouette_score: Optional[float] = None


@dataclass
class SegmentGroup:
    """Data class for a group of consecutive similar segments"""
    group_id: int
    start_time: float
    end_time: float
    segment_ids: List[int]
    avg_features: Dict[str, float]
    dominant_genre: Optional[str] = None
    genre_confidence: Optional[float] = None


@dataclass
class GroupedSegmentationResult:
    """Data class for grouped segmentation results"""
    segment_groups: List[SegmentGroup]
    original_segments: List[SegmentInfo]
    num_groups: int
    feature_names: List[str]


def group_similar_segments(segments: List[SegmentInfo],
                          feature_names: List[str],
                          similarity_threshold: float = 0.15) -> GroupedSegmentationResult:
    """
    Group consecutive segments with similar features.
    
    Parameters
    ----------
    segments : List[SegmentInfo]
        List of segments to group
    feature_names : List[str]
        Names of features to use for similarity calculation
    similarity_threshold : float, optional
        Maximum normalized difference between features to consider segments similar.
        Lower values mean stricter similarity. Default: 0.15
        
    Returns
    -------
    GroupedSegmentationResult
        Grouped segmentation results
    """
    if len(segments) == 0:
        return GroupedSegmentationResult(
            segment_groups=[],
            original_segments=segments,
            num_groups=0,
            feature_names=feature_names
        )
    
    # Extract feature matrix for normalization
    feature_matrix = []
    for seg in segments:
        feature_vector = []
        for fname in feature_names:
            if fname in seg.features:
                feature_vector.append(seg.features[fname])
            else:
                feature_vector.append(0.0)
        feature_matrix.append(feature_vector)
    
    feature_matrix = np.array(feature_matrix)
    
    # Normalize features to 0-1 range for fair comparison
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(feature_matrix)
    
    # Group consecutive segments with similar features
    groups = []
    current_group_segments = [0]  # Start with first segment
    
    for i in range(1, len(segments)):
        # Calculate normalized Euclidean distance between consecutive segments
        prev_features = normalized_features[i-1]
        curr_features = normalized_features[i]
        
        # Calculate distance
        distance = np.linalg.norm(prev_features - curr_features) / np.sqrt(len(feature_names))
        
        if distance <= similarity_threshold:
            # Similar enough, add to current group
            current_group_segments.append(i)
        else:
            # Different enough, start new group
            groups.append(current_group_segments)
            current_group_segments = [i]
    
    # Add the last group
    groups.append(current_group_segments)
    
    # Create SegmentGroup objects
    segment_groups = []
    for group_id, group_indices in enumerate(groups):
        # Get start and end times
        start_time = segments[group_indices[0]].start_time
        end_time = segments[group_indices[-1]].end_time
        
        # Average features across the group
        avg_features = {}
        for fname in feature_names:
            values = [segments[idx].features.get(fname, 0.0) for idx in group_indices]
            avg_features[fname] = float(np.mean(values))
        
        # Determine dominant genre (most common across the group)
        genres = [segments[idx].dominant_genre for idx in group_indices if segments[idx].dominant_genre]
        if genres:
            # Get most common genre
            from collections import Counter
            genre_counter = Counter(genres)
            dominant_genre = genre_counter.most_common(1)[0][0]
            
            # Average confidence for the dominant genre
            confidences = [segments[idx].genre_confidence 
                          for idx in group_indices 
                          if segments[idx].dominant_genre == dominant_genre 
                          and segments[idx].genre_confidence is not None]
            genre_confidence = float(np.mean(confidences)) if confidences else None
        else:
            dominant_genre = None
            genre_confidence = None
        
        segment_groups.append(SegmentGroup(
            group_id=group_id,
            start_time=start_time,
            end_time=end_time,
            segment_ids=[segments[idx].segment_id for idx in group_indices],
            avg_features=avg_features,
            dominant_genre=dominant_genre,
            genre_confidence=genre_confidence
        ))
    
    return GroupedSegmentationResult(
        segment_groups=segment_groups,
        original_segments=segments,
        num_groups=len(segment_groups),
        feature_names=feature_names
    )


@dataclass
class ComprehensiveAnalysisResult:
    """Data class for comprehensive analysis results"""
    path: Path
    essentia_features: EssentiaFeatures
    madmom_features: Optional[MadmomFeatures]
    discogs_info: Optional[DiscogsInfo]
    time_based_features: TimeBasedFeatures
    original_analysis: Optional[AnalysisResult] = None
    segmentation_result: Optional[SegmentationResult] = None
    grouped_segmentation: Optional[GroupedSegmentationResult] = None


class EssentiaAnalyzer:
    """Class for extracting features using Essentia"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        # Initialize Essentia algorithms - using only confirmed available algorithms
        self.danceability = es.Danceability()
        self.loudness = es.Loudness()
        self.energy = es.Energy()
        self.key_extractor = es.KeyExtractor()
        self.zcr = es.ZeroCrossingRate()
        self.tonal = es.TonalExtractor()
        self._tonal_output_names = self.tonal.outputNames()
        
    def extract_features(self, audio_path: PathLike) -> EssentiaFeatures:
        """Extract comprehensive features using Essentia"""
        audio_path = mkpath(audio_path)
        
        try:
            # Load audio
            loader = es.MonoLoader(filename=str(audio_path), sampleRate=self.sample_rate)
            audio = loader()
            
            # Extract basic features
            danceability_output = self.danceability(audio)
            if isinstance(danceability_output, tuple):
                danceability_value = float(danceability_output[0])
            else:
                danceability_value = float(danceability_output)
            loudness_value = float(self.loudness(audio))
            energy_value = float(self.energy(audio))
            
            # Extract tonal features
            key_output = self.key_extractor(audio)
            if isinstance(key_output, tuple):
                key_name, key_scale, key_strength = key_output
            else:
                key_name, key_scale, key_strength = 0, 'major', 0.0

            key = KEY_TO_INT.get(str(key_name), 0)
            scale = 1 if str(key_scale).lower() == 'major' else 0
            key_strength = float(key_strength) if not isinstance(key_strength, (list, np.ndarray)) else float(np.mean(key_strength))

            tonal_raw = self.tonal(audio)
            tonal_features = self._format_tonal_output(tonal_raw)
            
            # Extract spectral features
            zcr_values = self.zcr(audio)
            if isinstance(zcr_values, np.ndarray):
                zcr_mean = float(np.mean(zcr_values))
            else:
                zcr_mean = float(zcr_values)
            
            # Extract MFCC (Essentia returns bands and coefficients)
            _, mfcc_coeffs = es.MFCC()(audio)
            
            # Calculate aggregate values
            
            # Use librosa for spectral and chroma features that might not be available in Essentia
            y = np.asarray(audio, dtype=np.float32)
            sr = self.sample_rate
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            chroma_features = librosa.feature.chroma_stft(y=y, sr=sr)
            
            centroid_mean = float(np.mean(spectral_centroids))
            rolloff_mean = float(np.mean(spectral_rolloff))
            bandwidth_mean = float(np.mean(spectral_bandwidth))
            
            # Estimate some features similar to Spotify's API
            valence = self._estimate_valence(tonal_features, key_strength, scale)
            acousticness = self._estimate_acousticness(spectral_centroid=centroid_mean,
                                                      spectral_rolloff=rolloff_mean,
                                                      zero_crossing_rate=zcr_mean)
            instrumentalness = self._estimate_instrumentalness(audio)
            
            # Extract time signature (simplified) - using default value
            time_signature = 4  # Default to 4/4
            
            return EssentiaFeatures(
                danceability=danceability_value,
                energy=energy_value,
                loudness=loudness_value,
                valence=valence,
                acousticness=acousticness,
                instrumentalness=instrumentalness,
                key=key,
                mode=scale,  # 0 for minor, 1 for major
                time_signature=time_signature,
                spectral_centroid=centroid_mean,
                spectral_rolloff=rolloff_mean,
                spectral_bandwidth=bandwidth_mean,
                zero_crossing_rate=zcr_mean,
                mfcc=mfcc_coeffs,
                chroma=chroma_features,
                tonal=tonal_features
            )
        except Exception as e:
            print(f"Error in Essentia feature extraction: {e}")
            # Return a default set of features if extraction fails
            return EssentiaFeatures(
                danceability=0.5,
                energy=0.5,
                loudness=0.5,
                valence=0.5,
                acousticness=0.5,
                instrumentalness=0.5,
                key=0,
                mode=0,
                time_signature=4,
                spectral_centroid=0.5,
                spectral_rolloff=0.5,
                spectral_bandwidth=0.5,
                zero_crossing_rate=0.5,
                mfcc=np.zeros((13, 100)),  # Default MFCC shape
                chroma=np.zeros((12, 100)),  # Default chroma shape
                tonal={}
            )
    
    def _format_tonal_output(self, tonal_raw: Union[Dict, Tuple, List]) -> Dict[str, Any]:
        """Convert TonalExtractor output into a dictionary keyed by feature."""
        if isinstance(tonal_raw, dict):
            return tonal_raw
        if isinstance(tonal_raw, tuple):
            tonal_features: Dict[str, Any] = {}
            for idx, name in enumerate(self._tonal_output_names):
                tonal_features[name] = tonal_raw[idx] if idx < len(tonal_raw) else None
            return tonal_features
        if isinstance(tonal_raw, list):
            return {name: tonal_raw[idx] if idx < len(tonal_raw) else None for idx, name in enumerate(self._tonal_output_names)}
        return {}
    
    def _estimate_valence(self, tonal_features: Dict[str, Any], key_strength: float, default_mode: int) -> float:
        """Estimate valence (musical positiveness) from tonal and key features."""
        scale_hint = tonal_features.get('chords_scale') or tonal_features.get('key_scale')
        scale_value = default_mode

        if isinstance(scale_hint, (list, tuple)) and scale_hint:
            scale_hint = scale_hint[0]
        elif isinstance(scale_hint, np.ndarray) and scale_hint.size > 0:
            scale_hint = scale_hint.flat[0]

        if isinstance(scale_hint, str):
            if 'major' in scale_hint.lower():
                scale_value = 1
            elif 'minor' in scale_hint.lower():
                scale_value = 0

        polarity = 1 if scale_value == 1 else -1
        strength = float(np.clip(key_strength, 0, 1))

        chords_strength = tonal_features.get('chords_strength', 0.5)
        if isinstance(chords_strength, (list, tuple)) and chords_strength:
            chords_strength = chords_strength[0]
        elif isinstance(chords_strength, np.ndarray) and chords_strength.size > 0:
            chords_strength = float(np.mean(chords_strength))
        try:
            chords_strength = float(chords_strength)
        except (TypeError, ValueError):
            chords_strength = 0.5
        chords_strength = float(np.clip(chords_strength, 0, 1))

        valence = 0.5 + 0.4 * polarity * strength + 0.1 * (chords_strength - 0.5)
        return float(np.clip(valence, 0, 1))
    
    def _estimate_acousticness(self, spectral_centroid: float, 
                             spectral_rolloff: float, 
                             zero_crossing_rate: float) -> float:
        """Estimate acousticness from spectral features"""
        # Simple heuristic based on spectral characteristics
        # Lower centroid and rolloff, higher ZCR tend to indicate acoustic
        acoustic_score = (1.0 - spectral_centroid/10000) * 0.5 + \
                        (1.0 - spectral_rolloff/10000) * 0.3 + \
                        zero_crossing_rate * 0.2
        return np.clip(acoustic_score, 0, 1)
    
    def _estimate_instrumentalness(self, audio: np.ndarray) -> float:
        """Estimate instrumentalness from audio characteristics"""
        # Simple heuristic based on spectral variance
        if audio.size == 0:
            return 0.5
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        if magnitude.size == 0:
            return 0.5
        spectral_variance = np.var(magnitude, axis=0)
        if spectral_variance.size == 0:
            return 0.5
        # Lower variance might indicate instrumental music
        return float(np.clip(1.0 - np.mean(spectral_variance) / 1000, 0, 1))
    


class MadmomAnalyzer:
    """Class for extracting beats and downbeats using Madmom"""
    
    def __init__(self):
        if not MADMOM_AVAILABLE:
            raise ImportError("Madmom is not available. Install with: pip install madmom")
    
    def extract_beats_and_downbeats(self, audio_path: PathLike) -> Optional[MadmomFeatures]:
        """Extract beats and downbeats using Madmom"""
        audio_path = mkpath(audio_path)
        
        try:
            # Load the audio file using madmom
            signal, sample_rate = load_audio_file(str(audio_path))
            
            # Beat tracking with madmom
            beat_processor = RNNBeatProcessor()
            beat_activations = beat_processor(signal)
            
            # Track beats
            beat_tracker = BeatTrackingProcessor(fps=100)
            beats = beat_tracker(beat_activations)
            
            # Convert beat frames to time (madmom already returns times in seconds)
            beat_times = beats
            
            # Calculate tempo from beat times using improved methods
            if len(beat_times) > 1:
                # Calculate beat intervals
                beat_intervals = np.diff(beat_times)
                
                # Method 1: Primary - Use median for robustness against outliers
                median_interval = np.median(beat_intervals)
                tempo_from_median = 60.0 / median_interval if median_interval > 0 else 0.0
                
                # Method 2: Histogram-based estimation for most common interval
                # Create histogram of intervals (focus on reasonable tempo range: 60-200 BPM)
                valid_intervals = beat_intervals[(beat_intervals > 0.3) & (beat_intervals < 1.0)]  # 60-200 BPM range
                if len(valid_intervals) > 0:
                    hist_counts, hist_bins = np.histogram(valid_intervals, bins=50)
                    most_common_bin_idx = np.argmax(hist_counts)
                    most_common_interval = (hist_bins[most_common_bin_idx] + hist_bins[most_common_bin_idx + 1]) / 2
                    tempo_from_histogram = 60.0 / most_common_interval if most_common_interval > 0 else 0.0
                else:
                    tempo_from_histogram = tempo_from_median
                
                # Method 3: Weighted average (weight by inverse of local variance)
                # Calculate local variance for each interval (compare with neighbors)
                local_variances = []
                for i in range(len(beat_intervals)):
                    # Get neighboring intervals (within ±2 positions)
                    start_idx = max(0, i - 2)
                    end_idx = min(len(beat_intervals), i + 3)
                    local_intervals = beat_intervals[start_idx:end_idx]
                    if len(local_intervals) > 1:
                        local_variances.append(np.var(local_intervals))
                    else:
                        local_variances.append(0.0)
                
                # Use inverse variance as weights
                weights = 1.0 / (np.array(local_variances) + 1e-6)  # Add small epsilon to avoid division by zero
                weighted_interval = np.average(beat_intervals, weights=weights)
                tempo_weighted = 60.0 / weighted_interval if weighted_interval > 0 else 0.0
                
                # Combine methods: prefer histogram if it seems reliable, otherwise use median
                # Check histogram reliability by looking at the peak-to-average ratio
                if len(valid_intervals) > 0:
                    peak_to_avg = hist_counts[most_common_bin_idx] / (np.mean(hist_counts) + 1e-6)
                    if peak_to_avg > 2.0:  # Strong peak in histogram
                        tempo_value = tempo_from_histogram
                    else:
                        # Use weighted average of median and weighted tempo
                        tempo_value = 0.6 * tempo_from_median + 0.4 * tempo_weighted
                else:
                    tempo_value = tempo_from_median
                
                # Calculate beat consistency (coefficient of variation)
                if median_interval > 0:
                    beat_consistency = 1.0 - (np.std(beat_intervals) / median_interval)
                    beat_consistency = max(0.0, min(1.0, beat_consistency))  # Clamp between 0 and 1
                else:
                    beat_consistency = 0.0
                
                # Additional sanity check: if tempo seems unreasonable, fall back to median
                if tempo_value < 60 or tempo_value > 200:
                    tempo_value = tempo_from_median
            else:
                beat_intervals = np.array([])
                tempo_value = 0.0
                beat_consistency = 0
            
            # Downbeat tracking with madmom
            try:
                downbeat_processor = RNNDownBeatProcessor()
                downbeat_activations = downbeat_processor(signal)
                
                # Track downbeats
                downbeat_tracker = DBNDownBeatTrackingProcessor(fps=100, beats_per_bar=[4, 3])  # Common time signatures
                downbeats = downbeat_tracker(downbeat_activations)
                
                # Extract downbeat times (first column is time, second is beat number)
                downbeat_times = downbeats[:, 0] if len(downbeats) > 0 else np.array([])
                
                # Extract beat numbers (1-based, where 1 is downbeat)
                beat_numbers = downbeats[:, 1] if len(downbeats) > 0 else np.array([])
                
                # Find just the downbeats (where beat number is 1)
                downbeat_indices = np.where(beat_numbers == 1)[0]
                downbeat_times_only = downbeat_times[downbeat_indices] if len(downbeat_indices) > 0 else np.array([])
                
            except Exception as e:
                print(f"Warning: Downbeat detection failed: {e}")
                downbeat_times_only = np.array([])
            
            return MadmomFeatures(
                beats=beat_times,
                downbeats=downbeat_times_only,
                tempo=tempo_value,
                beat_consistency=beat_consistency,
                beat_intervals=beat_intervals
            )
            
        except Exception as e:
            print(f"Error in Madmom beat and downbeat extraction: {e}")
            return None


class DiscogsAnalyzer:
    """Class for getting genre information using Essentia's TensorflowPredictEffnetDiscogs model"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the Discogs genre classification model
        
        Parameters
        ----------
        model_path : str, optional
            Path to the Discogs model file. If not provided, will try to find it
            in the Essentia installation or use a default configuration.
        """
        try:
            # Try to load the genre labels from the JSON file
            labels_path = None
            if model_path and os.path.exists(model_path):
                # Try to find the JSON file in the same directory as the model
                model_dir = os.path.dirname(model_path)
                json_path = os.path.join(model_dir, os.path.basename(model_path).replace('.pb', '.json'))
                if os.path.exists(json_path):
                    labels_path = json_path
            
            if not labels_path:
                # Try to find the JSON file in the autotagging directory
                autotagging_path = os.path.join(os.getcwd(), 'autotagging', 'discogs-effnet-bs64-1.json')
                if os.path.exists(autotagging_path):
                    labels_path = autotagging_path
            
            if labels_path and os.path.exists(labels_path):
                # Load the labels from the JSON file
                with open(labels_path, 'r') as f:
                    labels_data = json.load(f)
                
                # Extract the genre labels from the "classes" field
                if "classes" in labels_data and isinstance(labels_data["classes"], list):
                    self.genre_labels = labels_data["classes"]
                    print(f"Loaded {len(self.genre_labels)} genre labels from {labels_path}")
                else:
                    # Fallback to default labels if classes field not found
                    self.genre_labels = [
                        'Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz',
                        'Metal', 'Pop', 'Reggae', 'Rock', 'Electronic', 'Folk'
                    ]
                    print(f"Classes field not found in {labels_path}, using default labels")
            else:
                # Use default labels if JSON file not found
                self.genre_labels = [
                    'Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz',
                    'Metal', 'Pop', 'Reggae', 'Rock', 'Electronic', 'Folk'
                ]
                
                # If we have more predictions than labels, extend with generic labels
                if len(self.genre_labels) < 400:
                    for i in range(len(self.genre_labels), 400):
                        self.genre_labels.append(f'Genre_{i}')
                
                print(f"Using default genre labels ({len(self.genre_labels)} total)")
            
            if model_path and os.path.exists(model_path):
                # Use the provided model path
                self.discogs_classifier = es.TensorflowPredictEffnetDiscogs(
                    graphFilename=model_path,
                    input="serving_default_melspectrogram",
                    output="PartitionedCall"
                )
                print(f"Discogs genre classifier (EffnetDiscogs) initialized with model: {model_path}")
            else:
                # Try to find the model in Essentia's installation
                import essentia
                
                essentia_dir = os.path.dirname(essentia.__file__)
                possible_paths = [
                    os.path.join(essentia_dir, 'models', 'effnet-discogs', 'effnet-discogs-1.pb'),
                    os.path.join(essentia_dir, 'models', 'discogs', 'effnet-discogs-1.pb'),
                ]
                
                found_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        found_path = path
                        break
                
                if found_path:
                    # Initialize the Discogs model with the found path
                    self.discogs_classifier = es.TensorflowPredictEffnetDiscogs(
                        graphFilename=found_path,
                        input="serving_default_melspectrogram",
                        output="PartitionedCall"
                    )
                    print(f"Discogs genre classifier (EffnetDiscogs) initialized with model: {found_path}")
                else:
                    # Model not found, provide instructions
                    print("Discogs model not found in Essentia installation.")
                    print("To use Discogs genre classification, you need to download the model:")
                    print("1. Go to https://essentia.upf.edu/models/")
                    print("2. Navigate to the autotagging/ folder")
                    print("3. Download the 'discogs-effnet-bs64-1.pb' model")
                    print("4. Save the model file (.pb) to a location of your choice")
                    print("5. Initialize the analyzer with the model path:")
                    print("   analyzer = DiscogsAnalyzer(model_path='/path/to/model.pb')")
                    
                    # Set classifier to None to indicate it's not available
                    self.discogs_classifier = None
            
        except Exception as e:
            print(f"Failed to initialize Discogs genre classifier: {e}")
            self.discogs_classifier = None
            self.genre_labels = []
        
    def analyze_genre(self, audio_path: PathLike) -> Optional[DiscogsInfo]:
        """Analyze audio file using Discogs genre classification to get genre information"""
        audio_path = mkpath(audio_path)
        
        if not self.discogs_classifier:
            print("Discogs genre classifier not available")
            return None
            
        try:
            # Load audio at 16kHz (required by the model)
            loader = es.MonoLoader(filename=str(audio_path), sampleRate=16000)
            audio = loader()
            
            # Get predictions from the model
            # The model handles the frame generation and feature extraction internally
            predictions = self.discogs_classifier(audio)
            
            # The model returns predictions for each frame, so we average them
            if predictions is not None:
                # Convert to numpy array for easier manipulation
                pred_array = np.array(predictions)
                
                # Check if we have valid predictions
                if pred_array.size > 0:
                    # Average predictions across all frames
                    if pred_array.ndim > 1:
                        avg_predictions = np.mean(pred_array, axis=0)
                    else:
                        avg_predictions = pred_array
                    
                    # Apply softmax to convert logits to probabilities
                    exp_pred = np.exp(avg_predictions - np.max(avg_predictions))
                    probabilities = exp_pred / np.sum(exp_pred)
                    
                    # Debug information
                    print(f"Prediction shape: {probabilities.shape}")
                    print(f"Number of genre labels: {len(self.genre_labels)}")
                    
                    # Get top genres by probability
                    sorted_indices = np.argsort(probabilities)[::-1]
                    
                    # Make sure we don't exceed the number of labels
                    num_genres = min(len(sorted_indices), len(self.genre_labels))
                    top_genres = [self.genre_labels[i] for i in sorted_indices[:num_genres]]
                    
                    return DiscogsInfo(
                        genres=top_genres[:5],  # Limit to top 5
                        styles=[],  # Discogs model doesn't provide styles
                        year=None,  # Discogs model doesn't provide year
                        title=audio_path.stem,
                        artist=None  # Discogs model doesn't provide artist
                    )
            
            return None
            
        except Exception as e:
            print(f"Error analyzing genre with Discogs classifier: {e}")
            return None
    
    def analyze_genre_over_time(self, audio_path: PathLike, segment_duration: float = 5.0) -> Tuple[Optional[DiscogsInfo], np.ndarray, np.ndarray]:
        """
        Analyze audio file using Discogs genre classification over time segments
        
        Parameters
        ----------
        audio_path : PathLike
            Path to the audio file
        segment_duration : float, optional
            Duration in seconds for each segment (default: 5.0)
            
        Returns
        -------
        Tuple[Optional[DiscogsInfo], np.ndarray, np.ndarray]
            Overall genre info, time stamps, and genre predictions for each segment
        """
        audio_path = mkpath(audio_path)
        
        if not self.discogs_classifier:
            print("Discogs genre classifier not available")
            return None, np.array([]), np.array([])
            
        try:
            # Load audio at 16kHz (required by the model)
            loader = es.MonoLoader(filename=str(audio_path), sampleRate=16000)
            audio = loader()
            
            # Calculate segment parameters
            segment_samples = int(segment_duration * 16000)  # 16kHz sample rate
            total_samples = len(audio)
            num_segments = int(np.ceil(total_samples / segment_samples))
            
            # Store predictions for each segment
            segment_predictions = []
            time_stamps = []
            
            # Process each segment
            for i in range(num_segments):
                start_sample = i * segment_samples
                end_sample = min((i + 1) * segment_samples, total_samples)
                segment_audio = audio[start_sample:end_sample]
                
                # Skip very short segments
                if len(segment_audio) < segment_samples * 0.5:
                    continue
                
                # Pad to minimum 3 seconds if needed (model requires 3-second patches)
                min_samples = int(3.0 * 16000)  # 3 seconds at 16kHz
                if len(segment_audio) < min_samples:
                    # Pad with zeros at the end
                    segment_audio = np.pad(segment_audio, (0, min_samples - len(segment_audio)), mode='constant')
                
                # Get predictions for this segment
                predictions = self.discogs_classifier(segment_audio)
                
                if predictions is not None:
                    # Convert to numpy array
                    pred_array = np.array(predictions)
                    
                    # Average predictions across all frames in this segment
                    if pred_array.ndim > 1:
                        avg_predictions = np.mean(pred_array, axis=0)
                    else:
                        avg_predictions = pred_array
                    
                    segment_predictions.append(avg_predictions)
                    # Store time stamp (middle of segment)
                    time_stamps.append((start_sample + end_sample) / 2 / 16000)  # Convert to seconds
            
            # Convert to numpy arrays
            time_stamps = np.array(time_stamps)
            segment_predictions = np.array(segment_predictions)
            
            # Calculate overall average predictions
            if segment_predictions.size > 0:
                overall_predictions = np.mean(segment_predictions, axis=0)
                
                # Get top genres by probability
                sorted_indices = np.argsort(overall_predictions)[::-1]
                num_genres = min(len(sorted_indices), len(self.genre_labels))
                top_genres = [self.genre_labels[i] for i in sorted_indices[:num_genres]]
                
                overall_info = DiscogsInfo(
                    genres=top_genres[:5],  # Limit to top 5
                    styles=[],  # Discogs model doesn't provide styles
                    year=None,  # Discogs model doesn't provide year
                    title=audio_path.stem,
                    artist=None  # Discogs model doesn't provide artist
                )
                
                return overall_info, time_stamps, segment_predictions
            
            return None, np.array([]), np.array([])
            
        except Exception as e:
            print(f"Error analyzing genre over time with Discogs classifier: {e}")
            return None, np.array([]), np.array([])
    
    # Keep the old API for backward compatibility but mark as deprecated
    def search_by_filename(self, filename: str) -> Optional[DiscogsInfo]:
        """Deprecated: Use analyze_genre instead"""
        print("Warning: search_by_filename is deprecated. Use analyze_genre with an audio file path.")
        return None


def calculate_downbeat_segments(downbeats: np.ndarray, audio_duration: float, 
                               min_edge_duration: float = 0.3) -> List[Tuple[float, float]]:
    """
    Calculate segment boundaries based on downbeats.
    
    Parameters
    ----------
    downbeats : np.ndarray
        Array of downbeat times in seconds
    audio_duration : float
        Total duration of the audio in seconds
    min_edge_duration : float, optional
        Minimum duration for edge segments (first/last). If shorter, 
        merge with adjacent segment. Default: 0.3 seconds
        
    Returns
    -------
    List[Tuple[float, float]]
        List of (start_time, end_time) tuples for each segment
    """
    if len(downbeats) == 0:
        # No downbeats, return single segment for entire track
        return [(0.0, audio_duration)]
    
    segments = []
    downbeats = np.sort(downbeats)  # Ensure sorted
    
    # Determine first segment
    first_downbeat = downbeats[0]
    if first_downbeat < min_edge_duration:
        # First downbeat is very early, create segment from 0 to second downbeat (or end if only one)
        first_segment_start = 0.0
        first_segment_end = downbeats[1] if len(downbeats) > 1 else audio_duration
        segments.append((first_segment_start, first_segment_end))
        segment_start_idx = 1  # Start middle segments from second downbeat
    else:
        # First segment includes audio before first downbeat
        first_segment_start = 0.0
        first_segment_end = first_downbeat
        segments.append((first_segment_start, first_segment_end))
        segment_start_idx = 0  # Start middle segments from first downbeat
    
    # Middle segments: between consecutive downbeats
    for i in range(segment_start_idx, len(downbeats) - 1):
        start_time = downbeats[i]
        end_time = downbeats[i + 1]
        segments.append((start_time, end_time))
    
    # Determine last segment
    last_downbeat = downbeats[-1]
    time_after_last = audio_duration - last_downbeat
    
    if time_after_last < min_edge_duration:
        # Last downbeat is very close to end, extend previous segment
        if len(segments) > 0:
            # Extend the last segment to the end
            segments[-1] = (segments[-1][0], audio_duration)
    else:
        # Last segment is everything after last downbeat
        segments.append((last_downbeat, audio_duration))
    
    return segments


class TimeBasedAnalyzer:
    """Class for extracting time-based features"""
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
    def extract_time_features(self, audio_path: PathLike, 
                            segment_duration: float = 5.0,
                            madmom_features: Optional[MadmomFeatures] = None) -> TimeBasedFeatures:
        """Extract time-based features by segmenting the audio"""
        audio_path = mkpath(audio_path)
        
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        audio_duration = len(audio) / sr
        
        # Determine segments based on downbeats or fixed duration
        if madmom_features is not None and len(madmom_features.downbeats) > 0:
            # Use downbeat-based segmentation
            segment_boundaries = calculate_downbeat_segments(
                madmom_features.downbeats, audio_duration, min_edge_duration=0.3
            )
            print(f"Using {len(segment_boundaries)} downbeat-based segments")
        else:
            # Fall back to fixed-duration segmentation
            segment_samples = int(segment_duration * sr)
            total_samples = len(audio)
            num_segments = int(np.ceil(total_samples / segment_samples))
            
            segment_boundaries = []
            for i in range(num_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, audio_duration)
                segment_boundaries.append((start_time, end_time))
            print(f"Using {len(segment_boundaries)} fixed-duration segments")
        
        # Initialize feature storage
        features = {
            'danceability': [],
            'energy': [],
            'valence': [],
            'spectral_centroid': [],
            'spectral_rolloff': [],
            'spectral_bandwidth': [],
            'zero_crossing_rate': [],
            'mfcc_mean': [],
            'chroma_mean': []
        }
        
        # Time stamps for each segment (middle of segment)
        time_stamps = []
        
        # Initialize Essentia analyzer for segment analysis
        essentia_analyzer = EssentiaAnalyzer(self.sample_rate)
        
        # Process each segment
        for seg_idx, (start_time, end_time) in enumerate(segment_boundaries):
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment_audio = audio[start_sample:end_sample]
            
            # Skip very short segments (less than 0.1 seconds)
            if len(segment_audio) < int(0.1 * sr):
                continue
                
            # Save segment to temporary file
            temp_path = f"temp_segment_{seg_idx}.wav"
            import soundfile as sf
            sf.write(temp_path, segment_audio, sr)
            
            try:
                # Extract features for this segment
                segment_features = essentia_analyzer.extract_features(temp_path)
                
                # Store features
                features['danceability'].append(segment_features.danceability)
                features['energy'].append(segment_features.energy)
                features['valence'].append(segment_features.valence)
                features['spectral_centroid'].append(segment_features.spectral_centroid)
                features['spectral_rolloff'].append(segment_features.spectral_rolloff)
                features['spectral_bandwidth'].append(segment_features.spectral_bandwidth)
                features['zero_crossing_rate'].append(segment_features.zero_crossing_rate)

                mfcc_matrix = np.asarray(segment_features.mfcc)
                if mfcc_matrix.ndim == 1:
                    mfcc_mean = mfcc_matrix
                else:
                    mfcc_axis = 1 if mfcc_matrix.shape[1] >= mfcc_matrix.shape[0] else 0
                    mfcc_mean = np.mean(mfcc_matrix, axis=mfcc_axis)
                features['mfcc_mean'].append(np.asarray(mfcc_mean, dtype=float).flatten())

                chroma_matrix = np.asarray(segment_features.chroma)
                if chroma_matrix.ndim == 1:
                    chroma_mean = chroma_matrix
                else:
                    chroma_axis = 1 if chroma_matrix.shape[1] >= chroma_matrix.shape[0] else 0
                    chroma_mean = np.mean(chroma_matrix, axis=chroma_axis)
                features['chroma_mean'].append(np.asarray(chroma_mean, dtype=float).flatten())
                
                # Store time stamp (middle of segment)
                time_stamps.append((start_time + end_time) / 2)
                
            except Exception as e:
                print(f"Error processing segment {seg_idx} ({start_time:.2f}-{end_time:.2f}): {e}")
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        
        # Convert scalar feature lists to numpy arrays
        for key, values in list(features.items()):
            if key in ('mfcc_mean', 'chroma_mean'):
                continue
            features[key] = np.array(values)
            
        time_stamps = np.array(time_stamps)
        
        # Handle MFCC and chroma (which are 2D arrays)
        if features['mfcc_mean']:
            features['mfcc_mean'] = np.vstack(features['mfcc_mean'])
        else:
            features['mfcc_mean'] = np.empty((0, 0))
        if features['chroma_mean']:
            features['chroma_mean'] = np.vstack(features['chroma_mean'])
        else:
            features['chroma_mean'] = np.empty((0, 0))
            
        feature_names = list(features.keys())
        
        return TimeBasedFeatures(
            time_stamps=time_stamps,
            features=features,
            feature_names=feature_names,
            genre_predictions=None,
            genre_labels=None,
            segment_boundaries=segment_boundaries
        )
    
    def extract_time_features_with_genre(self, audio_path: PathLike,
                                       segment_duration: float = 5.0,
                                       discogs_analyzer=None,
                                       madmom_features: Optional[MadmomFeatures] = None) -> TimeBasedFeatures:
        """Extract time-based features by segmenting the audio, including genre analysis"""
        audio_path = mkpath(audio_path)
        
        # First extract standard time-based features
        time_features = self.extract_time_features(audio_path, segment_duration, madmom_features)
        
        # If Discogs analyzer is provided, add genre analysis for each segment
        if discogs_analyzer and discogs_analyzer.discogs_classifier:
            try:
                # Load audio to get duration
                audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
                audio_duration = len(audio) / sr
                
                # Calculate segment boundaries (same as in extract_time_features)
                if madmom_features is not None and len(madmom_features.downbeats) > 0:
                    segment_boundaries = calculate_downbeat_segments(
                        madmom_features.downbeats, audio_duration, min_edge_duration=0.3
                    )
                else:
                    segment_samples = int(segment_duration * sr)
                    total_samples = len(audio)
                    num_segments = int(np.ceil(total_samples / segment_samples))
                    segment_boundaries = []
                    for i in range(num_segments):
                        start_time = i * segment_duration
                        end_time = min((i + 1) * segment_duration, audio_duration)
                        segment_boundaries.append((start_time, end_time))
                
                # Analyze genre for each segment
                genre_predictions_list = []
                for start_time, end_time in segment_boundaries:
                    start_sample = int(start_time * sr)
                    end_sample = int(end_time * sr)
                    
                    # Skip very short segments
                    if end_sample - start_sample < int(0.1 * sr):
                        continue
                    
                    segment_audio = audio[start_sample:end_sample]
                    
                    # Resample to 16kHz for Discogs model
                    segment_audio_16k = librosa.resample(segment_audio, orig_sr=sr, target_sr=16000)
                    
                    # Pad to minimum 3 seconds if needed (model requires 3-second patches)
                    min_samples = int(3.0 * 16000)  # 3 seconds at 16kHz
                    if len(segment_audio_16k) < min_samples:
                        # Pad with zeros at the end
                        segment_audio_16k = np.pad(segment_audio_16k, (0, min_samples - len(segment_audio_16k)), mode='constant')
                    
                    try:
                        # Get genre predictions directly from the classifier
                        predictions = discogs_analyzer.discogs_classifier(segment_audio_16k)
                        
                        if predictions is not None:
                            # Convert to numpy array
                            pred_array = np.array(predictions)
                            
                            # Average predictions across frames if needed
                            if pred_array.ndim > 1:
                                avg_predictions = np.mean(pred_array, axis=0)
                            else:
                                avg_predictions = pred_array
                            
                            # Apply softmax to convert logits to probabilities
                            # Softmax: exp(x) / sum(exp(x))
                            exp_pred = np.exp(avg_predictions - np.max(avg_predictions))  # Subtract max for numerical stability
                            probabilities = exp_pred / np.sum(exp_pred)
                            
                            # Ensure we have the right number of predictions
                            if len(probabilities) == len(discogs_analyzer.genre_labels):
                                genre_predictions_list.append(probabilities)
                            else:
                                print(f"Warning: Prediction size mismatch for segment at {start_time:.2f}s")
                                genre_predictions_list.append(np.zeros(len(discogs_analyzer.genre_labels)))
                        else:
                            # No predictions, use zeros
                            genre_predictions_list.append(np.zeros(len(discogs_analyzer.genre_labels)))
                    except Exception as e:
                        print(f"Error analyzing genre for segment at {start_time:.2f}s: {e}")
                        genre_predictions_list.append(np.zeros(len(discogs_analyzer.genre_labels)))
                
                if genre_predictions_list:
                    time_features.genre_predictions = np.array(genre_predictions_list)
                    time_features.genre_labels = discogs_analyzer.genre_labels
                    print(f"Added genre analysis for {len(genre_predictions_list)} segments")
                
                # Store segment boundaries if not already set
                if time_features.segment_boundaries is None:
                    time_features.segment_boundaries = segment_boundaries
                
            except Exception as e:
                print(f"Error adding genre analysis: {e}")
        
        return time_features


class HeatmapVisualizer:
    """Class for creating heatmap visualizations"""
    
    def __init__(self):
        self.colors = 'viridis'
        
    def create_heatmap(self, time_based_features: TimeBasedFeatures,
                      output_path: Optional[PathLike] = None) -> plt.Figure:
        """Create a heatmap visualization of time-based features"""
        # Prepare data for heatmap
        features = time_based_features.features
        time_stamps = time_based_features.time_stamps
        
        # Select features for heatmap (exclude 2D arrays for now)
        heatmap_data = {}
        for name in time_based_features.feature_names:
            if name not in ['mfcc_mean', 'chroma_mean'] and len(features[name]) > 0:
                heatmap_data[name] = features[name]
        
        # Add genre information if available
        if (time_based_features.genre_predictions is not None and
            time_based_features.genre_labels is not None and
            len(time_based_features.genre_predictions) > 0):
            
            # Get top 5 genres for each time segment
            genre_preds = time_based_features.genre_predictions
            genre_labels = time_based_features.genre_labels
            
            # For each time segment, get the top genre
            top_genres = []
            for i in range(genre_preds.shape[0]):
                sorted_indices = np.argsort(genre_preds[i])[::-1]
                top_genre = genre_labels[sorted_indices[0]]
                # Shorten the genre name for display
                if '---' in top_genre:
                    top_genre = top_genre.split('---')[1]
                top_genres.append(top_genre)
            
            # Convert genre names to numerical values for heatmap
            unique_genres = list(set(top_genres))
            genre_values = [unique_genres.index(genre) for genre in top_genres]
            
            # Add to heatmap data
            heatmap_data['genre'] = genre_values
        
        if not heatmap_data:
            print("No suitable features for heatmap")
            return None
            
        # Create DataFrame
        df = pd.DataFrame(heatmap_data, index=time_stamps)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df.T, ax=ax, cmap=self.colors, cbar_kws={'label': 'Feature Value'})
        
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Features')
        ax.set_title('Time-Based Audio Features Heatmap')
        
        # Format x-axis to show time in minutes:seconds
        time_labels = [f"{t/60:.1f}:{t%60:.0f}" for t in time_stamps[::max(1, len(time_stamps)//10)]]
        ax.set_xticks(np.arange(0, len(time_stamps), max(1, len(time_stamps)//10)))
        ax.set_xticklabels(time_labels, rotation=45)
        
        # Add genre labels if available
        if 'genre' in heatmap_data:
            # Create a second axis for genre labels
            ax2 = ax.twinx()
            ax2.set_ylabel('Genres')
            ax2.set_yticks(np.arange(len(unique_genres)) + 0.5)
            ax2.set_yticklabels(unique_genres)
            ax2.set_ylim(ax.get_ylim())
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {output_path}")
            
        return fig
    
    def create_genre_timeline(self, time_based_features: TimeBasedFeatures,
                             output_path: Optional[PathLike] = None) -> plt.Figure:
        """Create a timeline plot of genre predictions over time"""
        if (time_based_features.genre_predictions is None or
            time_based_features.genre_labels is None or
            len(time_based_features.genre_predictions) == 0):
            print("No genre predictions available for timeline")
            return None
        
        try:
            genre_preds = time_based_features.genre_predictions
            genre_labels = time_based_features.genre_labels
            time_stamps = time_based_features.time_stamps
            
            # Get top 5 genres for each time segment
            top_genres_data = []
            for i in range(genre_preds.shape[0]):
                sorted_indices = np.argsort(genre_preds[i])[::-1]
                top_genres = [genre_labels[idx] for idx in sorted_indices[:5]]
                top_genres_data.append(top_genres)
            
            # Create a DataFrame for easier plotting
            df = pd.DataFrame(top_genres_data, index=time_stamps)
            
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot each genre as a line
            for i, genre in enumerate(df.columns):
                # Shorten the genre name for display
                if isinstance(genre, str) and '---' in genre:
                    display_name = genre.split('---')[1]
                else:
                    display_name = str(genre)
                
                # Get the probability values for this genre
                probs = []
                for j in range(len(time_stamps)):
                    # Make sure top_genres_data[j] is iterable
                    if isinstance(top_genres_data[j], list) and genre in top_genres_data[j]:
                        idx = top_genres_data[j].index(genre)
                        # Get the probability for this genre
                        sorted_indices = np.argsort(genre_preds[j])[::-1]
                        if idx < len(sorted_indices):
                            probs.append(genre_preds[j, sorted_indices[idx]])
                        else:
                            probs.append(0)
                    else:
                        probs.append(0)
                
                ax.plot(time_stamps, probs, label=display_name)
            
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Genre Probability')
            ax.set_title('Genre Predictions Timeline')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format x-axis to show time in minutes:seconds
            time_labels = [f"{t/60:.1f}:{t%60:.0f}" for t in time_stamps[::max(1, len(time_stamps)//10)]]
            ax.set_xticks(time_stamps[::max(1, len(time_stamps)//10)])
            ax.set_xticklabels(time_labels, rotation=45)
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Genre timeline saved to {output_path}")
                
            return fig
            
        except Exception as e:
            print(f"Error creating genre timeline: {e}")
            return None
    
    def create_feature_timeline(self, time_based_features: TimeBasedFeatures,
                              output_path: Optional[PathLike] = None) -> plt.Figure:
        """Create a timeline plot of key features"""
        features = time_based_features.features
        time_stamps = time_based_features.time_stamps
        
        # Select key features for timeline
        key_features = ['danceability', 'energy', 'valence']
        available_features = [f for f in key_features if f in features and len(features[f]) > 0]
        
        if not available_features:
            print("No suitable features for timeline")
            return None
            
        # Create subplots
        fig, axes = plt.subplots(len(available_features), 1, 
                                figsize=(12, 2*len(available_features)), 
                                sharex=True)
        
        if len(available_features) == 1:
            axes = [axes]
            
        for i, feature in enumerate(available_features):
            axes[i].plot(time_stamps, features[feature], label=feature)
            axes[i].set_ylabel(feature.capitalize())
            axes[i].grid(True, alpha=0.3)
            axes[i].legend()
            
        axes[-1].set_xlabel('Time (seconds)')
        plt.suptitle('Audio Features Timeline', y=1.02)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Timeline saved to {output_path}")
            
        return fig
    
    def create_segmentation_visualization(self, segmentation_result: SegmentationResult,
                                        output_path: Optional[PathLike] = None) -> plt.Figure:
        """Create a visualization of track segmentation results"""
        segments = segmentation_result.segments
        cluster_labels = segmentation_result.cluster_labels
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                gridspec_kw={'height_ratios': [2, 1, 1]})
        
        # 1. Segment timeline with clusters
        ax1 = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, segmentation_result.num_clusters))
        
        for segment in segments:
            color = colors[segment.cluster_id % len(colors)]
            ax1.barh(0, segment.end_time - segment.start_time,
                    left=segment.start_time, height=0.5,
                    color=color, alpha=0.7,
                    label=f'Cluster {segment.cluster_id}' if segment.segment_id == 0 or
                    (segment.segment_id > 0 and segments[segment.segment_id-1].cluster_id != segment.cluster_id) else "")
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_title('Track Segmentation by Feature Similarity')
        ax1.set_yticks([])
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis to show time in minutes:seconds
        max_time = max([s.end_time for s in segments]) if segments else 100
        time_ticks = np.linspace(0, max_time, min(10, len(segments)))
        time_labels = [f"{t/60:.1f}:{t%60:.0f}" for t in time_ticks]
        ax1.set_xticks(time_ticks)
        ax1.set_xticklabels(time_labels)
        
        # 2. Feature heatmap by cluster
        ax2 = axes[1]
        
        # Create feature matrix for heatmap
        feature_matrix = []
        for segment in segments:
            feature_vector = [segment.features.get(name, 0) for name in segmentation_result.feature_names]
            feature_matrix.append(feature_vector)
        
        feature_matrix = np.array(feature_matrix)
        
        # Sort segments by cluster for better visualization
        sorted_indices = np.argsort(cluster_labels)
        sorted_matrix = feature_matrix[sorted_indices]
        sorted_labels = cluster_labels[sorted_indices]
        
        # Create heatmap
        im = ax2.imshow(sorted_matrix.T, aspect='auto', cmap='viridis', interpolation='nearest')
        ax2.set_xlabel('Segment (sorted by cluster)')
        ax2.set_ylabel('Features')
        ax2.set_title('Feature Values by Segment')
        ax2.set_yticks(np.arange(len(segmentation_result.feature_names)))
        ax2.set_yticklabels(segmentation_result.feature_names)
        
        # Add cluster boundaries
        cluster_boundaries = np.where(np.diff(sorted_labels) != 0)[0] + 0.5
        for boundary in cluster_boundaries:
            ax2.axvline(x=boundary, color='white', linestyle='--', linewidth=1)
        
        # Add colorbar
        plt.colorbar(im, ax=ax2, label='Normalized Feature Value')
        
        # 3. Cluster information
        ax3 = axes[2]
        ax3.axis('off')
        
        # Calculate cluster statistics
        cluster_info = {}
        for segment in segments:
            cluster_id = segment.cluster_id
            if cluster_id not in cluster_info:
                cluster_info[cluster_id] = {
                    'count': 0,
                    'duration': 0,
                    'genres': {},
                    'avg_features': {}
                }
            
            cluster_info[cluster_id]['count'] += 1
            cluster_info[cluster_id]['duration'] += segment.end_time - segment.start_time
            
            # Track dominant genres
            if segment.dominant_genre:
                genre = segment.dominant_genre
                if '---' in genre:
                    genre = genre.split('---')[1]
                if genre not in cluster_info[cluster_id]['genres']:
                    cluster_info[cluster_id]['genres'][genre] = 0
                cluster_info[cluster_id]['genres'][genre] += 1
        
        # Display cluster information
        info_text = f"Clustering Method: {segmentation_result.clustering_method}\n"
        info_text += f"Number of Clusters: {segmentation_result.num_clusters}\n"
        if segmentation_result.silhouette_score is not None:
            info_text += f"Silhouette Score: {segmentation_result.silhouette_score:.3f}\n"
        
        info_text += "\nCluster Information:\n"
        for cluster_id, info in sorted(cluster_info.items()):
            info_text += f"\nCluster {cluster_id}: {info['count']} segments, {info['duration']:.1f}s total\n"
            
            # Get top genre for this cluster
            if info['genres']:
                top_genre = max(info['genres'].items(), key=lambda x: x[1])
                info_text += f"  Dominant Genre: {top_genre[0]} ({top_genre[1]} segments)\n"
        
        ax3.text(0.05, 0.95, info_text, transform=ax3.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Segmentation visualization saved to {output_path}")
            
        return fig
    
    def create_beat_downbeat_visualization(self, audio_path: PathLike,
                                          madmom_features: MadmomFeatures,
                                          output_path: Optional[PathLike] = None,
                                          output_format: str = 'png') -> plt.Figure:
        """Create a wide visualization of waveform with beats and downbeats marked"""
        audio_path = mkpath(audio_path)
        
        # Set matplotlib parameters to handle large paths
        plt.rcParams['agg.path.chunksize'] = 10000
        plt.rcParams['path.simplify_threshold'] = 1.0
        
        # Load audio for waveform
        try:
            audio, sr = librosa.load(str(audio_path), sr=None)
            duration = len(audio) / sr
            time_axis = np.arange(len(audio)) / sr
        except Exception as e:
            print(f"Error loading audio for visualization: {e}")
            return None
        
        # Calculate figure width based on duration (wider for longer tracks)
        # Base width of 20 for 1 minute, add 10 for each additional minute
        base_width = 20
        additional_width = max(0, (duration - 60) / 60 * 10)
        figure_width = min(100, base_width + additional_width)  # Cap at 100 to prevent extremely wide figures
        
        # Create a very wide figure
        fig, ax = plt.subplots(figsize=(figure_width, 6))
        
        # Downsample the waveform for visualization if it's too long
        # Use more points for longer tracks to maintain detail
        max_points = min(200000, 50000 + duration * 500)  # Scale with duration
        if len(audio) > max_points:
            step = int(len(audio) // max_points)
            step = max(1, step)  # Ensure step is at least 1
            audio_downsampled = audio[::step]
            time_axis_downsampled = time_axis[::step]
        else:
            audio_downsampled = audio
            time_axis_downsampled = time_axis
        
        # Plot waveform
        ax.plot(time_axis_downsampled, audio_downsampled, color='gray', alpha=0.7, linewidth=0.5)
        
        # Plot Madmom beats
        if len(madmom_features.beats) > 0:
            ax.vlines(madmom_features.beats, min(audio_downsampled), max(audio_downsampled),
                     color='orange', alpha=0.6, linestyle=':', linewidth=1,
                     label=f'Madmom Beats ({len(madmom_features.beats)})')
        
        # Plot Madmom downbeats
        if len(madmom_features.downbeats) > 0:
            ax.vlines(madmom_features.downbeats, min(audio_downsampled), max(audio_downsampled),
                     color='blue', alpha=0.9, linestyle='-', linewidth=2,
                     label=f'Downbeats ({len(madmom_features.downbeats)})')
        
        # Formatting
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'Waveform with Beat and Downbeat Analysis - {audio_path.name}')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, duration)
        
        # Format x-axis to show time in minutes:seconds for better readability
        # Add more time ticks for longer tracks
        num_ticks = min(40, max(10, int(duration/10) + 1))
        time_ticks = np.linspace(0, duration, num_ticks)
        time_labels = [f"{int(t/60):.0f}:{int(t%60):.0f}" for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_labels, rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            # Convert to Path object
            output_path = mkpath(output_path)
            
            # Create both PNG and SVG versions if requested
            if output_format.lower() == 'svg':
                svg_path = output_path.with_suffix('.svg')
                plt.savefig(svg_path, format='svg', bbox_inches='tight')
                print(f"Beat and downbeat visualization saved to {svg_path}")
            else:
                # Default to PNG with high DPI
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"Beat and downbeat visualization saved to {output_path}")
                
                # Also create an SVG version for easier zooming
                svg_path = output_path.with_suffix('.svg')
                try:
                    plt.savefig(svg_path, format='svg', bbox_inches='tight')
                    print(f"Beat and downbeat visualization (SVG) saved to {svg_path}")
                except Exception as e:
                    print(f"Could not save SVG version: {e}")
        
        return fig


class TrackSegmenter:
    """Class for segmenting tracks based on feature similarity"""
    
    def __init__(self, method: str = 'kmeans', n_clusters: int = 4,
                 include_genre: bool = True, genre_type: str = 'sub', scaler: str = 'standard'):
        """
        Initialize the track segmenter
        
        Parameters
        ----------
        method : str, optional
            Clustering method ('kmeans', 'dbscan', 'hierarchical') (default: 'kmeans')
        n_clusters : int, optional
            Number of clusters for methods that require it (default: 4)
        include_genre : bool, optional
            Whether to include genre features in clustering (default: True)
        genre_type : str, optional
            Which part of the genre label to use ('primary', 'sub', 'full') (default: 'sub')
        scaler : str, optional
            Feature scaling method ('standard', 'minmax', 'none') (default: 'standard')
        """
        self.method = method
        self.n_clusters = n_clusters
        self.include_genre = include_genre
        self.genre_type = genre_type
        self.scaler_name = scaler
        
        # Initialize scaler
        if scaler == 'standard':
            self.scaler = StandardScaler()
        elif scaler == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            self.scaler = None
    
    def _extract_genre_label(self, genre_label: str) -> str:
        """
        Extract the appropriate part of the genre label based on genre_type
        
        Parameters
        ----------
        genre_label : str
            Full genre label (e.g., "Electronic---House")
            
        Returns
        -------
        str
            Extracted genre label based on genre_type
        """
        if '---' in genre_label:
            primary, sub = genre_label.split('---', 1)
            
            if self.genre_type == 'primary':
                return primary
            elif self.genre_type == 'sub':
                return sub
            else:  # 'full'
                return genre_label
        else:
            # If no separator, return the full label
            return genre_label
    
    def prepare_features(self, time_based_features: TimeBasedFeatures) -> Tuple[np.ndarray, List[str]]:
        """
        Prepare and normalize features for clustering
        
        Parameters
        ----------
        time_based_features : TimeBasedFeatures
            Time-based features from analysis
            
        Returns
        -------
        Tuple[np.ndarray, List[str]]
            Feature matrix and feature names
        """
        features = time_based_features.features
        feature_names = []
        feature_vectors = []
        
        # Select scalar features (exclude 2D arrays)
        scalar_features = [
            'danceability', 'energy', 'valence',
            'spectral_centroid', 'spectral_rolloff', 'spectral_bandwidth',
            'zero_crossing_rate'
        ]
        
        # Add scalar features
        for feature in scalar_features:
            if feature in features and len(features[feature]) > 0:
                feature_names.append(feature)
                feature_vectors.append(features[feature])
        
        # Add genre features if available and enabled
        if (self.include_genre and
            time_based_features.genre_predictions is not None and
            time_based_features.genre_labels is not None and
            len(time_based_features.genre_predictions) > 0):
            
            # Get top genre probabilities for each segment
            genre_preds = time_based_features.genre_predictions
            
            # Add top 3 genre probabilities as features
            for i in range(min(3, genre_preds.shape[1])):
                genre_probs = genre_preds[:, i]
                feature_names.append(f'genre_prob_{i+1}')
                feature_vectors.append(genre_probs)
        
        # Stack features horizontally
        if feature_vectors:
            feature_matrix = np.column_stack(feature_vectors)
        else:
            raise ValueError("No valid features found for clustering")
        
        # Apply scaling if specified
        if self.scaler is not None:
            feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        return feature_matrix, feature_names
    
    def segment_track(self, time_based_features: TimeBasedFeatures,
                     segment_duration: float = 5.0) -> SegmentationResult:
        """
        Segment the track based on feature similarity
        
        Parameters
        ----------
        time_based_features : TimeBasedFeatures
            Time-based features from analysis
        segment_duration : float, optional
            Duration of each segment in seconds (default: 5.0)
            
        Returns
        -------
        SegmentationResult
            Segmentation results with cluster information
        """
        # Prepare features for clustering
        feature_matrix, feature_names = self.prepare_features(time_based_features)
        time_stamps = time_based_features.time_stamps
        
        # Perform clustering
        if self.method == 'kmeans':
            clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(feature_matrix)
        elif self.method == 'dbscan':
            # DBSCAN doesn't require n_clusters, but we need to set eps and min_samples
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            cluster_labels = clusterer.fit_predict(feature_matrix)
            # DBSCAN may label some points as noise (-1)
            self.n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        elif self.method == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=self.n_clusters)
            cluster_labels = clusterer.fit_predict(feature_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")
        
        # Calculate silhouette score if we have more than 1 cluster
        silhouette = None
        if self.n_clusters > 1 and len(set(cluster_labels)) > 1:
            try:
                silhouette = silhouette_score(feature_matrix, cluster_labels)
            except Exception as e:
                print(f"Could not calculate silhouette score: {e}")
        
        # Create segment information
        segments = []
        for i, (time_stamp, cluster_id) in enumerate(zip(time_stamps, cluster_labels)):
            # Calculate segment boundaries
            # Use stored boundaries if available, otherwise estimate
            if time_based_features.segment_boundaries and i < len(time_based_features.segment_boundaries):
                start_time, end_time = time_based_features.segment_boundaries[i]
            else:
                # Fallback to old estimation method
                if i == 0:
                    start_time = 0.0
                else:
                    start_time = max(0.0, time_stamp - segment_duration / 2)
                
                if i == len(time_stamps) - 1:
                    # For the last segment, we don't know the exact end, so estimate
                    end_time = time_stamp + segment_duration / 2
                else:
                    end_time = time_stamps[i + 1] if i + 1 < len(time_stamps) else time_stamp + segment_duration
            
            # Extract features for this segment
            segment_features = {}
            for j, feature_name in enumerate(feature_names):
                if j < feature_matrix.shape[1]:
                    segment_features[feature_name] = float(feature_matrix[i, j])
            
            # Get dominant genre if available
            dominant_genre = None
            genre_confidence = None
            if (self.include_genre and
                time_based_features.genre_predictions is not None and
                time_based_features.genre_labels is not None and
                i < len(time_based_features.genre_predictions)):
                
                genre_probs = time_based_features.genre_predictions[i]
                if len(genre_probs) > 0:
                    top_idx = np.argmax(genre_probs)
                    if top_idx < len(time_based_features.genre_labels):
                        full_genre_label = time_based_features.genre_labels[top_idx]
                        dominant_genre = self._extract_genre_label(full_genre_label)
                        genre_confidence = float(genre_probs[top_idx])
            
            segments.append(SegmentInfo(
                start_time=start_time,
                end_time=end_time,
                segment_id=i,
                cluster_id=cluster_id,
                features=segment_features,
                dominant_genre=dominant_genre,
                genre_confidence=genre_confidence
            ))
        
        return SegmentationResult(
            segments=segments,
            num_clusters=self.n_clusters,
            cluster_labels=cluster_labels,
            feature_names=feature_names,
            clustering_method=self.method,
            silhouette_score=silhouette
        )


class ComprehensiveAnalyzer:
    """Main class for comprehensive audio analysis"""
    
    def __init__(self, enable_discogs: bool = True, discogs_model_path: Optional[str] = None, enable_madmom: bool = True):
        self.essentia_analyzer = EssentiaAnalyzer()
        self.time_analyzer = TimeBasedAnalyzer()
        self.visualizer = HeatmapVisualizer()
        self.segmenter = None
            
        # Initialize Discogs analyzer if enabled
        if enable_discogs:
            try:
                self.discogs_analyzer = DiscogsAnalyzer(model_path=discogs_model_path)
                if self.discogs_analyzer.discogs_classifier:
                    print("Discogs genre analyzer initialized successfully")
                else:
                    print("Discogs genre analyzer not available")
            except Exception as e:
                print(f"Failed to initialize Discogs genre analyzer: {e}")
                self.discogs_analyzer = None
        else:
            self.discogs_analyzer = None
        
        # Initialize Madmom analyzer if enabled
        if enable_madmom and MADMOM_AVAILABLE:
            try:
                self.madmom_analyzer = MadmomAnalyzer()
                print("Madmom beat and downbeat analyzer initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Madmom analyzer: {e}")
                self.madmom_analyzer = None
        else:
            if not MADMOM_AVAILABLE:
                print("Madmom not available. Install with: pip install madmom")
            self.madmom_analyzer = None
    
    def analyze(self, audio_path: PathLike,
                original_analysis: Optional[AnalysisResult] = None,
                output_dir: Optional[PathLike] = None,
                enable_segmentation: bool = False,
                segmentation_method: str = 'kmeans',
                n_clusters: int = 4,
                genre_type: str = 'sub') -> ComprehensiveAnalysisResult:
        """Perform comprehensive analysis on an audio file"""
        audio_path = mkpath(audio_path)
        
        print(f"Starting comprehensive analysis for {audio_path.name}")
        
        # Extract Essentia features
        print("Extracting Essentia features...")
        essentia_features = self.essentia_analyzer.extract_features(audio_path)
        
        # Get Discogs info if available
        discogs_info = None
        if self.discogs_analyzer:
            print("Analyzing genre with DiscogsResNet...")
            try:
                discogs_info = self.discogs_analyzer.analyze_genre(audio_path)
                if discogs_info and discogs_info.genres:
                    print(f"Detected genres: {', '.join(discogs_info.genres[:3])}")
            except Exception as e:
                print(f"Error analyzing genre with DiscogsResNet: {e}")
        
        # Extract Madmom features (beats and downbeats)
        madmom_features = None
        if self.madmom_analyzer:
            print("Extracting beats and downbeats with Madmom...")
            try:
                madmom_features = self.madmom_analyzer.extract_beats_and_downbeats(audio_path)
                if madmom_features:
                    print(f"Madmom detected {len(madmom_features.beats)} beats and {len(madmom_features.downbeats)} downbeats")
                    print(f"Madmom tempo: {madmom_features.tempo:.2f} BPM")
            except Exception as e:
                print(f"Error extracting beats and downbeats with Madmom: {e}")
                madmom_features = None
        
        # Extract time-based features
        print("Extracting time-based features...")
        if self.discogs_analyzer:
            time_based_features = self.time_analyzer.extract_time_features_with_genre(
                audio_path, discogs_analyzer=self.discogs_analyzer, madmom_features=madmom_features
            )
        else:
            time_based_features = self.time_analyzer.extract_time_features(audio_path, madmom_features=madmom_features)
        
        # Perform track segmentation if enabled
        segmentation_result = None
        grouped_segmentation = None
        if enable_segmentation:
            print(f"Performing track segmentation using {segmentation_method}...")
            try:
                # Initialize segmenter with specified parameters
                self.segmenter = TrackSegmenter(
                    method=segmentation_method,
                    n_clusters=n_clusters,
                    include_genre=self.discogs_analyzer is not None,
                    genre_type=genre_type  # Use the specified genre type
                )
                
                # Perform segmentation
                segmentation_result = self.segmenter.segment_track(time_based_features)
                print(f"Segmented track into {segmentation_result.num_clusters} clusters with {len(segmentation_result.segments)} segments")
                if segmentation_result.silhouette_score is not None:
                    print(f"Silhouette score: {segmentation_result.silhouette_score:.3f}")
                
                # Group similar consecutive segments
                print("Grouping similar consecutive segments...")
                grouped_segmentation = group_similar_segments(
                    segmentation_result.segments,
                    segmentation_result.feature_names,
                    similarity_threshold=0.15
                )
                print(f"Grouped {len(segmentation_result.segments)} segments into {grouped_segmentation.num_groups} groups")
                    
            except Exception as e:
                print(f"Error in track segmentation: {e}")
                segmentation_result = None
                grouped_segmentation = None
        
        # Create comprehensive result
        result = ComprehensiveAnalysisResult(
            path=audio_path,
            essentia_features=essentia_features,
            madmom_features=madmom_features,
            discogs_info=discogs_info,
            time_based_features=time_based_features,
            original_analysis=original_analysis,
            segmentation_result=segmentation_result,
            grouped_segmentation=grouped_segmentation
        )
        
        # Save results and create visualizations if output directory is provided
        if output_dir:
            self.save_results(result, output_dir)
            self.create_visualizations(result, output_dir)
        
        print("Comprehensive analysis complete!")
        return result
    
    def segment_track(self, audio_path: PathLike,
                     method: str = 'kmeans',
                     n_clusters: int = 4,
                     segment_duration: float = 5.0) -> SegmentationResult:
        """
        Segment a track based on feature similarity
        
        Parameters
        ----------
        audio_path : PathLike
            Path to the audio file
        method : str, optional
            Clustering method ('kmeans', 'dbscan', 'hierarchical') (default: 'kmeans')
        n_clusters : int, optional
            Number of clusters for methods that require it (default: 4)
        segment_duration : float, optional
            Duration of each segment in seconds (default: 5.0)
            
        Returns
        -------
        SegmentationResult
            Segmentation results with cluster information
        """
        audio_path = mkpath(audio_path)
        
        # Extract time-based features with genre if available
        if self.discogs_analyzer:
            time_based_features = self.time_analyzer.extract_time_features_with_genre(
                audio_path, discogs_analyzer=self.discogs_analyzer,
                segment_duration=segment_duration
            )
        else:
            time_based_features = self.time_analyzer.extract_time_features(
                audio_path, segment_duration=segment_duration
            )
        
        # Initialize segmenter
        self.segmenter = TrackSegmenter(
            method=method,
            n_clusters=n_clusters,
            include_genre=self.discogs_analyzer is not None,
            genre_type='sub'  # Default to sub-genre
        )
        
        # Perform segmentation
        segmentation_result = self.segmenter.segment_track(time_based_features, segment_duration)
        
        print(f"Segmented track into {segmentation_result.num_clusters} clusters")
        if segmentation_result.silhouette_score is not None:
            print(f"Silhouette score: {segmentation_result.silhouette_score:.3f}")
        
        return segmentation_result
    
    def save_results(self, result: ComprehensiveAnalysisResult, output_dir: PathLike):
        """Save analysis results to files"""
        output_dir = mkpath(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save main results as JSON
        results_file = output_dir / f"{result.path.stem}_comprehensive_analysis.json"
        
        # Convert dataclasses to JSON-serializable dicts
        result_dict = _to_serializable(asdict(result))
        
        with open(results_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def create_visualizations(self, result: ComprehensiveAnalysisResult, output_dir: PathLike):
        """Create visualizations for the analysis results"""
        output_dir = mkpath(output_dir)
        
        # Create heatmap (now includes genre information)
        heatmap_path = output_dir / f"{result.path.stem}_heatmap.png"
        self.visualizer.create_heatmap(result.time_based_features, heatmap_path)
        
        # Create timeline
        timeline_path = output_dir / f"{result.path.stem}_timeline.png"
        self.visualizer.create_feature_timeline(result.time_based_features, timeline_path)
        
        # Create genre timeline if genre information is available
        if (result.time_based_features.genre_predictions is not None and
            result.time_based_features.genre_labels is not None and
            len(result.time_based_features.genre_predictions) > 0):
            genre_timeline_path = output_dir / f"{result.path.stem}_genre_timeline.png"
            self.visualizer.create_genre_timeline(result.time_based_features, genre_timeline_path)
        
        # Create segmentation visualization if segmentation was performed
        if result.segmentation_result is not None:
            segmentation_path = output_dir / f"{result.path.stem}_segmentation.png"
            self.visualizer.create_segmentation_visualization(result.segmentation_result, segmentation_path)
        
        # Create beat and downbeat visualization if Madmom features are available
        if result.madmom_features is not None:
            beat_downbeat_path = output_dir / f"{result.path.stem}_beats_downbeats.png"
            self.visualizer.create_beat_downbeat_visualization(
                result.path, result.madmom_features, beat_downbeat_path, output_format='png'
            )
        
        print(f"Visualizations saved to {output_dir}")


def analyze_audio_comprehensive(audio_path: PathLike,
                               output_dir: Optional[PathLike] = None,
                               enable_discogs: bool = True,
                               discogs_model_path: Optional[str] = None,
                               enable_madmom: bool = True,
                               original_analysis: Optional[AnalysisResult] = None,
                               enable_segmentation: bool = False,
                               segmentation_method: str = 'kmeans',
                               n_clusters: int = 4,
                               genre_type: str = 'sub') -> ComprehensiveAnalysisResult:
    """
    Convenience function for comprehensive audio analysis
    
    Parameters
    ----------
    audio_path : PathLike
        Path to the audio file to analyze
    output_dir : PathLike, optional
        Directory to save results and visualizations
    enable_discogs : bool, optional
        Whether to enable Discogs genre classification (default: True)
    discogs_model_path : str, optional
        Path to the Discogs model file. Required if the model is not in Essentia's installation.
    enable_madmom : bool, optional
        Whether to enable Madmom beat and downbeat analysis (default: True)
    original_analysis : AnalysisResult, optional
        Original analysis result from the main allin1 analyzer
    enable_segmentation : bool, optional
        Whether to enable track segmentation (default: False)
    segmentation_method : str, optional
        Clustering method for segmentation ('kmeans', 'dbscan', 'hierarchical') (default: 'kmeans')
    n_clusters : int, optional
        Number of clusters for segmentation (default: 4)
    genre_type : str, optional
        Which part of the genre label to use for segmentation ('primary', 'sub', 'full') (default: 'sub')
        
    Returns
    -------
    ComprehensiveAnalysisResult
        Comprehensive analysis results
    """
    analyzer = ComprehensiveAnalyzer(
        enable_discogs=enable_discogs,
        discogs_model_path=discogs_model_path,
        enable_madmom=enable_madmom
    )
    return analyzer.analyze(
        audio_path, original_analysis, output_dir,
        enable_segmentation, segmentation_method, n_clusters, genre_type
    )


def segment_audio_track(audio_path: PathLike,
                       method: str = 'kmeans',
                       n_clusters: int = 4,
                       segment_duration: float = 5.0,
                       discogs_model_path: Optional[str] = None,
                       enable_madmom: bool = True) -> SegmentationResult:
    """
    Convenience function for track segmentation
    
    Parameters
    ----------
    audio_path : PathLike
        Path to the audio file to segment
    method : str, optional
        Clustering method ('kmeans', 'dbscan', 'hierarchical') (default: 'kmeans')
    n_clusters : int, optional
        Number of clusters for methods that require it (default: 4)
    segment_duration : float, optional
        Duration of each segment in seconds (default: 5.0)
    discogs_model_path : str, optional
        Path to the Discogs model file. Required if the model is not in Essentia's installation.
    enable_madmom : bool, optional
        Whether to enable Madmom beat and downbeat analysis (default: True)
        
    Returns
    -------
    SegmentationResult
        Segmentation results with cluster information
    """
    analyzer = ComprehensiveAnalyzer(
        enable_discogs=True,
        discogs_model_path=discogs_model_path,
        enable_madmom=enable_madmom
    )
    return analyzer.segment_track(audio_path, method, n_clusters, segment_duration)