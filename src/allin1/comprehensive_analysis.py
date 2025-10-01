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
class EssentiaFeatures:
    """Data class for Essentia extracted features"""
    danceability: float
    energy: float
    loudness: float
    valence: float
    acousticness: float
    instrumentalness: float
    tempo: float
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
    beats: np.ndarray


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
class ComprehensiveAnalysisResult:
    """Data class for comprehensive analysis results"""
    path: Path
    essentia_features: EssentiaFeatures
    discogs_info: Optional[DiscogsInfo]
    time_based_features: TimeBasedFeatures
    original_analysis: Optional[AnalysisResult] = None
    segmentation_result: Optional[SegmentationResult] = None


class EssentiaAnalyzer:
    """Class for extracting features using Essentia"""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        # Initialize Essentia algorithms - using only confirmed available algorithms
        self.danceability = es.Danceability()
        self.loudness = es.Loudness()
        self.energy = es.Energy()
        self.key_extractor = es.KeyExtractor()
        self.rhythmextractor2013 = es.RhythmExtractor2013()
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
            
            # Extract rhythm features
            rhythm_output = self.rhythmextractor2013(audio)
            if isinstance(rhythm_output, tuple):
                tempo = float(rhythm_output[0])
                beats = np.array(rhythm_output[1]) if len(rhythm_output) > 1 else np.array([])
                rhythm_confidence = float(rhythm_output[2]) if len(rhythm_output) > 2 else 0.0
            else:
                tempo = float(rhythm_output)
                beats = np.array([])
                rhythm_confidence = 0.0
            tonal_features.setdefault('rhythm_confidence', rhythm_confidence)
            
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
            
            # Extract time signature (simplified)
            time_signature = self._estimate_time_signature(beats, tempo)
            
            return EssentiaFeatures(
                danceability=danceability_value,
                energy=energy_value,
                loudness=loudness_value,
                valence=valence,
                acousticness=acousticness,
                instrumentalness=instrumentalness,
                tempo=tempo,
                key=key,
                mode=scale,  # 0 for minor, 1 for major
                time_signature=time_signature,
                spectral_centroid=centroid_mean,
                spectral_rolloff=rolloff_mean,
                spectral_bandwidth=bandwidth_mean,
                zero_crossing_rate=zcr_mean,
                mfcc=mfcc_coeffs,
                chroma=chroma_features,
                tonal=tonal_features,
                beats=beats
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
                tempo=120.0,
                key=0,
                mode=0,
                time_signature=4,
                spectral_centroid=0.5,
                spectral_rolloff=0.5,
                spectral_bandwidth=0.5,
                zero_crossing_rate=0.5,
                mfcc=np.zeros((13, 100)),  # Default MFCC shape
                chroma=np.zeros((12, 100)),  # Default chroma shape
                tonal={},
                beats=np.array([])  # Empty beat array
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
    
    def _estimate_time_signature(self, beats: np.ndarray, tempo: float) -> int:
        """Estimate time signature from beat patterns"""
        if len(beats) < 4:
            return 4  # Default to 4/4
        
        # Calculate beat intervals
        intervals = np.diff(beats)
        if len(intervals) == 0:
            return 4
            
        # Simple heuristic based on tempo and beat regularity
        regularity = 1.0 - np.std(intervals) / np.mean(intervals)
        
        if tempo > 120 and regularity > 0.8:
            return 4  # Likely 4/4
        elif tempo < 100 and regularity > 0.7:
            return 3  # Possibly 3/4
        else:
            return 4  # Default to 4/4


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
                    
                    # Debug information
                    print(f"Prediction shape: {avg_predictions.shape}")
                    print(f"Number of genre labels: {len(self.genre_labels)}")
                    
                    # Get top genres by probability
                    sorted_indices = np.argsort(avg_predictions)[::-1]
                    
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


class TimeBasedAnalyzer:
    """Class for extracting time-based features"""
    
    def __init__(self, sample_rate: int = 44100, hop_length: int = 512):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        
    def extract_time_features(self, audio_path: PathLike, 
                            segment_duration: float = 5.0) -> TimeBasedFeatures:
        """Extract time-based features by segmenting the audio"""
        audio_path = mkpath(audio_path)
        
        # Load audio
        audio, sr = librosa.load(str(audio_path), sr=self.sample_rate)
        
        # Calculate segment parameters
        segment_samples = int(segment_duration * sr)
        total_samples = len(audio)
        num_segments = int(np.ceil(total_samples / segment_samples))
        
        # Initialize feature storage
        features = {
            'danceability': [],
            'energy': [],
            'valence': [],
            'tempo': [],
            'spectral_centroid': [],
            'spectral_rolloff': [],
            'spectral_bandwidth': [],
            'zero_crossing_rate': [],
            'mfcc_mean': [],
            'chroma_mean': []
        }
        
        # Time stamps for each segment
        time_stamps = []
        
        # Initialize Essentia analyzer for segment analysis
        essentia_analyzer = EssentiaAnalyzer(self.sample_rate)
        
        # Process each segment
        for i in range(num_segments):
            start_sample = i * segment_samples
            end_sample = min((i + 1) * segment_samples, total_samples)
            segment_audio = audio[start_sample:end_sample]
            
            if len(segment_audio) < segment_samples * 0.5:  # Skip very short segments
                continue
                
            # Save segment to temporary file
            temp_path = f"temp_segment_{i}.wav"
            import soundfile as sf
            sf.write(temp_path, segment_audio, sr)
            
            try:
                # Extract features for this segment
                segment_features = essentia_analyzer.extract_features(temp_path)
                
                # Store features
                features['danceability'].append(segment_features.danceability)
                features['energy'].append(segment_features.energy)
                features['valence'].append(segment_features.valence)
                features['tempo'].append(segment_features.tempo)
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
                time_stamps.append((start_sample + end_sample) / 2 / sr)
                
            except Exception as e:
                print(f"Error processing segment {i}: {e}")
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
            genre_labels=None
        )
    
    def extract_time_features_with_genre(self, audio_path: PathLike,
                                       segment_duration: float = 5.0,
                                       discogs_analyzer=None) -> TimeBasedFeatures:
        """Extract time-based features by segmenting the audio, including genre analysis"""
        audio_path = mkpath(audio_path)
        
        # First extract standard time-based features
        time_features = self.extract_time_features(audio_path, segment_duration)
        
        # If Discogs analyzer is provided, add genre analysis
        if discogs_analyzer and discogs_analyzer.discogs_classifier:
            try:
                # Get genre predictions over time
                _, genre_time_stamps, genre_predictions = discogs_analyzer.analyze_genre_over_time(
                    audio_path, segment_duration
                )
                
                # Update the time-based features with genre information
                time_features.genre_predictions = genre_predictions
                time_features.genre_labels = discogs_analyzer.genre_labels
                
                print(f"Added genre analysis for {len(genre_time_stamps)} time segments")
                
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
        key_features = ['danceability', 'energy', 'valence', 'tempo']
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
            'danceability', 'energy', 'valence', 'tempo',
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
    
    def __init__(self, enable_discogs: bool = True, discogs_model_path: Optional[str] = None):
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
        
        # Extract time-based features
        print("Extracting time-based features...")
        if self.discogs_analyzer:
            time_based_features = self.time_analyzer.extract_time_features_with_genre(
                audio_path, discogs_analyzer=self.discogs_analyzer
            )
        else:
            time_based_features = self.time_analyzer.extract_time_features(audio_path)
        
        # Perform track segmentation if enabled
        segmentation_result = None
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
                print(f"Segmented track into {segmentation_result.num_clusters} clusters")
                if segmentation_result.silhouette_score is not None:
                    print(f"Silhouette score: {segmentation_result.silhouette_score:.3f}")
                    
            except Exception as e:
                print(f"Error in track segmentation: {e}")
                segmentation_result = None
        
        # Create comprehensive result
        result = ComprehensiveAnalysisResult(
            path=audio_path,
            essentia_features=essentia_features,
            discogs_info=discogs_info,
            time_based_features=time_based_features,
            original_analysis=original_analysis,
            segmentation_result=segmentation_result
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
        
        print(f"Visualizations saved to {output_dir}")


def analyze_audio_comprehensive(audio_path: PathLike,
                               output_dir: Optional[PathLike] = None,
                               enable_discogs: bool = True,
                               discogs_model_path: Optional[str] = None,
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
    analyzer = ComprehensiveAnalyzer(enable_discogs=enable_discogs, discogs_model_path=discogs_model_path)
    return analyzer.analyze(
        audio_path, original_analysis, output_dir,
        enable_segmentation, segmentation_method, n_clusters, genre_type
    )


def segment_audio_track(audio_path: PathLike,
                       method: str = 'kmeans',
                       n_clusters: int = 4,
                       segment_duration: float = 5.0,
                       discogs_model_path: Optional[str] = None) -> SegmentationResult:
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
        
    Returns
    -------
    SegmentationResult
        Segmentation results with cluster information
    """
    analyzer = ComprehensiveAnalyzer(enable_discogs=True, discogs_model_path=discogs_model_path)
    return analyzer.segment_track(audio_path, method, n_clusters, segment_duration)