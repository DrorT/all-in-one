"""
Comprehensive Audio Analysis Module

This module provides advanced audio analysis capabilities using multiple libraries:
- Essentia for musical feature extraction
- MusicNN for tag-based analysis
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
musicnn_top_tags = None
try:
    from musicnn.tagger import top_tags as _musicnn_top_tags
    import keras

    keras_version = getattr(keras, "__version__", "0")
    try:
        keras_major = int(str(keras_version).split('.')[0])
    except (ValueError, TypeError):
        keras_major = 0

    if keras_major >= 3:
        MUSICNN_AVAILABLE = False
        print("MusicNN disabled: keras>=3 is not supported. Install tensorflow<=2.13 and keras<3 to enable MusicNN tags.")
    else:
        MUSICNN_AVAILABLE = True
        musicnn_top_tags = _musicnn_top_tags
except ImportError:
    MUSICNN_AVAILABLE = False
    print("MusicNN not available. Install with: pip install musicnn tensorflow<2.14 keras<3")

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
class MusicNNTags:
    """Data class for MusicNN extracted tags"""
    tags: Dict[str, float]
    top_tags: List[Tuple[str, float]]


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


@dataclass
class ComprehensiveAnalysisResult:
    """Data class for comprehensive analysis results"""
    path: Path
    essentia_features: EssentiaFeatures
    musicnn_tags: Optional[MusicNNTags]
    discogs_info: Optional[DiscogsInfo]
    time_based_features: TimeBasedFeatures
    original_analysis: Optional[AnalysisResult] = None


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


class MusicNNAnalyzer:
    """Class for extracting tags using MusicNN"""
    
    def __init__(self, model: str = 'MTT_musicnn'):
        if not MUSICNN_AVAILABLE or musicnn_top_tags is None:
            raise ImportError("MusicNN is not available. Install with: pip install musicnn tensorflow<=2.13 keras<3")
        self.model = model
        
    def extract_tags(self, audio_path: PathLike) -> MusicNNTags:
        """Extract tags using MusicNN"""
        audio_path = mkpath(audio_path)
        
        # Extract tags using MusicNN top_tags helper (returns ordered names and scores)
        tag_names, tag_scores = musicnn_top_tags(str(audio_path), model=self.model, topN=50)
        tags = {name: float(score) for name, score in zip(tag_names, tag_scores)}
        top_tags = list(zip(tag_names[:10], [float(s) for s in tag_scores[:10]]))

        return MusicNNTags(tags=tags, top_tags=top_tags)


class DiscogsAnalyzer:
    """Class for getting genre information from Discogs"""
    
    def __init__(self, user_token: Optional[str] = None):
        if not DISCOGS_AVAILABLE:
            raise ImportError("Discogs client is not available. Install with: pip install discogs-client")
        
        self.client = discogs_client.Client('AllInOne-Analyzer/1.0', user_token=user_token)
        
    def search_by_filename(self, filename: str) -> Optional[DiscogsInfo]:
        """Search Discogs by filename (extract artist and title if possible)"""
        try:
            # Simple heuristic to extract artist and title from filename
            # This is a basic implementation - could be improved
            name_parts = Path(filename).stem.split(' - ')
            
            if len(name_parts) >= 2:
                artist = name_parts[0].strip()
                title = ' - '.join(name_parts[1:]).strip()
                
                # Search on Discogs
                results = self.client.search(f'{artist} {title}', type='release')
                
                if results:
                    release = results[0]
                    return DiscogsInfo(
                        genres=release.genres if hasattr(release, 'genres') else [],
                        styles=release.styles if hasattr(release, 'styles') else [],
                        year=release.year if hasattr(release, 'year') else None,
                        title=title,
                        artist=artist
                    )
            
            return None
        except Exception as e:
            print(f"Error searching Discogs: {e}")
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
            feature_names=feature_names
        )


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
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Heatmap saved to {output_path}")
            
        return fig
    
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


class ComprehensiveAnalyzer:
    """Main class for comprehensive audio analysis"""
    
    def __init__(self, discogs_token: Optional[str] = None):
        self.essentia_analyzer = EssentiaAnalyzer()
        self.time_analyzer = TimeBasedAnalyzer()
        self.visualizer = HeatmapVisualizer()
        
        if MUSICNN_AVAILABLE:
            self.musicnn_analyzer = MusicNNAnalyzer()
        else:
            self.musicnn_analyzer = None
            
        if DISCOGS_AVAILABLE and discogs_token:
            self.discogs_analyzer = DiscogsAnalyzer(discogs_token)
        else:
            self.discogs_analyzer = None
    
    def analyze(self, audio_path: PathLike, 
                original_analysis: Optional[AnalysisResult] = None,
                output_dir: Optional[PathLike] = None) -> ComprehensiveAnalysisResult:
        """Perform comprehensive analysis on an audio file"""
        audio_path = mkpath(audio_path)
        
        print(f"Starting comprehensive analysis for {audio_path.name}")
        
        # Extract Essentia features
        print("Extracting Essentia features...")
        essentia_features = self.essentia_analyzer.extract_features(audio_path)
        
        # Extract MusicNN tags if available
        musicnn_tags = None
        if self.musicnn_analyzer:
            print("Extracting MusicNN tags...")
            try:
                musicnn_tags = self.musicnn_analyzer.extract_tags(audio_path)
            except Exception as e:
                print(f"Error extracting MusicNN tags: {e}")
        
        # Get Discogs info if available
        discogs_info = None
        if self.discogs_analyzer:
            print("Searching Discogs for genre information...")
            try:
                discogs_info = self.discogs_analyzer.search_by_filename(audio_path.name)
            except Exception as e:
                print(f"Error searching Discogs: {e}")
        
        # Extract time-based features
        print("Extracting time-based features...")
        time_based_features = self.time_analyzer.extract_time_features(audio_path)
        
        # Create comprehensive result
        result = ComprehensiveAnalysisResult(
            path=audio_path,
            essentia_features=essentia_features,
            musicnn_tags=musicnn_tags,
            discogs_info=discogs_info,
            time_based_features=time_based_features,
            original_analysis=original_analysis
        )
        
        # Save results and create visualizations if output directory is provided
        if output_dir:
            self.save_results(result, output_dir)
            self.create_visualizations(result, output_dir)
        
        print("Comprehensive analysis complete!")
        return result
    
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
        
        # Create heatmap
        heatmap_path = output_dir / f"{result.path.stem}_heatmap.png"
        self.visualizer.create_heatmap(result.time_based_features, heatmap_path)
        
        # Create timeline
        timeline_path = output_dir / f"{result.path.stem}_timeline.png"
        self.visualizer.create_feature_timeline(result.time_based_features, timeline_path)
        
        print(f"Visualizations saved to {output_dir}")


def analyze_audio_comprehensive(audio_path: PathLike, 
                               output_dir: Optional[PathLike] = None,
                               discogs_token: Optional[str] = None,
                               original_analysis: Optional[AnalysisResult] = None) -> ComprehensiveAnalysisResult:
    """
    Convenience function for comprehensive audio analysis
    
    Parameters
    ----------
    audio_path : PathLike
        Path to the audio file to analyze
    output_dir : PathLike, optional
        Directory to save results and visualizations
    discogs_token : str, optional
        Discogs API token for genre information
    original_analysis : AnalysisResult, optional
        Original analysis result from the main allin1 analyzer
        
    Returns
    -------
    ComprehensiveAnalysisResult
        Comprehensive analysis results
    """
    analyzer = ComprehensiveAnalyzer(discogs_token=discogs_token)
    return analyzer.analyze(audio_path, original_analysis, output_dir)