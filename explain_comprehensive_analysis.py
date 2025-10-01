#!/usr/bin/env python3
"""
Script to analyze and explain the comprehensive analysis JSON structure
"""

import json
import numpy as np
from pathlib import Path

def analyze_json_structure(file_path):
    """Analyze the structure of the comprehensive analysis JSON file"""
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print("COMPREHENSIVE AUDIO ANALYSIS JSON STRUCTURE EXPLANATION")
    print("=" * 60)
    print()
    
    # Top-level structure
    print("TOP-LEVEL STRUCTURE:")
    print("-" * 30)
    for key in data.keys():
        print(f"  {key}: {type(data[key]).__name__}")
    print()
    
    # Path information
    print("PATH INFORMATION:")
    print("-" * 30)
    print(f"  path: {data['path']}")
    print("  Meaning: Path to the analyzed audio file")
    print()
    
    # Essentia features
    print("ESSENTIA FEATURES:")
    print("-" * 30)
    essentia = data['essentia_features']
    print("Essentia is a library for audio analysis that extracts musical features.")
    print()
    
    # Basic musical features
    basic_features = [
        ('danceability', "How suitable the track is for dancing (0.0-1.0)"),
        ('energy', "Perceptual measure of intensity and activity (0.0-1.0)"),
        ('loudness', "Overall loudness in decibels"),
        ('valence', "Musical positiveness/happiness (0.0-1.0)"),
        ('acousticness', "Confidence that the track is acoustic (0.0-1.0)"),
        ('instrumentalness', "Confidence that the track has no vocals (0.0-1.0)"),
    ]
    
    print("Basic Musical Features:")
    for feature, description in basic_features:
        value = essentia[feature]
        print(f"  {feature}: {value:.4f} - {description}")
    print()
    
    # Rhythm and tempo features
    rhythm_features = [
        ('tempo', "Beats per minute (BPM)"),
        ('time_signature', "Number of beats per measure (e.g., 4 for 4/4 time)"),
    ]
    
    print("Rhythm and Tempo Features:")
    for feature, description in rhythm_features:
        value = essentia[feature]
        print(f"  {feature}: {value} - {description}")
    
    # Beat positions
    beats = essentia['beats']
    print(f"  beats: Array of {len(beats)} beat positions in seconds")
    print("         Represents the timing of detected beats throughout the track")
    print()
    
    # Key and mode
    key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key_name = key_names[essentia['key']] if essentia['key'] < len(key_names) else "Unknown"
    mode_name = "Major" if essentia['mode'] == 1 else "Minor"
    
    print("Key and Mode:")
    print(f"  key: {essentia['key']} ({key_name}) - Musical key (0=C, 1=C#, etc.)")
    print(f"  mode: {essentia['mode']} ({mode_name}) - Musical scale (0=minor, 1=major)")
    print()
    
    # Spectral features
    spectral_features = [
        ('spectral_centroid', "Center of mass of the spectrum (brightness)"),
        ('spectral_rolloff', "Frequency below which 85% of energy is contained"),
        ('spectral_bandwidth', "Standard deviation of the spectral centroid"),
        ('zero_crossing_rate', "Rate at which signal changes sign"),
    ]
    
    print("Spectral Features:")
    for feature, description in spectral_features:
        value = essentia[feature]
        print(f"  {feature}: {value:.4f} - {description}")
    print()
    
    # MFCC and Chroma
    mfcc = np.array(essentia['mfcc'])
    chroma = np.array(essentia['chroma'])
    
    print("Advanced Features:")
    print(f"  mfcc: Matrix of shape {mfcc.shape}")
    print("       Mel-Frequency Cepstral Coefficients - represents timbral qualities")
    print("       Shape: (n_coefficients, time_frames)")
    print()
    print(f"  chroma: Matrix of shape {chroma.shape}")
    print("       Chroma features - represents the 12 different pitch classes")
    print("       Shape: (12, time_frames) - one row for each pitch class (C, C#, D, ..., B)")
    print()
    
    # Tonal features
    print("Tonal Features:")
    tonal = essentia['tonal']
    for key, value in tonal.items():
        if isinstance(value, list):
            print(f"  {key}: Array of {len(value)} values")
        else:
            print(f"  {key}: {value}")
    print("       Advanced tonal analysis including chord detection, key strength, etc.")
    print()
    
    # Discogs info
    if data['discogs_info']:
        print("DISCOGS GENRE INFORMATION:")
        print("-" * 30)
        discogs = data['discogs_info']
        print("Discogs is a music database and marketplace. The genre classification")
        print("uses a neural network trained on Discogs metadata to predict genres.")
        print()
        
        print(f"  genres: {discogs['genres'][:5]}")
        print("          Predicted music genres (up to 5 most likely)")
        print(f"  styles: {discogs['styles']}")
        print("          More specific style classifications (often empty)")
        print(f"  year: {discogs['year']}")
        print("          Release year (not available from audio analysis)")
        print(f"  title: {discogs['title']}")
        print("          Track title (extracted from filename)")
        print(f"  artist: {discogs['artist']}")
        print("          Artist name (not available from audio analysis)")
        print()
    else:
        print("DISCOGS GENRE INFORMATION:")
        print("-" * 30)
        print("  Not available (Discogs model not loaded or analysis failed)")
        print()
    
    # Time-based features
    print("TIME-BASED FEATURES:")
    print("-" * 30)
    time_features = data['time_based_features']
    print("Time-based features track how the musical characteristics change over time.")
    print("The audio is segmented into chunks (typically 5 seconds each) and features")
    print("are extracted for each segment.")
    print()
    
    print(f"  time_stamps: Array of {len(time_features['time_stamps'])} values")
    print("               Time points (in seconds) for each segment")
    print()
    
    print("  Segment Features:")
    for feature_name, feature_values in time_features['features'].items():
        if feature_name in ['mfcc_mean', 'chroma_mean']:
            arr = np.array(feature_values)
            print(f"    {feature_name}: Matrix of shape {arr.shape}")
            if feature_name == 'mfcc_mean':
                print("                 Average MFCC coefficients for each segment")
            else:
                print("                 Average chroma features for each segment")
        else:
            print(f"    {feature_name}: Array of {len(feature_values)} values")
            # Provide specific meaning for each feature
            meanings = {
                'danceability': "How danceable each segment is",
                'energy': "Energy level of each segment",
                'valence': "Musical positiveness of each segment",
                'tempo': "Tempo (BPM) of each segment",
                'spectral_centroid': "Brightness of each segment",
                'spectral_rolloff': "Frequency distribution of each segment",
                'spectral_bandwidth': "Spectral width of each segment",
                'zero_crossing_rate': "Noisiness of each segment"
            }
            meaning = meanings.get(feature_name, "Feature value for each segment")
            print(f"                 {meaning}")
    print()
    
    # Genre predictions over time
    if 'genre_predictions' in time_features and time_features['genre_predictions']:
        genre_preds = np.array(time_features['genre_predictions'])
        print("  genre_predictions: Matrix of shape {genre_preds.shape}")
        print("                     Genre classification probabilities for each time segment")
        print("                     Shape: (n_segments, n_genres)")
        print("                     Each row contains probabilities for all possible genres")
        print()
        
        print("  genre_labels: List of genre names corresponding to prediction columns")
        print("                 These are the Discogs genre labels used for classification")
        print()
    
    # Original analysis
    if data['original_analysis']:
        print("ORIGINAL ANALYSIS:")
        print("-" * 30)
        print("Contains results from the basic allin1 analysis (beats, downbeats, segments, etc.)")
        print("This is the standard analysis output from the main allin1 library.")
        print()

if __name__ == "__main__":
    json_file = "comprehensive_results/Amazing [vuGFJv3qU58]_comprehensive_analysis.json"
    analyze_json_structure(json_file)