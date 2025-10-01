#!/usr/bin/env python3
"""
Test script to check each librosa component individually to identify CPU-intensive operations.
"""

import sys
import time
import librosa
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_beat_tracking(audio_path):
    """Test basic beat tracking only."""
    print("1. Testing basic beat tracking...")
    start_time = time.time()
    
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
        load_time = time.time()
        print(f"   Audio loading time: {load_time - start_time:.2f}s")
        
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_time = time.time()
        print(f"   Beat tracking time: {beat_time - load_time:.2f}s")
        
        beat_times = librosa.frames_to_time(beats, sr=sr)
        print(f"   Total time: {beat_time - start_time:.2f}s")
        print(f"   Tempo: {tempo}, Beats: {len(beats)}")
        return True, beat_time - start_time
    except Exception as e:
        print(f"   Error: {e}")
        return False, 0

def test_chroma_analysis(audio_path):
    """Test chroma feature extraction."""
    print("\n2. Testing chroma analysis...")
    start_time = time.time()
    
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
        load_time = time.time()
        print(f"   Audio loading time: {load_time - start_time:.2f}s")
        
        y_harmonic = librosa.effects.harmonic(y)
        harmonic_time = time.time()
        print(f"   Harmonic separation time: {harmonic_time - load_time:.2f}s")
        
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        chroma_time = time.time()
        print(f"   Chroma extraction time: {chroma_time - harmonic_time:.2f}s")
        print(f"   Total time: {chroma_time - start_time:.2f}s")
        print(f"   Chroma shape: {chroma.shape}")
        return True, chroma_time - start_time
    except Exception as e:
        print(f"   Error: {e}")
        return False, 0

def test_structure_analysis(audio_path):
    """Test structure analysis (most CPU-intensive)."""
    print("\n3. Testing structure analysis...")
    start_time = time.time()
    
    try:
        y, sr = librosa.load(str(audio_path), sr=None)
        load_time = time.time()
        print(f"   Audio loading time: {load_time - start_time:.2f}s")
        
        y_harmonic = librosa.effects.harmonic(y)
        harmonic_time = time.time()
        print(f"   Harmonic separation time: {harmonic_time - load_time:.2f}s")
        
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        chroma_time = time.time()
        print(f"   Chroma extraction time: {chroma_time - harmonic_time:.2f}s")
        
        # Use the more efficient direct approach without self-similarity matrix
        print("   Using direct chroma segmentation...")
        hop_length = 1024
        
        # Try with fewer segments to reduce CPU load
        print("   Computing segments (k=4)...")
        bounds_frames = librosa.segment.agglomerative(chroma, k=4)
        bound_times = librosa.frames_to_time(bounds_frames, sr=sr, hop_length=hop_length)
        segment_time = time.time()
        print(f"   Segmentation time: {segment_time - chroma_time:.2f}s")
        print(f"   Segment boundaries: {bound_times}")
        
        print(f"   Total time: {segment_time - start_time:.2f}s")
        return True, segment_time - start_time
    except Exception as e:
        print(f"   Error: {e}")
        return False, 0

def test_lightweight_structure(audio_path):
    """Test a lightweight version of structure analysis."""
    print("\n4. Testing lightweight structure analysis...")
    start_time = time.time()
    
    try:
        y, sr = librosa.load(str(audio_path), sr=None, duration=60)  # Limit to 60 seconds
        load_time = time.time()
        print(f"   Audio loading (60s limit) time: {load_time - start_time:.2f}s")
        
        # Use a simpler chroma extraction
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_time = time.time()
        print(f"   Simple chroma extraction time: {chroma_time - load_time:.2f}s")
        print(f"   Chroma shape: {chroma.shape}")
        
        # Downsample chroma to reduce computation
        chroma_downsampled = chroma[:, ::4]  # Take every 4th frame
        downsample_time = time.time()
        print(f"   Downsampling time: {downsample_time - chroma_time:.2f}s")
        print(f"   Downsampled chroma shape: {chroma_downsampled.shape}")
        
        # Use direct segmentation without similarity matrix
        print("   Using direct chroma segmentation...")
        hop_length = 4096  # Use larger hop length for downsampled chroma
        
        # Fewer segments
        print("   Computing segments (k=3)...")
        bounds_frames = librosa.segment.agglomerative(chroma_downsampled, k=3)
        bound_times = librosa.frames_to_time(bounds_frames, sr=sr, hop_length=hop_length)
        segment_time = time.time()
        print(f"   Segmentation time: {segment_time - downsample_time:.2f}s")
        print(f"   Segment boundaries: {bound_times}")
        
        print(f"   Total time: {segment_time - start_time:.2f}s")
        return True, segment_time - start_time
    except Exception as e:
        print(f"   Error: {e}")
        return False, 0

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_librosa_components.py <audio_file>")
        sys.exit(1)
    
    audio_path = Path(sys.argv[1])
    if not audio_path.exists():
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)
    
    print(f"Testing librosa components with: {audio_path}")
    print("=" * 60)
    
    # Test each component
    results = []
    
    success, time_taken = test_basic_beat_tracking(audio_path)
    results.append(("Basic Beat Tracking", success, time_taken))
    
    success, time_taken = test_chroma_analysis(audio_path)
    results.append(("Chroma Analysis", success, time_taken))
    
    success, time_taken = test_structure_analysis(audio_path)
    results.append(("Full Structure Analysis", success, time_taken))
    
    success, time_taken = test_lightweight_structure(audio_path)
    results.append(("Lightweight Structure Analysis", success, time_taken))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    for test_name, success, time_taken in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"  {test_name}: {status} ({time_taken:.2f}s)")
    
    print("\nIf your computer crashed during the 'Full Structure Analysis',")
    print("that's likely the CPU-intensive component causing the issue.")

if __name__ == "__main__":
    main()