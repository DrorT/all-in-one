#!/home/chester/venvs/pydemucs/bin/python
"""
Simple example to test beat_this and madmom comparison
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from allin1.comprehensive_analysis import (
    MadmomAnalyzer, 
    BeatThisAnalyzer,
    MADMOM_AVAILABLE,
    BEAT_THIS_AVAILABLE
)

def test_availability():
    """Test which libraries are available"""
    print("="*60)
    print("Beat Detection Libraries Availability")
    print("="*60)
    print(f"Madmom: {'✓ Available' if MADMOM_AVAILABLE else '✗ Not installed'}")
    print(f"beat_this: {'✓ Available' if BEAT_THIS_AVAILABLE else '✗ Not installed'}")
    print()
    
    if not MADMOM_AVAILABLE:
        print("To install Madmom: pip install madmom")
    if not BEAT_THIS_AVAILABLE:
        print("To install beat_this (requires PyTorch 2.0+):")
        print("  1. pip install tqdm einops soxr rotary-embedding-torch")
        print("  2. pip install https://github.com/CPJKU/beat_this/archive/main.zip")
        print("  3. Install ffmpeg: conda install ffmpeg (or via apt/brew)")
    print()

def quick_test(audio_path: str):
    """Quick test of both libraries"""
    print("="*60)
    print(f"Testing: {audio_path}")
    print("="*60)
    
    # Test Madmom
    if MADMOM_AVAILABLE:
        print("\nMadmom:")
        try:
            analyzer = MadmomAnalyzer()
            features = analyzer.extract_beats_and_downbeats(audio_path)
            if features:
                print(f"  ✓ Beats: {len(features.beats)}")
                print(f"  ✓ Downbeats: {len(features.downbeats)}")
                print(f"  ✓ Tempo: {features.tempo:.2f} BPM")
                print(f"  ✓ Consistency: {features.beat_consistency:.3f}")
            else:
                print("  ✗ Failed to extract features")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("\nMadmom: Not available")
    
    # Test beat_this
    if BEAT_THIS_AVAILABLE:
        print("\nbeat_this:")
        try:
            analyzer = BeatThisAnalyzer()
            features = analyzer.extract_beats_and_downbeats(audio_path)
            if features:
                print(f"  ✓ Beats: {len(features.beats)}")
                print(f"  ✓ Downbeats: {len(features.downbeats)}")
                print(f"  ✓ Tempo: {features.tempo:.2f} BPM")
                print(f"  ✓ Consistency: {features.beat_consistency:.3f}")
            else:
                print("  ✗ Failed to extract features")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    else:
        print("\nbeat_this: Not available")

if __name__ == '__main__':
    test_availability()
    
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        quick_test(audio_path)
    else:
        print("Usage: python example_beat_comparison.py <audio_file>")
        print("\nFor full comparison with metrics:")
        print("  python test_beat_comparison.py <audio_file>")
