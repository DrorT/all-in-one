#!/usr/bin/env python3
"""
Unit test to verify the librosa BPM analysis implementation.
"""

import sys
import numpy as np
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from allin1.postprocessing.tempo import analyze_bpm_with_librosa, estimate_tempo_from_beats

def test_estimate_tempo_from_beats():
    """Test the original tempo estimation function."""
    # Test with regular beat intervals (120 BPM)
    beats_120 = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]  # 120 BPM
    bpm = estimate_tempo_from_beats(beats_120)
    print(f"Estimated BPM for 120 BPM test: {bpm}")
    assert bpm == 120, f"Expected 120 BPM, got {bpm}"
    
    # Test with irregular beat intervals
    beats_irregular = [0.0, 0.52, 1.01, 1.53, 2.05, 2.58, 3.09, 3.61]
    bpm = estimate_tempo_from_beats(beats_irregular)
    print(f"Estimated BPM for irregular beats: {bpm}")
    assert 115 <= bpm <= 125, f"Expected BPM around 120, got {bpm}"
    
    # Test with less than 2 beats
    beats_few = [0.0]
    bpm = estimate_tempo_from_beats(beats_few)
    print(f"Estimated BPM for single beat: {bpm}")
    assert bpm is None, f"Expected None for single beat, got {bpm}"
    
    print("✓ estimate_tempo_from_beats tests passed")

def test_analyze_bpm_with_librosa():
    """Test the librosa BPM analysis function with mock data."""
    # Create a mock audio file path
    mock_path = Path("/nonexistent/audio.wav")
    
    # Test with mock beat data
    model_beats = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]  # 120 BPM
    
    # This should fail gracefully since the file doesn't exist
    result = analyze_bpm_with_librosa(mock_path, model_beats)
    
    # Check that the result has the expected structure
    assert 'librosa_bpm' in result, "Missing librosa_bpm in result"
    assert 'librosa_beats' in result, "Missing librosa_beats in result"
    assert 'model_bpm' in result, "Missing model_bpm in result"
    assert 'bpm_difference' in result, "Missing bpm_difference in result"
    assert 'beat_comparison' in result, "Missing beat_comparison in result"
    
    # Check that model_bpm was calculated correctly
    assert result['model_bpm'] == 120, f"Expected model_bpm=120, got {result['model_bpm']}"
    
    # Check that error handling works
    assert 'error' in result, "Expected error field for nonexistent file"
    assert result['librosa_bpm'] is None, "Expected librosa_bpm to be None for error case"
    
    print("✓ analyze_bpm_with_librosa error handling test passed")

def test_integration():
    """Test that the integration works by importing the updated modules."""
    try:
        from allin1.helpers import run_inference
        from allin1.typings import AnalysisResult
        from allin1.postprocessing import analyze_bpm_with_librosa
        
        # Check that AnalysisResult has the librosa_analysis field
        import inspect
        result_fields = [field.name for field in inspect.signature(AnalysisResult).parameters.values()]
        assert 'librosa_analysis' in result_fields, "librosa_analysis field not found in AnalysisResult"
        
        print("✓ Integration test passed - all modules import correctly")
        return True
    except ImportError as e:
        print(f"✗ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running unit tests for librosa BPM analysis...")
    
    test_estimate_tempo_from_beats()
    test_analyze_bpm_with_librosa()
    
    if test_integration():
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)