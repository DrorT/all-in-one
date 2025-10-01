#!/usr/bin/env python3
"""
Test script for Comprehensive Audio Analysis

This script tests the comprehensive analysis functionality.
It can be used to verify that all components are working correctly.
"""

import sys
import os
import tempfile
import numpy as np
from pathlib import Path

# Add the src directory to the path so we can import allin1 modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

def create_test_audio(duration=10, sample_rate=44100, filename="test_audio.wav"):
    """Create a simple test audio file"""
    import soundfile as sf
    
    # Generate a simple test signal: sine wave with some noise
    t = np.linspace(0, duration, int(duration * sample_rate))
    # Create a more complex signal with multiple frequencies
    signal = (
        0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.2 * np.sin(2 * np.pi * 554.37 * t) +  # C#5 note
        0.1 * np.sin(2 * np.pi * 659.25 * t) +  # E5 note
        0.05 * np.random.randn(len(t))  # Some noise
    )
    
    # Normalize
    signal = signal / np.max(np.abs(signal)) * 0.8
    
    # Add some envelope to make it more realistic
    envelope = np.exp(-t / (duration / 2))  # Exponential decay
    signal = signal * envelope
    
    # Save to file
    sf.write(filename, signal, sample_rate)
    print(f"Created test audio file: {filename}")
    return filename

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        from allin1.comprehensive_analysis import (
            EssentiaAnalyzer, 
            DiscogsAnalyzer, 
            TimeBasedAnalyzer,
            HeatmapVisualizer,
            ComprehensiveAnalyzer,
            analyze_audio_comprehensive
        )
        print("✓ All comprehensive analysis modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_essentia_analyzer():
    """Test the Essentia analyzer"""
    print("\nTesting Essentia analyzer...")
    
    try:
        from allin1.comprehensive_analysis import EssentiaAnalyzer
        
        # Create test audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            test_file = create_test_audio(duration=5, filename=tmp.name)
        
        # Test analysis
        analyzer = EssentiaAnalyzer()
        features = analyzer.extract_features(test_file)
        
        # Check that we got reasonable values
        assert 0 <= features.danceability <= 1, "Danceability should be between 0 and 1"
        assert 0 <= features.energy <= 1, "Energy should be between 0 and 1"
        assert features.tempo > 0, "Tempo should be positive"
        assert 0 <= features.key <= 11, "Key should be between 0 and 11"
        assert features.mode in [0, 1], "Mode should be 0 (minor) or 1 (major)"
        
        print("✓ Essentia analyzer test passed")
        print(f"  Danceability: {features.danceability:.3f}")
        print(f"  Energy: {features.energy:.3f}")
        print(f"  Tempo: {features.tempo:.1f} BPM")
        print(f"  Key: {features.key} ({'Major' if features.mode == 1 else 'Minor'})")
        
        # Clean up
        os.unlink(test_file)
        return True
        
    except Exception as e:
        print(f"✗ Essentia analyzer test failed: {e}")
        return False

def test_time_based_analyzer():
    """Test the time-based analyzer"""
    print("\nTesting time-based analyzer...")
    
    try:
        from allin1.comprehensive_analysis import TimeBasedAnalyzer
        
        # Create test audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            test_file = create_test_audio(duration=10, filename=tmp.name)
        
        # Test analysis
        analyzer = TimeBasedAnalyzer()
        features = analyzer.extract_time_features(test_file, segment_duration=2.0)
        
        # Check that we got reasonable results
        assert len(features.time_stamps) > 0, "Should have extracted time stamps"
        assert len(features.feature_names) > 0, "Should have feature names"
        assert all(name in features.features for name in features.feature_names), "All features should be present"
        
        print("✓ Time-based analyzer test passed")
        print(f"  Analyzed {len(features.time_stamps)} segments")
        print(f"  Duration: {features.time_stamps[-1]:.1f} seconds")
        print(f"  Features: {', '.join(features.feature_names)}")
        
        # Clean up
        os.unlink(test_file)
        return True
        
    except Exception as e:
        print(f"✗ Time-based analyzer test failed: {e}")
        return False

def test_visualizer():
    """Test the heatmap visualizer"""
    print("\nTesting heatmap visualizer...")
    
    try:
        from allin1.comprehensive_analysis import TimeBasedAnalyzer, HeatmapVisualizer
        
        # Create test audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            test_file = create_test_audio(duration=10, filename=tmp.name)
        
        # Extract features
        analyzer = TimeBasedAnalyzer()
        features = analyzer.extract_time_features(test_file, segment_duration=2.0)
        
        # Test visualization
        visualizer = HeatmapVisualizer()
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            heatmap_path = tmp.name
        
        fig = visualizer.create_heatmap(features, heatmap_path)
        assert fig is not None, "Should have created a figure"
        assert os.path.exists(heatmap_path), "Heatmap file should exist"
        
        print("✓ Heatmap visualizer test passed")
        print(f"  Heatmap saved to: {heatmap_path}")
        
        # Clean up
        os.unlink(test_file)
        os.unlink(heatmap_path)
        return True
        
    except Exception as e:
        print(f"✗ Heatmap visualizer test failed: {e}")
        return False

def test_comprehensive_analyzer():
    """Test the comprehensive analyzer"""
    print("\nTesting comprehensive analyzer...")
    
    try:
        from allin1.comprehensive_analysis import ComprehensiveAnalyzer
        
        # Create test audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            test_file = create_test_audio(duration=10, filename=tmp.name)
        
        # Test analysis
        analyzer = ComprehensiveAnalyzer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyzer.analyze(test_file, output_dir=tmpdir)
            
            # Check results
            assert result.essentia_features is not None, "Should have Essentia features"
            assert result.time_based_features is not None, "Should have time-based features"
            
            # Check that files were created
            expected_files = [
                f"{Path(test_file).stem}_comprehensive_analysis.json",
                f"{Path(test_file).stem}_heatmap.png",
                f"{Path(test_file).stem}_timeline.png"
            ]
            
            for filename in expected_files:
                filepath = Path(tmpdir) / filename
                assert filepath.exists(), f"Expected file {filename} should exist"
        
        print("✓ Comprehensive analyzer test passed")
        print(f"  Analyzed: {result.path.name}")
        print(f"  Danceability: {result.essentia_features.danceability:.3f}")
        print(f"  Energy: {result.essentia_features.energy:.3f}")
        print(f"  Time segments: {len(result.time_based_features.time_stamps)}")
        
        # Clean up
        os.unlink(test_file)
        return True
        
    except Exception as e:
        print(f"✗ Comprehensive analyzer test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Comprehensive Audio Analysis Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_essentia_analyzer,
        test_time_based_analyzer,
        test_visualizer,
        test_comprehensive_analyzer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The comprehensive analysis system is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())