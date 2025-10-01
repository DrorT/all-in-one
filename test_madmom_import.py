#!/usr/bin/env python3
"""
Test script to check madmom imports
"""

print("Testing madmom imports...")

try:
    import madmom
    print("✓ Successfully imported madmom")
    print(f"  Version: {madmom.__version__}")
except ImportError as e:
    print(f"✗ Failed to import madmom: {e}")
    exit(1)

# Test each import individually
imports_to_test = [
    ("madmom.io.audio", "load_audio_file"),
    ("madmom.features.beats", "RNNBeatProcessor"),
    ("madmom.features.beats", "BeatTrackingProcessor"),
    ("madmom.features.downbeats", "RNNDownBeatProcessor"),
    ("madmom.features.downbeats", "DBNDownBeatTrackingProcessor"),
    ("madmom.features.tempo", "TempoEstimationProcessor"),
    ("madmom.features.tempo", "RNNTempoProcessor"),
]

for module_name, import_name in imports_to_test:
    try:
        module = __import__(module_name, fromlist=[import_name])
        func_or_class = getattr(module, import_name)
        print(f"✓ Successfully imported {import_name} from {module_name}")
    except (ImportError, AttributeError) as e:
        print(f"✗ Failed to import {import_name} from {module_name}: {e}")

print("\nTesting basic madmom functionality...")
try:
    from madmom.io.audio import load_audio_file
    print("✓ Successfully imported load_audio_file")
except ImportError as e:
    print(f"✗ Failed to import load_audio_file: {e}")

try:
    from madmom.features.beats import RNNBeatProcessor
    print("✓ Successfully imported RNNBeatProcessor")
except ImportError as e:
    print(f"✗ Failed to import RNNBeatProcessor: {e}")

print("\nDone testing madmom imports.")