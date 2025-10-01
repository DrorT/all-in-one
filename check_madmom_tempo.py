#!/usr/bin/env python3
"""
Check what tempo processors are available in madmom
"""

import madmom.features.tempo as tempo

print("Available attributes in madmom.features.tempo:")
for attr in dir(tempo):
    if not attr.startswith('_'):
        print(f"  {attr}")

print("\nChecking for tempo-related classes:")
tempo_classes = [attr for attr in dir(tempo) if 'Tempo' in attr and not attr.startswith('_')]
print(f"Tempo-related classes: {tempo_classes}")