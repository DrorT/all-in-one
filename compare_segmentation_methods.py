#!/usr/bin/env python3
"""
Compare segmentation results from different libraries.

This script compares the segmentation results from librosa, madmom, and
our custom librosa segmentation approach.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple

import matplotlib.pyplot as plt
import numpy as np


def load_results(json_path: str) -> Dict[str, Any]:
    """Load analysis results from a JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def compare_segmentation(librosa_path: str, madmom_path: str, librosa_seg_path: str, output_dir: str = None):
    """Compare segmentation results from different libraries."""
    
    # Load results
    librosa_data = load_results(librosa_path)
    madmom_results = load_results(madmom_path)
    librosa_seg_results = load_results(librosa_seg_path)
    
    # Extract librosa results from the comparison file
    if 'librosa_results' in librosa_data:
        librosa_results = librosa_data['librosa_results']
    else:
        librosa_results = librosa_data
    
    # Print comparison
    print("=== Segmentation Comparison ===")
    print(f"librosa (original): {librosa_results['segment_count']} sections")
    print(f"madmom: {madmom_results['segment_count']} sections")
    print(f"librosa (segmentation): {librosa_seg_results['segment_count']} sections")
    print()
    
    print("Repeating sections:")
    print(f"librosa (original): {librosa_results['repeating_section_count']} repeating sections")
    print(f"madmom: {madmom_results['repeating_section_count']} repeating sections")
    print(f"librosa (segmentation): {librosa_seg_results['repeating_section_count']} repeating sections")
    print()
    
    # Create visualization if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
        
        duration = librosa_results['duration']
        
        # Plot segments for each method
        methods = [
            ("librosa (original)", librosa_results, 'blue'),
            ("madmom", madmom_results, 'green'),
            ("librosa (segmentation)", librosa_seg_results, 'red')
        ]
        
        for ax, (title, results, color) in zip([ax1, ax2, ax3], methods):
            ax.set_title(f"{title} - {results['segment_count']} sections")
            ax.set_ylim(0, 1)
            ax.set_xlim(0, duration)
            
            # Plot segments
            for i, segment in enumerate(results['segments']):
                start = segment['start']
                end = segment['end']
                ax.add_patch(plt.Rectangle((start, 0), end - start, 1, 
                                          facecolor=color, alpha=0.5))
                # Add section label for larger segments
                if end - start > duration / 20:  # Only label larger segments
                    ax.text((start + end) / 2, 0.5, segment['label'], 
                           ha='center', va='center', fontsize=8)
            
            ax.set_ylabel('Sections')
            ax.grid(True, alpha=0.3)
        
        ax3.set_xlabel('Time (seconds)')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = output_dir / "segmentation_comparison.png"
        plt.savefig(viz_path, dpi=150)
        print(f"Visualization saved to: {viz_path}")
        
        # Save comparison results
        comparison_results = {
            "librosa_original": {
                "segment_count": librosa_results['segment_count'],
                "repeating_section_count": librosa_results['repeating_section_count']
            },
            "madmom": {
                "segment_count": madmom_results['segment_count'],
                "repeating_section_count": madmom_results['repeating_section_count']
            },
            "librosa_segmentation": {
                "segment_count": librosa_seg_results['segment_count'],
                "repeating_section_count": librosa_seg_results['repeating_section_count']
            }
        }
        
        json_path = output_dir / "segmentation_comparison.json"
        with open(json_path, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"Comparison results saved to: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare segmentation results from different libraries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python compare_segmentation_methods.py librosa.json madmom.json librosa_seg.json
  python compare_segmentation_methods.py librosa.json madmom.json librosa_seg.json --output ./results
        """
    )
    
    parser.add_argument(
        "librosa_file",
        help="Path to the librosa analysis JSON file"
    )
    
    parser.add_argument(
        "madmom_file",
        help="Path to the madmom analysis JSON file"
    )
    
    parser.add_argument(
        "librosa_seg_file",
        help="Path to the librosa segmentation JSON file"
    )
    
    parser.add_argument(
        "--output", "-o",
        help="Directory to save the comparison results (optional)"
    )
    
    args = parser.parse_args()
    
    # Check if the files exist
    for path in [args.librosa_file, args.madmom_file, args.librosa_seg_file]:
        if not Path(path).exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)
    
    try:
        # Run the comparison
        compare_segmentation(
            args.librosa_file,
            args.madmom_file,
            args.librosa_seg_file,
            args.output
        )
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()