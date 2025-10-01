#!/usr/bin/env python3
"""
Example script demonstrating how to use the genre over time functionality
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from allin1.comprehensive_analysis import (
    DiscogsAnalyzer, 
    TimeBasedAnalyzer, 
    HeatmapVisualizer,
    mkpath
)

def main():
    """Demonstrate genre over time analysis"""
    # Path to the audio file
    audio_path = "~/Music/test/Chemical Disco - Even Flow (Pearl Jam Tribute) [NEbTIcgt-lo].m4a"
    audio_path = mkpath(os.path.expanduser(audio_path))
    
    # Path to the Discogs model
    model_path = "autotagging/discogs-effnet-bs64-1.pb"
    
    print("=== Genre Over Time Analysis Example ===")
    
    # Initialize the Discogs analyzer
    print("Initializing Discogs analyzer...")
    discogs_analyzer = DiscogsAnalyzer(model_path=model_path)
    
    if not discogs_analyzer.discogs_classifier:
        print("Discogs classifier not available. Please check the model path.")
        return
    
    # Analyze genre over time
    print("Analyzing genre over time...")
    overall_info, time_stamps, genre_predictions = discogs_analyzer.analyze_genre_over_time(
        audio_path, segment_duration=5.0
    )
    
    if overall_info is None:
        print("Failed to analyze genre over time.")
        return
    
    print(f"Overall genres: {', '.join(overall_info.genres[:5])}")
    print(f"Analyzed {len(time_stamps)} time segments")
    
    # Create a simple visualization of genre changes
    print("Creating genre change visualization...")
    
    # Get the top genre for each segment
    top_genres = []
    top_genre_probs = []
    
    for i in range(genre_predictions.shape[0]):
        sorted_indices = np.argsort(genre_predictions[i])[::-1]
        top_genre = discogs_analyzer.genre_labels[sorted_indices[0]]
        top_genre_prob = genre_predictions[i, sorted_indices[0]]
        
        # Shorten the genre name for display
        if '---' in top_genre:
            top_genre = top_genre.split('---')[1]
        
        top_genres.append(top_genre)
        top_genre_probs.append(top_genre_prob)
    
    # Create a visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot top genre probabilities
    ax1.plot(time_stamps, top_genre_probs, 'o-')
    ax1.set_ylabel('Probability')
    ax1.set_title('Top Genre Probability Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot genre changes
    unique_genres = list(set(top_genres))
    genre_values = [unique_genres.index(genre) for genre in top_genres]
    
    ax2.step(time_stamps, genre_values, where='mid')
    ax2.set_ylabel('Genre')
    ax2.set_yticks(np.arange(len(unique_genres)))
    ax2.set_yticklabels(unique_genres)
    ax2.set_title('Genre Changes Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Format x-axis to show time in minutes:seconds
    time_labels = [f"{t/60:.1f}:{t%60:.0f}" for t in time_stamps[::max(1, len(time_stamps)//10)]]
    ax2.set_xticks(time_stamps[::max(1, len(time_stamps)//10)])
    ax2.set_xticklabels(time_labels, rotation=45)
    
    plt.tight_layout()
    plt.savefig('genre_over_time_example.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to genre_over_time_example.png")
    
    # Identify significant genre changes
    print("\n=== Genre Change Analysis ===")
    for i in range(1, len(top_genres)):
        if top_genres[i] != top_genres[i-1]:
            change_time = time_stamps[i]
            prev_genre = top_genres[i-1]
            new_genre = top_genres[i]
            print(f"Genre change at {change_time/60:.1f}:{change_time%60:.0f}: {prev_genre} → {new_genre}")
    
    print("\n=== Analysis Complete ===")

if __name__ == "__main__":
    main()