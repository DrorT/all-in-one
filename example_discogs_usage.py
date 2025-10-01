#!/usr/bin/env python3
"""
Example script demonstrating how to use the comprehensive analysis with Discogs genre classification
"""

import sys
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from allin1.comprehensive_analysis import analyze_audio_comprehensive, ComprehensiveAnalyzer

def main():
    """Example usage of comprehensive analysis with Discogs genre classification"""
    
    print("Comprehensive Audio Analysis with Discogs Genre Classification Example")
    print("=" * 65)
    
    # Example 1: Using the convenience function with Discogs enabled
    print("\n1. Using the convenience function:")
    print("   This will analyze the audio content to determine genre")
    
    # Replace with your actual audio file path
    audio_file = "path/to/your/audio/file.wav"
    
    # Replace with the path to your Discogs model file
    discogs_model_path = "path/to/discogs-effnet-bs64-1.pb"
    
    # Check if the files exist before proceeding
    if not Path(audio_file).exists():
        print(f"   Audio file not found: {audio_file}")
        print("   Please replace 'path/to/your/audio/file.wav' with an actual audio file path")
        print("   For this example, we'll skip the actual analysis")
    else:
        try:
            # Analyze with Discogs genre classification enabled
            result = analyze_audio_comprehensive(
                audio_file,
                output_dir="analysis_output",
                enable_discogs=True,
                discogs_model_path=discogs_model_path if Path(discogs_model_path).exists() else None
            )
            
            # Print genre information if found
            if result.discogs_info and result.discogs_info.genres:
                print(f"\n   Genre Classification Results:")
                print(f"   Top genres: {', '.join(result.discogs_info.genres[:3])}")
                print(f"   All detected genres: {', '.join(result.discogs_info.genres)}")
            else:
                print("\n   No genre information found")
                
        except Exception as e:
            print(f"   Error during analysis: {e}")
    
    # Example 2: Using the class directly
    print("\n2. Using the ComprehensiveAnalyzer class directly:")
    print("   This gives you more control over the analysis process")
    
    try:
        # Initialize analyzer with Discogs enabled
        analyzer = ComprehensiveAnalyzer(
            enable_discogs=True,
            discogs_model_path=discogs_model_path if Path(discogs_model_path).exists() else None
        )
        
        # Analyze audio file
        if Path(audio_file).exists():
            result = analyzer.analyze(
                audio_file,
                output_dir="analysis_output_direct"
            )
            
            # Print genre information
            if result.discogs_info and result.discogs_info.genres:
                print(f"\n   Genre Classification Results (direct):")
                print(f"   Top genres: {', '.join(result.discogs_info.genres[:3])}")
                
    except Exception as e:
        print(f"   Error during direct analysis: {e}")
    
    # Example 3: Disabling Discogs genre classification
    print("\n3. Disabling Discogs genre classification:")
    print("   If you don't need genre information, you can disable it")
    
    if Path(audio_file).exists():
        try:
            result = analyze_audio_comprehensive(
                audio_file,
                output_dir="analysis_output_no_discogs",
                enable_discogs=False
            )
            print("   Analysis completed without genre classification")
        except Exception as e:
            print(f"   Error during analysis: {e}")
    
    # Example 4: Using only the Discogs analyzer
    print("\n4. Using only the Discogs genre classifier:")
    print("   This is useful if you only need genre information")
    
    if Path(audio_file).exists():
        try:
            from allin1.comprehensive_analysis import DiscogsAnalyzer
            
            # Initialize the Discogs analyzer
            discogs_analyzer = DiscogsAnalyzer(
                model_path=discogs_model_path if Path(discogs_model_path).exists() else None
            )
            
            # Analyze genre only
            discogs_info = discogs_analyzer.analyze_genre(audio_file)
            
            if discogs_info and discogs_info.genres:
                print(f"\n   Genre Classification Results (Discogs only):")
                print(f"   Top genres: {', '.join(discogs_info.genres[:3])}")
            else:
                print("   No genre information found")
                
        except Exception as e:
            print(f"   Error during genre-only analysis: {e}")
    
    print("\n" + "=" * 65)
    print("Example completed!")
    print("\nNotes:")
    print("- The Discogs genre classification analyzes the audio content directly")
    print("- No external API calls or authentication required")
    print("- The model works best with audio longer than 3 seconds")
    print("- The analysis will still work even if genre classification fails")
    print("- Supported genres: Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae, Rock, Electronic, Folk")
    print("\nModel Setup:")
    print("- Download the discogs-effnet-bs64-1.pb model from https://essentia.upf.edu/models/")
    print("- Navigate to the autotagging/ folder")
    print("- Download the discogs-effnet-bs64-1.pb file (standard genre classification model)")
    print("- Save the model file to a location of your choice")
    print("- Provide the model path when initializing the analyzer")

if __name__ == "__main__":
    main()