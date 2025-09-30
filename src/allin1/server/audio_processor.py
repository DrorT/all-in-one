#!/usr/bin/env python3
"""
Audio Processor Script

This script:
1. Receives an audio file as a parameter
2. Converts it to WAV format if needed
3. Sends it to the audio processing server
4. Tracks progress and displays status
5. Receives stems and JSON structure data
6. Saves all files to disk

Usage:
    python audio_processor.py <audio_file> [--hash_key <key>] [--server_url <url>]
"""

import argparse
import os
import sys
import tempfile
import json
import time
import zipfile
import requests
from pathlib import Path
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AudioProcessor:
    def __init__(self, server_url="http://localhost:8000"):
        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()
        
    def convert_to_wav(self, input_file, output_file=None):
        """
        Convert audio file to WAV format using ffmpeg
        
        Args:
            input_file (str): Path to input audio file
            output_file (str): Path to output WAV file (optional)
            
        Returns:
            str: Path to the converted WAV file
        """
        if output_file is None:
            output_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        
        logger.info(f"Converting {input_file} to WAV format...")
        
        try:
            # Use ffmpeg to convert to WAV
            cmd = [
                'ffmpeg', '-i', input_file,
                '-acodec', 'pcm_s16le',  # 16-bit PCM
                '-ar', '44100',          # 44.1 kHz sample rate
                '-ac', '2',              # Stereo
                '-y',                    # Overwrite output file
                output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Successfully converted to WAV: {output_file}")
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to convert audio file: {e}")
            logger.error(f"FFmpeg stderr: {e.stderr}")
            raise RuntimeError(f"Audio conversion failed: {e}")
        except FileNotFoundError:
            raise RuntimeError("ffmpeg is not installed. Please install ffmpeg to use audio conversion.")
    
    def upload_audio(self, audio_file, hash_key):
        """
        Upload audio file to the server
        
        Args:
            audio_file (str): Path to audio file
            hash_key (str): Unique identifier for the job
            
        Returns:
            dict: Server response
        """
        logger.info(f"Uploading audio file: {audio_file}")
        logger.info(f"Using hash_key: {hash_key}")
        
        try:
            with open(audio_file, 'rb') as f:
                files = {'audio_file': f}
                params = {'hash_key': hash_key}
                
                response = self.session.post(
                    f"{self.server_url}/upload",
                    files=files,
                    params=params
                )
                
            if response.status_code == 202:
                logger.info("✅ Audio uploaded successfully")
                return response.json()
            else:
                logger.error(f"❌ Upload failed: {response.status_code} - {response.text}")
                raise RuntimeError(f"Upload failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"❌ Upload error: {e}")
            raise
    
    def wait_for_completion(self, hash_key, poll_interval=5, timeout=3600):
        """
        Wait for processing to complete
        
        Args:
            hash_key (str): Job identifier
            poll_interval (int): Seconds between status checks
            timeout (int): Maximum time to wait in seconds
            
        Returns:
            bool: True if completed successfully, False if failed
        """
        logger.info(f"⏳ Waiting for job {hash_key} to complete...")
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.session.get(f"{self.server_url}/status/{hash_key}")
                
                if response.status_code == 200:
                    status = response.json()
                    progress = status.get('progress', 0) * 100
                    message = status.get('message', '')
                    
                    logger.info(f"📊 Status: {status['status']} ({progress:.1f}%) - {message}")
                    
                    if status['status'] == 'completed':
                        logger.info("✅ Processing completed successfully!")
                        return True
                    elif status['status'] == 'error':
                        error_msg = status.get('error', 'Unknown error')
                        logger.error(f"❌ Processing failed: {error_msg}")
                        return False
                        
                else:
                    logger.error(f"❌ Status check failed: {response.status_code}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Status check error: {e}")
                return False
            
            time.sleep(poll_interval)
        
        logger.error(f"❌ Timeout after {timeout} seconds")
        return False
    
    def download_results(self, hash_key, output_dir, data_types=['all']):
        """
        Download processing results
        
        Args:
            hash_key (str): Job identifier
            output_dir (str): Directory to save results
            data_types (list): Types of data to download ['all', 'stems', 'struct']
            
        Returns:
            dict: Paths to downloaded files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        downloaded_files = {}
        
        for data_type in data_types:
            logger.info(f"Downloading {data_type} data...")
            
            try:
                response = self.session.post(
                    f"{self.server_url}/data",
                    json={'hash_key': hash_key, 'data_type': data_type}
                )
                
                if response.status_code == 200:
                    if data_type == 'struct':
                        # JSON response
                        struct_data = response.json()
                        json_file = output_path / f"{hash_key}_structure.json"
                        
                        with open(json_file, 'w') as f:
                            json.dump(struct_data, f, indent=2)
                        
                        downloaded_files['struct'] = str(json_file)
                        logger.info(f"✅ Structure data saved to {json_file}")
                        
                        # Log some info about the structure
                        logger.info(f"   BPM: {struct_data.get('bpm', 'N/A')}")
                        logger.info(f"   Beats: {len(struct_data.get('beats', []))}")
                        logger.info(f"   Segments: {len(struct_data.get('segments', []))}")
                        
                    else:
                        # ZIP file response
                        if data_type == 'all':
                            zip_file = output_path / f"{hash_key}_all.zip"
                        else:  # stems
                            zip_file = output_path / f"{hash_key}_stems.zip"
                        
                        with open(zip_file, 'wb') as f:
                            f.write(response.content)
                        
                        downloaded_files[data_type] = str(zip_file)
                        logger.info(f"✅ {data_type} data saved to {zip_file}")
                        
                        # Extract ZIP file
                        extract_dir = output_path / f"{hash_key}_{data_type}"
                        extract_dir.mkdir(exist_ok=True)
                        
                        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                            zip_ref.extractall(extract_dir)
                        
                        logger.info(f"✅ Extracted to {extract_dir}")
                        
                        if data_type == 'stems':
                            # List the stem files
                            stem_files = list(extract_dir.glob("*.wav"))
                            logger.info(f"   Found {len(stem_files)} stem files:")
                            for stem_file in stem_files:
                                logger.info(f"     - {stem_file.name}")
                        
                else:
                    logger.error(f"❌ Failed to download {data_type}: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Error downloading {data_type}: {e}")
        
        return downloaded_files
    
    def cleanup_job(self, hash_key):
        """
        Delete job from server
        
        Args:
            hash_key (str): Job identifier
        """
        logger.info(f"Cleaning up job {hash_key}...")
        
        try:
            response = self.session.delete(f"{self.server_url}/job/{hash_key}")
            
            if response.status_code == 200:
                logger.info("✅ Job cleaned up successfully")
            else:
                logger.warning(f"⚠️  Job cleanup failed: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"⚠️  Job cleanup error: {e}")
    
    def process_audio_file(self, input_file, hash_key=None, output_dir=None, 
                          convert_to_wav=True, download_types=['all'], 
                          cleanup=True):
        """
        Complete audio processing workflow
        
        Args:
            input_file (str): Path to input audio file
            hash_key (str): Job identifier (auto-generated if None)
            output_dir (str): Output directory (current directory if None)
            convert_to_wav (bool): Whether to convert to WAV format
            download_types (list): Types of data to download
            cleanup (bool): Whether to clean up server job after completion
            
        Returns:
            dict: Processing results and file paths
        """
        # Generate hash_key if not provided
        if hash_key is None:
            hash_key = f"audio_{int(time.time())}"
        
        # Set output directory
        if output_dir is None:
            output_dir = Path.cwd() / "processed_audio" / hash_key
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎵 Starting audio processing for: {input_file}")
        logger.info(f"Hash key: {hash_key}")
        logger.info(f"Output directory: {output_dir}")
        
        temp_wav_file = None
        
        try:
            # Step 1: Convert to WAV if needed
            if convert_to_wav:
                temp_wav_file = self.convert_to_wav(input_file)
                audio_to_upload = temp_wav_file
            else:
                audio_to_upload = input_file
            
            # Step 2: Upload to server
            upload_result = self.upload_audio(audio_to_upload, hash_key)
            
            # Step 3: Wait for completion
            success = self.wait_for_completion(hash_key)
            
            if not success:
                raise RuntimeError("Processing failed on server")
            
            # Step 4: Download results
            downloaded_files = self.download_results(hash_key, output_dir, download_types)
            
            # Step 5: Clean up server job
            if cleanup:
                self.cleanup_job(hash_key)
            
            logger.info("🎉 Audio processing completed successfully!")
            
            return {
                'success': True,
                'hash_key': hash_key,
                'output_dir': str(output_dir),
                'downloaded_files': downloaded_files
            }
            
        except Exception as e:
            logger.error(f"❌ Processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'hash_key': hash_key
            }
            
        finally:
            # Clean up temporary WAV file
            if temp_wav_file and os.path.exists(temp_wav_file):
                try:
                    os.unlink(temp_wav_file)
                    logger.info("🧹 Cleaned up temporary WAV file")
                except:
                    pass


def main():
    parser = argparse.ArgumentParser(description='Process audio file using the audio processing server')
    parser.add_argument('audio_file', help='Path to the audio file to process')
    parser.add_argument('--hash_key', help='Unique identifier for the job (auto-generated if not provided)')
    parser.add_argument('--server_url', default='http://localhost:8000', help='Server URL (default: http://localhost:8000)')
    parser.add_argument('--output_dir', help='Output directory (default: ./processed_audio/<hash_key>)')
    parser.add_argument('--no_convert', action='store_true', help='Skip WAV conversion (assume file is already WAV)')
    parser.add_argument('--download_types', nargs='+', default=['all'], 
                       choices=['all', 'stems', 'struct'], 
                       help='Types of data to download (default: all)')
    parser.add_argument('--no_cleanup', action='store_true', help='Keep job on server after completion')
    parser.add_argument('--poll_interval', type=int, default=5, help='Status poll interval in seconds (default: 5)')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout in seconds (default: 3600)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.audio_file):
        logger.error(f"Input file not found: {args.audio_file}")
        sys.exit(1)
    
    # Create processor
    processor = AudioProcessor(args.server_url)
    
    # Process the audio file
    result = processor.process_audio_file(
        input_file=args.audio_file,
        hash_key=args.hash_key,
        output_dir=args.output_dir,
        convert_to_wav=not args.no_convert,
        download_types=args.download_types,
        cleanup=not args.no_cleanup
    )
    
    if result['success']:
        logger.info("✅ Processing completed successfully!")
        logger.info(f"Results saved to: {result['output_dir']}")
        
        if 'downloaded_files' in result:
            logger.info("Downloaded files:")
            for file_type, file_path in result['downloaded_files'].items():
                logger.info(f"  {file_type}: {file_path}")
    else:
        logger.error("❌ Processing failed!")
        logger.error(f"Error: {result.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
