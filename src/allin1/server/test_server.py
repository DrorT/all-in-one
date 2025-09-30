#!/usr/bin/env python3
"""
Simple test script to demonstrate the Audio Processing Server functionality
"""

import requests
import time
import json
import tempfile
import os
from pathlib import Path

def test_server():
    """Test the server with a simple workflow"""
    
    base_url = "http://localhost:8000"
    
    print("🎵 Audio Processing Server Test")
    print("=" * 40)
    
    # Check if server is running
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code != 200:
            print("❌ Server is not responding properly")
            return False
    except requests.ConnectionError:
        print("❌ Cannot connect to server. Please start the server first:")
        print("   cd src/allin1/server && python run_server.py")
        return False
    
    print("✅ Server is running")
    
    # Test 1: List jobs (should be empty initially)
    print("\n1. Testing job listing...")
    try:
        response = requests.get(f"{base_url}/jobs")
        if response.status_code == 200:
            jobs = response.json()['jobs']
            print(f"   Found {len(jobs)} existing jobs")
        else:
            print(f"   ❌ Failed to list jobs: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error listing jobs: {e}")
    
    # Test 2: Create a dummy audio file for testing
    print("\n2. Creating test audio file...")
    # Create a simple silent WAV file for testing
    import wave
    import struct
    
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_audio.close()
    
    # Create a minimal WAV file (1 second of silence)
    with wave.open(temp_audio.name, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)   # 16-bit
        wav_file.setframerate(44100)  # 44.1 kHz
        
        # Write 1 second of silence
        num_frames = 44100
        silence_data = struct.pack('<' + 'h' * num_frames, *[0] * num_frames)
        wav_file.writeframes(silence_data)
    
    print(f"   ✅ Created test file: {temp_audio.name}")
    
    # Test 3: Upload audio file
    print("\n3. Testing file upload...")
    hash_key = "test_audio_001"
    
    try:
        with open(temp_audio.name, 'rb') as f:
            files = {'audio_file': f}
            response = requests.post(f"{base_url}/upload?hash_key={hash_key}", files=files)
        
        if response.status_code == 202:
            print("   ✅ File uploaded successfully")
            upload_data = response.json()
            print(f"   Job ID: {upload_data['hash_key']}")
        else:
            print(f"   ❌ Upload failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Upload error: {e}")
        return False
    
    # Test 4: Monitor job status
    print("\n4. Monitoring job status...")
    max_wait = 30  # Maximum 30 seconds wait
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{base_url}/status/{hash_key}")
            if response.status_code == 200:
                status = response.json()
                print(f"   Status: {status['status']} ({status['progress']:.1%}) - {status['message']}")
                
                if status['status'] == 'completed':
                    print("   ✅ Processing completed!")
                    break
                elif status['status'] == 'error':
                    print(f"   ❌ Processing failed: {status.get('error', 'Unknown error')}")
                    break
            else:
                print(f"   ❌ Status check failed: {response.status_code}")
                break
        except Exception as e:
            print(f"   ❌ Status check error: {e}")
            break
        
        time.sleep(2)
    
    # Test 5: Retrieve data (if completed)
    if status.get('status') == 'completed':
        print("\n5. Testing data retrieval...")
        
        # Test structure data
        try:
            response = requests.post(
                f"{base_url}/data",
                json={'hash_key': hash_key, 'data_type': 'struct'}
            )
            if response.status_code == 200:
                struct_data = response.json()
                print(f"   ✅ Structure data retrieved")
                print(f"   BPM: {struct_data.get('bpm', 'N/A')}")
                print(f"   Beats: {len(struct_data.get('beats', []))}")
                print(f"   Segments: {len(struct_data.get('segments', []))}")
            else:
                print(f"   ❌ Structure data failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Structure data error: {e}")
        
        # Test stems data
        try:
            response = requests.post(
                f"{base_url}/data",
                json={'hash_key': hash_key, 'data_type': 'stems'}
            )
            if response.status_code == 200:
                print("   ✅ Stems data retrieved (ZIP file)")
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as f:
                    f.write(response.content)
                    print(f"   Saved to: {f.name}")
            else:
                print(f"   ❌ Stems data failed: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Stems data error: {e}")
    
    # Test 6: Clean up
    print("\n6. Cleaning up...")
    try:
        response = requests.delete(f"{base_url}/job/{hash_key}")
        if response.status_code == 200:
            print("   ✅ Job deleted successfully")
        else:
            print(f"   ❌ Job deletion failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Job deletion error: {e}")
    
    # Clean up temporary audio file
    try:
        os.unlink(temp_audio.name)
        print("   ✅ Temporary files cleaned up")
    except:
        pass
    
    print("\n" + "=" * 40)
    print("✅ Server test completed!")
    print("\nTo use the server with real audio files:")
    print("1. Start the server: python run_server.py")
    print("2. Use client_example.py or curl commands to upload and process audio")
    print("3. Check http://localhost:8000/docs for API documentation")
    
    return True

if __name__ == "__main__":
    test_server()
