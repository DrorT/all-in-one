import requests
import time
import json
from pathlib import Path


class AudioProcessingClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def upload_audio(self, hash_key: str, audio_file_path: str):
        """Upload audio file for processing"""
        url = f"{self.base_url}/upload"
        
        with open(audio_file_path, 'rb') as f:
            files = {'audio_file': f}
            params = {'hash_key': hash_key}
            
            response = requests.post(url, files=files, params=params)
            
        if response.status_code == 202:
            print(f"✅ Audio uploaded successfully. Job ID: {hash_key}")
            return response.json()
        else:
            print(f"❌ Upload failed: {response.status_code} - {response.text}")
            return None
    
    def get_status(self, hash_key: str):
        """Check processing status"""
        url = f"{self.base_url}/status/{hash_key}"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            status = response.json()
            print(f"📊 Status: {status['status']} ({status['progress']:.1%}) - {status['message']}")
            return status
        else:
            print(f"❌ Status check failed: {response.status_code} - {response.text}")
            return None
    
    def wait_for_completion(self, hash_key: str, poll_interval=5):
        """Wait for processing to complete"""
        print(f"⏳ Waiting for job {hash_key} to complete...")
        
        while True:
            status = self.get_status(hash_key)
            if not status:
                return False
            
            if status['status'] == 'completed':
                print("✅ Processing completed!")
                return True
            elif status['status'] == 'error':
                print(f"❌ Processing failed: {status.get('error', 'Unknown error')}")
                return False
            
            time.sleep(poll_interval)
    
    def get_data(self, hash_key: str, data_type: str, output_path: str = None):
        """Retrieve processed data"""
        url = f"{self.base_url}/data"
        
        data = {
            'hash_key': hash_key,
            'data_type': data_type  # 'all', 'stems', or 'struct'
        }
        
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            if data_type == 'struct':
                # Return JSON data
                struct_data = response.json()
                if output_path:
                    with open(output_path, 'w') as f:
                        json.dump(struct_data, f, indent=2)
                    print(f"✅ Structure data saved to {output_path}")
                return struct_data
            else:
                # Return file (zip)
                if output_path:
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ Data saved to {output_path}")
                return response.content
        else:
            print(f"❌ Data retrieval failed: {response.status_code} - {response.text}")
            return None
    
    def delete_job(self, hash_key: str):
        """Delete a job and its files"""
        url = f"{self.base_url}/job/{hash_key}"
        
        response = requests.delete(url)
        
        if response.status_code == 200:
            print(f"✅ Job {hash_key} deleted successfully")
            return True
        else:
            print(f"❌ Job deletion failed: {response.status_code} - {response.text}")
            return False
    
    def list_jobs(self):
        """List all jobs"""
        url = f"{self.base_url}/jobs"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            jobs = response.json()['jobs']
            print(f"📋 Found {len(jobs)} jobs:")
            for job in jobs:
                print(f"  - {job['hash_key']}: {job['status']} ({job['progress']:.1%})")
            return jobs
        else:
            print(f"❌ Failed to list jobs: {response.status_code} - {response.text}")
            return None


def main():
    """Example usage of the AudioProcessingClient"""
    client = AudioProcessingClient()
    
    # Example: Process an audio file
    hash_key = "test_audio_001"
    audio_file = "test_audio.wav"  # Replace with actual audio file path
    
    print("🎵 Audio Processing Server Client Example")
    print("=" * 50)
    
    # Step 1: Upload audio
    print("\n1. Uploading audio...")
    upload_result = client.upload_audio(hash_key, audio_file)
    if not upload_result:
        return
    
    # Step 2: Monitor progress
    print("\n2. Monitoring progress...")
    if client.wait_for_completion(hash_key):
        # Step 3: Retrieve results
        print("\n3. Retrieving results...")
        
        # Get structure data
        struct_data = client.get_data(hash_key, 'struct', f"{hash_key}_structure.json")
        if struct_data:
            print(f"   BPM: {struct_data.get('bpm', 'N/A')}")
            print(f"   Beats: {len(struct_data.get('beats', []))}")
            print(f"   Segments: {len(struct_data.get('segments', []))}")
        
        # Get stems
        stems_data = client.get_data(hash_key, 'stems', f"{hash_key}_stems.zip")
        
        # Get all data
        all_data = client.get_data(hash_key, 'all', f"{hash_key}_all.zip")
        
        print("\n✅ All data retrieved successfully!")
        
        # Optional: Clean up
        # print("\n4. Cleaning up...")
        # client.delete_job(hash_key)
    
    else:
        print("❌ Processing failed")


if __name__ == "__main__":
    main()
