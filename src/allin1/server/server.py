import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Dict, Optional, List, Union
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import shutil
import tempfile

from allin1.analyze import analyze
from allin1.typings import AnalysisResult
from allin1.utils import mkpath


app = FastAPI(title="Audio Processing Server", version="1.0.0")

# Global state to track processing jobs
processing_jobs: Dict[str, Dict] = {}

# Directory for storing results
RESULTS_DIR = Path("client_results")
RESULTS_DIR.mkdir(exist_ok=True)


class JobStatus(BaseModel):
    hash_key: str
    status: str  # "pending", "processing", "completed", "error"
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None


class ProcessRequest(BaseModel):
    hash_key: str


class DataRequest(BaseModel):
    hash_key: str
    data_type: str  # "all", "stems", "struct"


@app.post("/upload", status_code=202)
async def upload_audio(
    background_tasks: BackgroundTasks,
    hash_key: str,
    audio_file: UploadFile = File(...)
):
    """
    Upload audio file and start processing
    """
    # Validate hash_key
    if not hash_key or len(hash_key) > 100:
        raise HTTPException(status_code=400, detail="Invalid hash_key")
    
    # Check if job already exists
    if hash_key in processing_jobs:
        job = processing_jobs[hash_key]
        if job["status"] in ["completed", "error"]:
            # Allow reprocessing
            pass
        else:
            raise HTTPException(status_code=409, detail="Job already processing")
    
    # Create job directory
    job_dir = RESULTS_DIR / hash_key
    job_dir.mkdir(exist_ok=True)
    
    # Save uploaded file
    audio_path = job_dir / f"input_{hash_key}.wav"
    try:
        with open(audio_path, "wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save audio file: {str(e)}")
    
    # Initialize job status
    processing_jobs[hash_key] = {
        "status": "pending",
        "progress": 0.0,
        "message": "Job queued",
        "error": None,
        "audio_path": audio_path,
        "job_dir": job_dir,
        "result": None
    }
    
    # Start background processing
    background_tasks.add_task(process_audio_job, hash_key)
    
    return {"message": "File uploaded successfully", "hash_key": hash_key, "status": "queued"}


@app.get("/status/{hash_key}", response_model=JobStatus)
async def get_status(hash_key: str):
    """
    Get processing status for a job
    """
    if hash_key not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[hash_key]
    return JobStatus(
        hash_key=hash_key,
        status=job["status"],
        progress=job["progress"],
        message=job["message"],
        error=job["error"]
    )


@app.post("/data")
async def get_data(request: DataRequest):
    """
    Get processed data (stems, struct, or all)
    """
    hash_key = request.hash_key
    data_type = request.data_type
    
    if hash_key not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[hash_key]
    
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if data_type not in ["all", "stems", "struct"]:
        raise HTTPException(status_code=400, detail="Invalid data_type. Must be 'all', 'stems', or 'struct'")
    
    job_dir = job["job_dir"]
    
    if data_type == "struct":
        # Return JSON structure data
        struct_file = job_dir / f"{hash_key}.json"
        if not struct_file.exists():
            raise HTTPException(status_code=404, detail="Structure data not found")
        
        with open(struct_file, 'r') as f:
            struct_data = json.load(f)
        
        return JSONResponse(content=struct_data)
    
    elif data_type == "stems":
        # Return stems as a zip file
        stems_dir = job_dir / "stems"
        if not stems_dir.exists():
            raise HTTPException(status_code=404, detail="Stems not found")
        
        # Create zip file
        zip_path = job_dir / f"{hash_key}_stems.zip"
        shutil.make_archive(str(zip_path).replace('.zip', ''), 'zip', stems_dir)
        
        return FileResponse(
            path=zip_path,
            filename=f"{hash_key}_stems.zip",
            media_type="application/zip"
        )
    
    else:  # "all"
        # Return everything as a zip file
        zip_path = job_dir / f"{hash_key}_all.zip"
        shutil.make_archive(str(zip_path).replace('.zip', ''), 'zip', job_dir)
        
        return FileResponse(
            path=zip_path,
            filename=f"{hash_key}_all.zip",
            media_type="application/zip"
        )


@app.delete("/job/{hash_key}")
async def delete_job(hash_key: str):
    """
    Delete a job and its associated files
    """
    if hash_key not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = processing_jobs[hash_key]
    job_dir = job["job_dir"]
    
    # Remove job directory and all files
    try:
        shutil.rmtree(job_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete job files: {str(e)}")
    
    # Remove from processing jobs
    del processing_jobs[hash_key]
    
    return {"message": "Job deleted successfully"}


@app.get("/jobs")
async def list_jobs():
    """
    List all jobs and their status
    """
    jobs_summary = []
    for hash_key, job in processing_jobs.items():
        jobs_summary.append({
            "hash_key": hash_key,
            "status": job["status"],
            "progress": job["progress"],
            "message": job["message"]
        })
    
    return {"jobs": jobs_summary}


async def process_audio_job(hash_key: str):
    """
    Background task to process audio file
    """
    job = processing_jobs.get(hash_key)
    if not job:
        return
    
    try:
        # Update status to processing
        job["status"] = "processing"
        job["progress"] = 0.1
        job["message"] = "Starting audio processing"
        
        audio_path = job["audio_path"]
        job_dir = job["job_dir"]
        
        # Create stems directory
        stems_dir = job_dir / "stems"
        stems_dir.mkdir(exist_ok=True)
        
        # Step 1: Demix (stem separation)
        job["progress"] = 0.2
        job["message"] = "Performing stem separation"
        
        demix_dir = job_dir / "demix"
        demix_dir.mkdir(exist_ok=True)
        
        # Use the existing demix functionality
        from allin1.demix import demix
        demix_paths = demix([audio_path], demix_dir, "cpu")
        
        # Copy stems to the stems directory
        if demix_paths:
            demix_output_dir = demix_paths[0]
            for stem_file in ["bass.wav", "drums.wav", "other.wav", "vocals.wav"]:
                src = demix_output_dir / stem_file
                if src.exists():
                    dst = stems_dir / stem_file
                    shutil.copy2(src, dst)
        
        job["progress"] = 0.6
        job["message"] = "Stem separation completed"
        
        # Step 2: Analyze
        job["progress"] = 0.7
        job["message"] = "Performing audio analysis"
        
        # Use the existing analyze functionality
        result = analyze(
            paths=[audio_path],
            out_dir=job_dir,
            demix_dir=demix_dir,
            keep_byproducts=True,  # Keep stems
            device="cpu"
        )
        
        if isinstance(result, list):
            result = result[0]  # Take first result
        
        job["result"] = result
        job["progress"] = 0.9
        job["message"] = "Analysis completed"
        
        # Rename the JSON file to use hash_key instead of original filename
        original_json = job_dir / audio_path.with_suffix('.json').name
        target_json = job_dir / f"{hash_key}.json"
        if original_json.exists() and not target_json.exists():
            original_json.rename(target_json)
        
        # Clean up temporary files
        if demix_dir.exists():
            shutil.rmtree(demix_dir)
        
        job["progress"] = 1.0
        job["status"] = "completed"
        job["message"] = "Processing completed successfully"
        
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["message"] = f"Processing failed: {str(e)}"
        print(f"Error processing job {hash_key}: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
