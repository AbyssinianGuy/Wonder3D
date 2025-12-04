# Wonder3D API Integration Guide

This document outlines how to expose Wonder3D as an API service for generating 3D models from images, enabling integration with other services and applications.

## Overview

The goal is to create a REST API that allows clients to:
1. Submit an image and receive multi-view images (normals + colors)
2. Generate a 3D mesh from those views
3. Download the resulting 3D model (GLB/OBJ format)

## Architecture Options

### Option 1: FastAPI Service (Recommended)

A lightweight FastAPI server that wraps Wonder3D's inference and mesh generation pipelines.

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Client    │────▶│   FastAPI        │────▶│   Wonder3D      │
│  (REST/SDK) │◀────│   Server         │◀────│   Pipeline      │
└─────────────┘     └──────────────────┘     └─────────────────┘
                           │
                           ▼
                    ┌──────────────────┐
                    │  Redis/Celery    │
                    │  (Job Queue)     │
                    └──────────────────┘
```

### Option 2: Gradio API (Quick Start)

Wonder3D already has `gradio_app_recon.py` which can expose an API endpoint automatically.

### Option 3: Serverless (AWS Lambda / Google Cloud Run)

Deploy as a containerized serverless function for auto-scaling.

---

## Implementation Plan

### Phase 1: FastAPI Server

Create a new file `api/server.py`:

```python
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional
import uuid
import os
import asyncio

app = FastAPI(title="Wonder3D API", version="1.0.0")

# Job storage (use Redis in production)
jobs = {}

class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: Optional[float] = None
    result_url: Optional[str] = None
    error: Optional[str] = None

class GenerationRequest(BaseModel):
    guidance_scale: float = 1.0
    num_inference_steps: int = 20
    output_format: str = "glb"  # "glb", "obj", "ply"
    generate_mesh: bool = True

@app.post("/api/v1/generate", response_model=JobStatus)
async def generate_3d_model(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    guidance_scale: float = 1.0,
    generate_mesh: bool = True
):
    """
    Submit an image to generate a 3D model.
    Returns a job ID for tracking progress.
    """
    job_id = str(uuid.uuid4())
    
    # Save uploaded image
    upload_dir = f"./uploads/{job_id}"
    os.makedirs(upload_dir, exist_ok=True)
    image_path = f"{upload_dir}/input.png"
    
    with open(image_path, "wb") as f:
        content = await image.read()
        f.write(content)
    
    # Initialize job
    jobs[job_id] = {
        "status": "pending",
        "progress": 0.0,
        "image_path": image_path,
        "guidance_scale": guidance_scale,
        "generate_mesh": generate_mesh
    }
    
    # Queue processing
    background_tasks.add_task(process_job, job_id)
    
    return JobStatus(job_id=job_id, status="pending", progress=0.0)

@app.get("/api/v1/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress"),
        result_url=job.get("result_url"),
        error=job.get("error")
    )

@app.get("/api/v1/download/{job_id}")
async def download_result(job_id: str, format: str = "glb"):
    """Download the generated 3D model."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    file_path = job.get("mesh_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Result file not found")
    
    return FileResponse(
        file_path,
        media_type="model/gltf-binary",
        filename=f"{job_id}.glb"
    )

async def process_job(job_id: str):
    """Background task to process the generation job."""
    job = jobs[job_id]
    
    try:
        job["status"] = "processing"
        job["progress"] = 0.1
        
        # Step 1: Generate multi-view images
        await run_mvdiffusion(
            image_path=job["image_path"],
            output_dir=f"./outputs/{job_id}",
            guidance_scale=job["guidance_scale"]
        )
        job["progress"] = 0.5
        
        # Step 2: Generate mesh (if requested)
        if job["generate_mesh"]:
            mesh_path = await run_neus_reconstruction(
                input_dir=f"./outputs/{job_id}",
                output_dir=f"./meshes/{job_id}"
            )
            job["mesh_path"] = mesh_path
            job["progress"] = 1.0
        
        job["status"] = "completed"
        job["result_url"] = f"/api/v1/download/{job_id}"
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
```

### Phase 2: Pipeline Wrappers

Create `api/pipeline.py` to wrap Wonder3D's existing code:

```python
import torch
from diffusers import DiffusionPipeline
import subprocess
import asyncio
from pathlib import Path

# Global pipeline instance (loaded once)
_pipeline = None

def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = DiffusionPipeline.from_pretrained(
            'flamehaze1115/wonder3d-v1.0',
            custom_pipeline='flamehaze1115/wonder3d-pipeline',
            torch_dtype=torch.float16
        )
        _pipeline.unet.enable_xformers_memory_efficient_attention()
        if torch.cuda.is_available():
            _pipeline.to('cuda:0')
    return _pipeline

async def run_mvdiffusion(image_path: str, output_dir: str, guidance_scale: float = 1.0):
    """Run the multi-view diffusion pipeline."""
    # Use accelerate to run the existing script
    cmd = [
        "accelerate", "launch", "--config_file", "1gpu.yaml",
        "test_mvdiffusion_seq.py",
        "--config", "configs/mvdiffusion-joint-ortho-6views.yaml",
        f"validation_dataset.root_dir={Path(image_path).parent}",
        f"validation_dataset.filepaths=['{Path(image_path).name}']",
        f"save_dir={output_dir}",
        f"validation_guidance_scales=[{guidance_scale}]"
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError("Multi-view generation failed")

async def run_neus_reconstruction(input_dir: str, output_dir: str) -> str:
    """Run NeuS mesh reconstruction."""
    scene_name = Path(input_dir).name
    
    cmd = [
        "python", "NeuS/exp_runner.py",
        "--mode", "train",
        "--conf", "NeuS/confs/wmask.conf",
        "--case", scene_name,
        "--data_dir", input_dir
    ]
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd="."
    )
    await process.communicate()
    
    if process.returncode != 0:
        raise RuntimeError("Mesh reconstruction failed")
    
    mesh_path = f"NeuS/exp/neus/{scene_name}/meshes/tmp.glb"
    return mesh_path
```

### Phase 3: Docker Deployment

Update the Dockerfile for API deployment:

```dockerfile
# Add to existing Dockerfile or create docker/Dockerfile.api

# ... existing base setup ...

# Install API dependencies
RUN pip install fastapi uvicorn python-multipart redis celery

# Copy API code
COPY api/ /workspace/Wonder3D/api/

# Expose API port
EXPOSE 8000

# Default command for API server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Phase 4: Production Enhancements

#### 4.1 Job Queue with Celery + Redis

For production, use Celery for job management:

```python
# api/tasks.py
from celery import Celery

celery_app = Celery(
    'wonder3d',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

@celery_app.task(bind=True)
def generate_3d_model_task(self, job_id: str, image_path: str, options: dict):
    """Celery task for 3D model generation."""
    self.update_state(state='PROCESSING', meta={'progress': 0.1})
    
    # Run pipeline...
    
    self.update_state(state='PROCESSING', meta={'progress': 0.5})
    
    # Run mesh generation...
    
    return {'status': 'completed', 'mesh_path': mesh_path}
```

#### 4.2 GPU Resource Management

For multi-GPU or GPU sharing:

```python
# api/gpu_manager.py
import threading
from queue import Queue

class GPUManager:
    def __init__(self, gpu_ids: list[int]):
        self.gpu_queue = Queue()
        for gpu_id in gpu_ids:
            self.gpu_queue.put(gpu_id)
    
    def acquire(self) -> int:
        return self.gpu_queue.get()
    
    def release(self, gpu_id: int):
        self.gpu_queue.put(gpu_id)

# Usage
gpu_manager = GPUManager([0, 1, 2, 3])  # 4 GPUs

async def process_with_gpu(job_id: str):
    gpu_id = gpu_manager.acquire()
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        await process_job(job_id)
    finally:
        gpu_manager.release(gpu_id)
```

#### 4.3 S3/Cloud Storage Integration

```python
# api/storage.py
import boto3
from pathlib import Path

s3_client = boto3.client('s3')
BUCKET_NAME = "wonder3d-outputs"

async def upload_to_s3(local_path: str, job_id: str) -> str:
    """Upload result to S3 and return presigned URL."""
    key = f"results/{job_id}/model.glb"
    s3_client.upload_file(local_path, BUCKET_NAME, key)
    
    url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': BUCKET_NAME, 'Key': key},
        ExpiresIn=3600  # 1 hour
    )
    return url
```

---

## API Endpoints Summary

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/generate` | Submit image for 3D generation |
| GET | `/api/v1/status/{job_id}` | Check job status |
| GET | `/api/v1/download/{job_id}` | Download generated model |
| GET | `/api/v1/views/{job_id}` | Download multi-view images |
| DELETE | `/api/v1/jobs/{job_id}` | Cancel/delete a job |

---

## Client SDK Example

### Python SDK

```python
# wonder3d_client.py
import requests
import time

class Wonder3DClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def generate(self, image_path: str, wait: bool = True, **options) -> dict:
        """Generate a 3D model from an image."""
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/api/v1/generate",
                files={"image": f},
                data=options
            )
        
        result = response.json()
        job_id = result["job_id"]
        
        if wait:
            return self.wait_for_completion(job_id)
        return result
    
    def wait_for_completion(self, job_id: str, timeout: int = 300) -> dict:
        """Wait for a job to complete."""
        start = time.time()
        while time.time() - start < timeout:
            status = self.get_status(job_id)
            if status["status"] == "completed":
                return status
            if status["status"] == "failed":
                raise RuntimeError(f"Job failed: {status.get('error')}")
            time.sleep(2)
        raise TimeoutError("Job timed out")
    
    def get_status(self, job_id: str) -> dict:
        response = requests.get(f"{self.base_url}/api/v1/status/{job_id}")
        return response.json()
    
    def download(self, job_id: str, output_path: str):
        response = requests.get(
            f"{self.base_url}/api/v1/download/{job_id}",
            stream=True
        )
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

# Usage
client = Wonder3DClient("http://localhost:8000")
result = client.generate("my_image.png")
client.download(result["job_id"], "output.glb")
```

### JavaScript/TypeScript SDK

```typescript
// wonder3d-client.ts
class Wonder3DClient {
  constructor(private baseUrl: string = "http://localhost:8000") {}

  async generate(imageFile: File, options?: GenerateOptions): Promise<JobStatus> {
    const formData = new FormData();
    formData.append("image", imageFile);
    
    const response = await fetch(`${this.baseUrl}/api/v1/generate`, {
      method: "POST",
      body: formData,
    });
    
    return response.json();
  }

  async waitForCompletion(jobId: string, timeoutMs = 300000): Promise<JobStatus> {
    const start = Date.now();
    while (Date.now() - start < timeoutMs) {
      const status = await this.getStatus(jobId);
      if (status.status === "completed") return status;
      if (status.status === "failed") throw new Error(status.error);
      await new Promise(r => setTimeout(r, 2000));
    }
    throw new Error("Timeout");
  }

  async getStatus(jobId: string): Promise<JobStatus> {
    const response = await fetch(`${this.baseUrl}/api/v1/status/${jobId}`);
    return response.json();
  }

  async download(jobId: string): Promise<Blob> {
    const response = await fetch(`${this.baseUrl}/api/v1/download/${jobId}`);
    return response.blob();
  }
}
```

---

## Deployment Options

### Docker Compose (Development)

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    ports:
      - "8000:8000"
    volumes:
      - ./uploads:/workspace/Wonder3D/uploads
      - ./outputs:/workspace/Wonder3D/outputs
      - ./meshes:/workspace/Wonder3D/meshes
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  worker:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    command: celery -A api.tasks worker --loglevel=info
    volumes:
      - ./uploads:/workspace/Wonder3D/uploads
      - ./outputs:/workspace/Wonder3D/outputs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis
```

### Kubernetes (Production)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: wonder3d-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: wonder3d-api
  template:
    metadata:
      labels:
        app: wonder3d-api
    spec:
      containers:
      - name: api
        image: wonder3d/api:latest
        ports:
        - containerPort: 8000
        resources:
          limits:
            nvidia.com/gpu: 1
        volumeMounts:
        - name: storage
          mountPath: /workspace/Wonder3D/outputs
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: wonder3d-pvc
```

---

## Performance Considerations

| Stage | Time (approx) | GPU Memory |
|-------|---------------|------------|
| Multi-view generation | 60-90 seconds | ~8GB |
| NeuS mesh reconstruction | 60-120 seconds | ~4GB |
| **Total** | **2-4 minutes** | **8GB peak** |

### Optimization Strategies

1. **Model Caching**: Keep the diffusion model loaded in memory
2. **Batch Processing**: Process multiple images in parallel (requires more GPU memory)
3. **Lower Resolution**: Use smaller crop sizes for faster generation
4. **Instant-NSR**: Faster mesh reconstruction (requires tiny-cuda-nn)
5. **GPU Pooling**: Use multiple GPUs for concurrent requests

---

## Security Considerations

1. **Input Validation**: Validate image format, size, and content
2. **Rate Limiting**: Implement per-user/IP rate limits
3. **Authentication**: Add API key or OAuth2 authentication
4. **File Cleanup**: Automatically delete old uploads and results
5. **Resource Limits**: Set timeouts and memory limits per job

---

## Next Steps

1. [ ] Implement basic FastAPI server (`api/server.py`)
2. [ ] Create pipeline wrappers (`api/pipeline.py`)
3. [ ] Add Celery job queue for production
4. [ ] Create Docker Compose setup
5. [ ] Write API documentation (OpenAPI/Swagger)
6. [ ] Implement client SDKs (Python, JavaScript)
7. [ ] Add monitoring and logging (Prometheus, Grafana)
8. [ ] Deploy to cloud (AWS/GCP/Azure)
