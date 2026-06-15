# RunPod GPU workers need CUDA libs in the base image. python:3.10-slim + pip torch
# often crash-loops on startup (container "begin" x6, job stuck IN_QUEUE).
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch/torchaudio already in the RunPod PyTorch base — only install worker deps.
COPY runpod_worker_requirements.docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Handler last (frequent edits); does not bust the pip layer above on GitHub rebuilds.
COPY handler.py .

CMD ["python", "-u", "handler.py"]