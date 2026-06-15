# RunPod GPU workers need CUDA libs in the base image. python:3.10-slim + pip torch
# often crash-loops on startup (container "begin" x6, job stuck IN_QUEUE).
FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch/torchaudio already in the RunPod PyTorch base — only install worker deps.
# Inlined so the worker GitHub repo does not need a separate requirements file.
RUN pip install --no-cache-dir \
    audio-separator==0.41.1 \
    soundfile==0.13.1 \
    "numpy>=1.26" \
    "runpod>=1.7.0" \
    onnxruntime

COPY handler.py .

CMD ["python", "-u", "handler.py"]
