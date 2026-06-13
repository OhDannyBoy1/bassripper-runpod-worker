FROM python:3.10-slim

# Install system dependencies (ffmpeg for audio, minimal to avoid bloat)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy handler and requirements (direct names)
COPY handler.py .
COPY requirements.txt .

# Pre-install torch/torchaudio with explicit CPU wheels (helps audio-separator on slim base)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install the rest
RUN pip install --no-cache-dir -r requirements.txt

# Run the handler (RunPod will execute this)
CMD ["python", "handler.py"]
