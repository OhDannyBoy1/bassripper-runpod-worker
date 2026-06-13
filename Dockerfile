FROM python:3.10-slim

# Install system dependencies including ffmpeg (needed for MP3 output)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy handler and requirements
COPY runpod_handler.py .
COPY runpod_worker_requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the handler (RunPod will execute this)
CMD ["python", "runpod_handler.py"]
