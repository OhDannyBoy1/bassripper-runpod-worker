FROM python:3.10-slim

# Install system dependencies (ffmpeg for audio, minimal to avoid bloat)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# =====================================================
# DEPLOYMENT NOTES (read this)
# =====================================================
# Two common ways to deploy on RunPod Serverless:
#
# A) "Upload code" (easiest for quick iteration):
#    - In the endpoint creation UI, choose a GPU Python template.
#    - Upload handler.py (the worker handler from the repo).
#    - Upload the worker requirements as a file named exactly "requirements.txt"
#      (source is runpod_worker_requirements.txt in the repo; it contains audio-separator + runpod + torch etc.).
#    - In Advanced / Env, set MODEL_CACHE=/runpod-volume/models (or whatever your volume mount is).
#    - Attach your Network Volume and make sure the mount path matches what you use in the handler.
#
# B) Custom Docker image (what this Dockerfile is for):
#    - Build & push this image to a registry RunPod can pull (Docker Hub, GHCR, etc.).
#    - Use the image in a "Custom" template or the serverless endpoint.
#    - The COPY below maps the repo's descriptive worker file (runpod_worker_requirements.txt)
#      to the name expected inside the container (requirements.txt).
#    - Same MODEL_CACHE env + volume attachment as above.
#
# IMPORTANT for fast/no-cold-start experience:
# - Attach a Network Volume.
# - Point MODEL_CACHE at a dir on that volume.
# - After the endpoint first reaches "Running", send one job (or use the desktop app once)
#   so the model is downloaded + cached on the volume. Subsequent workers should start healthy quickly.
# - Set min workers = 1 if you want to avoid frequent scale-to-zero cold starts.
# =====================================================

# Copy handler and requirements for the worker image.
# - handler.py goes in as-is.
# - The worker requirements live in the repo as runpod_worker_requirements.txt (to keep it clearly separate
#   from the desktop app's root requirements.txt).
# - We copy it in as "requirements.txt" because the handler and RunPod's "upload code" path expect the
#   worker deps in a file with that exact name inside the container.
COPY handler.py .
COPY runpod_worker_requirements.txt requirements.txt

# For GPU workers (recommended for speed):
# Install torch + torchaudio with CUDA support matching the RunPod template (cu121 / cu118 etc. are common).
# The base RunPod GPU templates already have CUDA drivers / toolkit.
# If you see CUDA errors, adjust the index URL.
# CPU wheels are only for local testing of the handler.
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install the rest (audio-separator etc.). The requirements file lists torch again; --no-deps or the order above usually avoids downgrades.
RUN pip install --no-cache-dir -r requirements.txt

# Optional: pre-warm the model into the image (big image, only do this if you are not using a volume).
# RUN python -c "
# import os
# os.environ['MODEL_CACHE'] = '/tmp/models'
# from audio_separator.separator import Separator
# s = Separator(model_file_dir='/tmp/models', output_dir='/tmp', output_format='wav')
# s.load_model(model_filename='htdemucs_ft.yaml')
# print('Model pre-cached in image')
# " || echo "pre-warm step failed or skipped (ok if using volume)"

# Run the handler (RunPod serverless will call it)
CMD ["python", "handler.py"]
