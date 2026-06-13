"""
RunPod Serverless Handler for BassRipper Cloud Processing

This is the code that runs on RunPod's GPU.

Deploy as a Serverless endpoint:
1. Create new Serverless endpoint in RunPod dashboard
2. Use Python 3.10+ template (or custom Docker)
3. Upload this file as handler.py
4. Upload a requirements.txt (see below)
5. Choose a good GPU (RTX 4090 recommended)
6. Set "Handler" to "handler.py" or the function name
7. Deploy and copy the endpoint ID (you will use it in the desktop app later)

Input example:
{
  "audio_base64": "<base64 encoded audio>",
  "filename": "song.mp3",
  "output_format": "MP3",   # WAV, MP3 or FLAC
  "speed": "Quality"        # "Fast" or "Quality"
}

Output:
{
  "output_base64": "<base64 encoded result>",
  "filename": "song_no_bass.mp3"
}
"""

import base64
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from audio_separator.separator import Separator

# ================== CONFIG (match your local settings) ==================
BASS_KEYWORDS = ["bass", "drums+bass", "bass+drums", "low", "bassline"]
MODEL_FILENAME = "htdemucs_ft.yaml"
CHUNK_DURATION = 60

# ================== HELPER FUNCTIONS (adapted from main.py) ==================
def get_ffmpeg_path():
    """On RunPod we assume ffmpeg is available in PATH (install via template if needed)."""
    return "ffmpeg"


def convert_to_mp3(input_wav: Path, output_mp3: Path):
    """Convert WAV to MP3 using ffmpeg."""
    try:
        ffmpeg = get_ffmpeg_path()
        cmd = [
            str(ffmpeg),
            "-y",
            "-i", str(input_wav),
            "-codec:a", "libmp3lame",
            "-b:a", "192k",
            "-q:a", "0",
            str(output_mp3)
        ]
        subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MP3 conversion failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found in the RunPod environment.")


def process_audio_cloud(input_path: Path, output_format: str, speed: str, work_dir: Path) -> Path:
    """
    Core separation + mixing logic.
    Returns the path to the final mixed file.
    """
    shifts = 1 if speed == "Fast" else 2

    # Use a subfolder inside the work_dir for separated stems
    output_dir_abs = work_dir / "separated"
    output_dir_abs.mkdir(parents=True, exist_ok=True)

    separator = Separator(
        output_dir=str(output_dir_abs),
        output_format="wav",
        log_level=20,  # INFO
        model_file_dir="/tmp/models",  # Will be cached by RunPod if you use a volume
        chunk_duration=CHUNK_DURATION,
        demucs_params={
            "segment_size": "Default",
            "shifts": shifts,
            "overlap": 0.25,
            "segments_enabled": True
        }
    )

    separator.load_model(model_filename=MODEL_FILENAME)

    # Run separation
    stems = separator.separate(str(input_path))

    # Resolve stem paths
    output_dir_abs = Path(separator.output_dir)

    if isinstance(stems, dict):
        stem_paths = list(stems.values())
    else:
        stem_paths = stems

    abs_stem_paths = []
    for sp in stem_paths:
        p = Path(sp)
        if not p.is_absolute():
            p = output_dir_abs / p
        abs_stem_paths.append(p)

    # Filter out bass-related stems
    non_bass_stems = [
        p for p in abs_stem_paths
        if not any(k in p.stem.lower() for k in BASS_KEYWORDS)
    ]

    if not non_bass_stems:
        raise ValueError("No non-bass stems found after separation")

    # Mix
    mixed = None
    sample_rate = None

    for stem_path in non_bass_stems:
        data, sr = sf.read(str(stem_path), dtype='float32')
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError("Sample rate mismatch between stems")

        if mixed is None:
            mixed = data.copy()
        else:
            mixed += data

    if mixed is None:
        raise ValueError("Failed to mix stems")

    # Normalize
    max_val = np.max(np.abs(mixed))
    if max_val > 0:
        mixed = mixed / max_val * 0.95

    # Determine output
    base_name = input_path.stem
    if output_format == "MP3":
        ext = "mp3"
    elif output_format == "FLAC":
        ext = "flac"
    else:
        ext = "wav"

    output_path = work_dir / f"{base_name}_no_bass.{ext}"

    if output_format in ["WAV", "FLAC"]:
        subtype = "PCM_24" if output_format != "MP3" else None
        sf.write(str(output_path), mixed, sample_rate, subtype=subtype)
    else:
        # MP3 path
        temp_wav = work_dir / f"{base_name}_temp.wav"
        sf.write(str(temp_wav), mixed, sample_rate, subtype="PCM_24")
        convert_to_mp3(temp_wav, output_path)
        if temp_wav.exists():
            temp_wav.unlink()

    # Cleanup separated stems
    shutil.rmtree(output_dir_abs, ignore_errors=True)

    return output_path


# ================== RUNPOD HANDLER ==================
def handler(job):
    """
    Main entry point for RunPod Serverless.
    """
    job_input = job.get("input", {})

    audio_b64 = job_input.get("audio_base64")
    filename = job_input.get("filename", "input.mp3")
    output_format = job_input.get("output_format", "MP3")
    speed = job_input.get("speed", "Quality")

    if not audio_b64:
        return {"error": "Missing audio_base64 in input"}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)

            # Save input
            input_path = work_dir / filename
            input_path.write_bytes(base64.b64decode(audio_b64))

            # Process
            output_path = process_audio_cloud(input_path, output_format, speed, work_dir)

            # Return as base64
            output_bytes = output_path.read_bytes()
            output_b64 = base64.b64encode(output_bytes).decode("utf-8")

            return {
                "output_base64": output_b64,
                "filename": output_path.name
            }

    except Exception as e:
        # Always return errors so the client can fallback to local
        return {"error": str(e)}


# This is what RunPod calls
if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
