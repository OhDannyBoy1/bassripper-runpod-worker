"""
RunPod Serverless Handler for BassRipper Cloud Processing

This is the code that runs on RunPod's GPU.

Deploy as a Serverless endpoint:
1. Create new Serverless endpoint in RunPod dashboard (or edit existing).
2. Preferred: build & push the Dockerfile as a **Custom** container image (see below), or use "Upload code" for quick iteration.
3. For "Upload code" path: upload this file as handler.py + the worker requirements (runpod_worker_requirements.txt renamed to requirements.txt).
4. The Dockerfile (in repo) produces the proper image for the "Custom" path.
5. **Attach a Network Volume** and note the mount path (e.g. /runpod-volume). This is critical for model caching.
6. In endpoint advanced settings / env: set MODEL_CACHE=/runpod-volume/models (must match the path in handler.py and volume mount).
7. Choose a strong GPU (RTX 4090-class recommended). Set min workers=1 + reasonable idle timeout.
8. After first "Running", send at least one job (or trigger from desktop app) so the htdemucs_ft model downloads to the volume.
9. Copy the Serverless Endpoint ID + your RunPod API key into the desktop app config (see below). The desktop app now *automatically* routes to cloud when credit >= $0.05 (see MIN_CLOUD_CREDIT in main.py), with seamless local fallback.

Desktop app configuration (secrets, never committed):
- Files live in: %LOCALAPPDATA%\Kinell\BassRipper\   (or equivalent platformdirs.user_config_dir)
- Create runpod_api_key.txt (contents: your rpa_... key)
- Create runpod_endpoint_id.txt (contents: your endpoint id like pvnc91yr9qm5af)
- Optionally create dev_mode.txt (empty file) to see credit balance, technical statuses and the "cancel to local" button.
- The app calls get_runpod_credit before every rip and only uses cloud when balance is sufficient.

Note on requirements files (important):
- Root requirements.txt in the repo = for the desktop BassRipper app + PyInstaller builds (includes customtkinter etc.).
- runpod_worker_requirements.txt = only for the RunPod worker (audio-separator, runpod, torch etc.).
- Inside the worker container/image we always end up with a file named requirements.txt containing the worker deps.

Model cache (critical for fast cold starts):
  The large model is downloaded on first load_model(). Point MODEL_CACHE (env var or edit below)
  at a directory **on your Network Volume** so it persists across workers and cold starts.
  Example: set env MODEL_CACHE=/runpod-volume/models in the endpoint advanced settings.

Input example:
{
  "audio_base64": "<base64 encoded audio>",
  "filename": "song.mp3",
  "output_format": "MP3",   # WAV, MP3 or FLAC
  "speed": "Quality",        # "Fast" or "Quality"
  "save_bass_track": false   # optional — also return bass-only stem
}

Output:
{
  "output_base64": "<base64 encoded result>",
  "filename": "song_no_bass.mp3",
  "bass_output_base64": "<optional>",
  "bass_filename": "song_bass.mp3"
}
"""

import base64
import json
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

print("=== HANDLER.PY MODULE-LEVEL START ===", flush=True)
print("Python:", sys.version, flush=True)
print("MODEL_CACHE env:", os.environ.get("MODEL_CACHE"), flush=True)
print("sys.path:", sys.path[:3], "...", flush=True)

print("[handler] imports successful", flush=True)

# ================== CONFIG (match your local settings) ==================
BASS_KEYWORDS = ["bass", "drums+bass", "bass+drums", "low", "bassline"]
MODEL_FILENAME = "htdemucs_ft.yaml"
CHUNK_DURATION = 60

# RunPod /run responses are also capped at 10 MiB (same as upload).
MAX_CLOUD_RESPONSE_BYTES = int(9.5 * 1024 * 1024)
CLOUD_RETURN_BITRATES = ("128k", "96k", "64k", "48k", "32k")

# Model cache directory. Override with env var MODEL_CACHE so you can point it at your Network Volume mount.
# Example (in RunPod endpoint settings): MODEL_CACHE=/runpod-volume/models
MODEL_CACHE_DIR = os.environ.get("MODEL_CACHE", "/tmp/models")
try:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
except OSError as cache_err:
    print(
        f"[handler] WARNING: cannot create MODEL_CACHE_DIR={MODEL_CACHE_DIR}: {cache_err}. "
        "Falling back to /tmp/models (attach a Network Volume and set MODEL_CACHE for persistence).",
        flush=True,
    )
    MODEL_CACHE_DIR = "/tmp/models"
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
print(f"[handler] Using MODEL_CACHE_DIR={MODEL_CACHE_DIR} (set MODEL_CACHE env for your volume mount)", flush=True)

# ================== HELPER FUNCTIONS (adapted from main.py) ==================
def get_ffmpeg_path():
    """On RunPod we assume ffmpeg is available in PATH (install via template if needed)."""
    return "ffmpeg"


def _ffmpeg_mp3(input_path: Path, output_mp3: Path, bitrate: str) -> None:
    """CBR MP3. Do not pass -q:a — it forces VBR and can blow past RunPod's 10 MiB cap."""
    ffmpeg = get_ffmpeg_path()
    subprocess.run(
        [
            str(ffmpeg),
            "-y",
            "-i", str(input_path),
            "-vn",
            "-codec:a", "libmp3lame",
            "-b:a", bitrate,
            str(output_mp3),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def convert_to_mp3(input_wav: Path, output_mp3: Path, bitrate: str = "128k"):
    """Convert WAV to MP3 using ffmpeg."""
    try:
        _ffmpeg_mp3(input_wav, output_mp3, bitrate)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MP3 conversion failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("ffmpeg not found in the RunPod environment.")


def estimate_response_payload_bytes(
    no_bass_bytes: bytes,
    no_bass_filename: str,
    bass_bytes: bytes | None = None,
    bass_filename: str | None = None,
) -> int:
    payload = {
        "output_base64": base64.b64encode(no_bass_bytes).decode("utf-8"),
        "filename": no_bass_filename,
    }
    if bass_bytes is not None and bass_filename:
        payload["bass_output_base64"] = base64.b64encode(bass_bytes).decode("utf-8")
        payload["bass_filename"] = bass_filename
    return len(json.dumps(payload).encode("utf-8"))


def _encode_mp3_candidate(source_mp3: Path, work_dir: Path, bitrate: str) -> bytes:
    temp_out = work_dir / f"cloud_ret_{uuid.uuid4().hex}_{bitrate}.mp3"
    _ffmpeg_mp3(source_mp3, temp_out, bitrate)
    try:
        return temp_out.read_bytes()
    finally:
        temp_out.unlink(missing_ok=True)


def build_cloud_response_payload(
    no_bass_path: Path,
    work_dir: Path,
    bass_path: Path | None = None,
) -> dict:
    """Re-encode outputs until the JSON job result fits RunPod's 10 MiB response cap."""
    no_bass_name = no_bass_path.name
    bass_name = bass_path.name if bass_path else None

    for no_bass_bitrate in CLOUD_RETURN_BITRATES:
        no_bass_bytes = _encode_mp3_candidate(no_bass_path, work_dir, no_bass_bitrate)
        if bass_path is None:
            size = estimate_response_payload_bytes(no_bass_bytes, no_bass_name)
            if size <= MAX_CLOUD_RESPONSE_BYTES:
                print(
                    f"[handler] response payload {size / (1024 * 1024):.2f} MiB "
                    f"(no_bass @ {no_bass_bitrate})",
                    flush=True,
                )
                return {
                    "output_base64": base64.b64encode(no_bass_bytes).decode("utf-8"),
                    "filename": no_bass_name,
                }
            continue

        for bass_bitrate in CLOUD_RETURN_BITRATES:
            bass_bytes = _encode_mp3_candidate(bass_path, work_dir, bass_bitrate)
            size = estimate_response_payload_bytes(
                no_bass_bytes, no_bass_name, bass_bytes, bass_name
            )
            if size <= MAX_CLOUD_RESPONSE_BYTES:
                print(
                    f"[handler] response payload {size / (1024 * 1024):.2f} MiB "
                    f"(no_bass @ {no_bass_bitrate}, bass @ {bass_bitrate})",
                    flush=True,
                )
                return {
                    "output_base64": base64.b64encode(no_bass_bytes).decode("utf-8"),
                    "filename": no_bass_name,
                    "bass_output_base64": base64.b64encode(bass_bytes).decode("utf-8"),
                    "bass_filename": bass_name,
                }

    raise RuntimeError(
        "Processed audio is too large for RunPod's 10 MiB response limit. "
        "Try a shorter clip or disable the bass-only export."
    )


def _mix_stem_paths(stem_paths: list[Path], bass_only: bool) -> tuple[np.ndarray, int]:
    """Mix stems matching (bass_only) or excluding (not bass_only) BASS_KEYWORDS."""
    selected = [
        p for p in stem_paths
        if any(k in p.stem.lower() for k in BASS_KEYWORDS) == bass_only
    ]
    if not selected:
        kind = "bass" if bass_only else "non-bass"
        raise ValueError(f"No {kind} stems found")

    mixed = None
    sample_rate = None
    for stem_path in selected:
        data, sr = sf.read(str(stem_path), dtype="float32")
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise ValueError("Sample rate mismatch between stems")
        mixed = data.copy() if mixed is None else mixed + data

    max_val = np.max(np.abs(mixed))
    if max_val > 0:
        mixed = mixed / max_val * 0.95
    return mixed, sample_rate


def _write_mixed_output(
    mixed: np.ndarray,
    sample_rate: int,
    output_path: Path,
    output_format: str,
    work_dir: Path,
    temp_stem: str,
) -> Path:
    if output_format in ("WAV", "FLAC"):
        subtype = "PCM_24" if output_format != "MP3" else None
        sf.write(str(output_path), mixed, sample_rate, subtype=subtype)
        return output_path

    temp_wav = work_dir / temp_stem
    sf.write(str(temp_wav), mixed, sample_rate, subtype="PCM_24")
    convert_to_mp3(temp_wav, output_path)
    if temp_wav.exists():
        temp_wav.unlink()
    return output_path


def process_audio_cloud(
    input_path: Path,
    output_format: str,
    speed: str,
    work_dir: Path,
    save_bass_track: bool = False,
) -> tuple[Path, Path | None]:
    """
    Core separation + mixing logic.
    Returns (no_bass_path, bass_path_or_none).
    """
    shifts = 1 if speed == "Fast" else 2

    # Use a subfolder inside the work_dir for separated stems
    output_dir_abs = work_dir / "separated"
    output_dir_abs.mkdir(parents=True, exist_ok=True)

    print(f"[handler] process_audio_cloud start, speed={speed}, output_format={output_format}", flush=True)
    from audio_separator.separator import Separator

    separator = Separator(
        output_dir=str(output_dir_abs),
        output_format="wav",
        log_level=20,  # INFO
        model_file_dir=MODEL_CACHE_DIR,
        chunk_duration=CHUNK_DURATION,
        demucs_params={
            "segment_size": "Default",
            "shifts": shifts,
            "overlap": 0.25,
            "segments_enabled": True
        }
    )

    print(f"[handler] loading model {MODEL_FILENAME} into {MODEL_CACHE_DIR} ... (this is the slow step on first run per worker if not cached on volume)", flush=True)
    try:
        separator.load_model(model_filename=MODEL_FILENAME)
        print("[handler] model loaded successfully", flush=True)
    except Exception as load_err:
        print(f"[handler] FATAL during load_model: {load_err}", flush=True)
        traceback.print_exc()
        raise

    # Run separation
    print("[handler] starting separation...", flush=True)
    stems = separator.separate(str(input_path))
    print("[handler] separation complete", flush=True)

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

    base_name = input_path.stem
    if output_format == "MP3":
        ext = "mp3"
    elif output_format == "FLAC":
        ext = "flac"
    else:
        ext = "wav"

    mixed, sample_rate = _mix_stem_paths(abs_stem_paths, bass_only=False)
    output_path = work_dir / f"{base_name}_no_bass.{ext}"
    _write_mixed_output(
        mixed, sample_rate, output_path, output_format, work_dir, f"{base_name}_no_bass_temp.wav"
    )

    bass_path = None
    if save_bass_track:
        bass_mixed, bass_sr = _mix_stem_paths(abs_stem_paths, bass_only=True)
        bass_path = work_dir / f"{base_name}_bass.{ext}"
        _write_mixed_output(
            bass_mixed, bass_sr, bass_path, output_format, work_dir, f"{base_name}_bass_temp.wav"
        )

    # Cleanup separated stems
    shutil.rmtree(output_dir_abs, ignore_errors=True)

    return output_path, bass_path


# ================== RUNPOD HANDLER ==================
def handler(job):
    """
    Main entry point for RunPod Serverless.
    Any uncaught exception here will typically cause the worker to be marked unhealthy.
    We try very hard to return a proper {"error": "..."} dict so the job fails cleanly
    and the desktop app can fall back, without killing the worker process.
    """
    job_id = job.get("id", "unknown")
    print(f"[handler] received job {job_id}", flush=True)
    try:
        job_input = job.get("input", {})

        audio_b64 = job_input.get("audio_base64")
        filename = job_input.get("filename", "input.mp3")
        output_format = job_input.get("output_format", "MP3")
        speed = job_input.get("speed", "Quality")
        save_bass_track = bool(job_input.get("save_bass_track", False))

        if not audio_b64:
            print("[handler] ERROR: Missing audio_base64", flush=True)
            return {"error": "Missing audio_base64 in input"}

        with tempfile.TemporaryDirectory() as tmpdir:
            work_dir = Path(tmpdir)

            # Save input
            input_path = work_dir / filename
            input_path.write_bytes(base64.b64decode(audio_b64))
            print(f"[handler] input saved ({len(audio_b64)} base64 chars) to {input_path}", flush=True)

            # Process
            output_path, bass_path = process_audio_cloud(
                input_path, output_format, speed, work_dir, save_bass_track
            )

            result = build_cloud_response_payload(output_path, work_dir, bass_path)
            print(f"[handler] SUCCESS, returning {result['filename']}", flush=True)
            return result

    except Exception as e:
        # Log the full traceback — this is what you will see in the RunPod worker logs.
        err_msg = f"{type(e).__name__}: {e}"
        print(f"[handler] JOB {job_id} FAILED: {err_msg}", flush=True)
        traceback.print_exc()
        # Always return error dict so RunPod marks the *job* failed (not the whole worker)
        # and the desktop app can fall back to local.
        return {"error": err_msg}


print("[handler] importing runpod SDK...", flush=True)
try:
    import runpod
except Exception as import_err:
    print(f"[handler] FATAL: runpod import failed: {import_err}", flush=True)
    traceback.print_exc()
    sys.exit(1)

print("[handler] starting runpod.serverless (this process must stay alive)...", flush=True)
try:
    runpod.serverless.start({"handler": handler})
except Exception as start_err:
    print(f"[handler] FATAL: runpod.serverless.start failed: {start_err}", flush=True)
    traceback.print_exc()
    sys.exit(1)
