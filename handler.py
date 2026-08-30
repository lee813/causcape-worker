import os
import sys
from pathlib import Path

# Add LongCat repository to python path
sys.path.insert(0, "/app/LongCat-Video")

import runpod
from longcat_worker import AvatarWorker

CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")
MODEL_ID = os.getenv("CACHED_MODEL_ID", "frank/longcat-avatar-runpod")

def resolve_cached_model(model_id: str) -> Path:
    """Resolve RunPod Cached Model snapshot directory dynamically."""
    if "/" in model_id:
        org, name = model_id.split("/", 1)
        model_dir = CACHE_ROOT / f"models--{org}--{name}"
    else:
        model_dir = CACHE_ROOT / f"models--{model_id}"
        
    ref_file = model_dir / "refs" / "main"
    if ref_file.exists():
        revision = ref_file.read_text().strip()
        snapshot_dir = model_dir / "snapshots" / revision
        if snapshot_dir.exists():
            return snapshot_dir

    snapshots = list((model_dir / "snapshots").glob("*"))
    if snapshots:
        return snapshots[0]

    # Fallback to local weights directory for local development or testing
    local_fallback = Path("/workspace/LongCat-Video/weights")
    if local_fallback.exists():
        return local_fallback

    raise FileNotFoundError(f"Cached model not found in {model_dir} and local fallback not present")

# Resolve model paths
MODEL_ROOT = resolve_cached_model(MODEL_ID)
BASE_MODEL_DIR = MODEL_ROOT / "LongCat-Video"
AVATAR_MODEL_DIR = MODEL_ROOT / "LongCat-Video-Avatar-1.5"

# Global Singleton initialization on Worker Cold Start (Executes once per worker)
print(f"=== INITIALIZING LONGCAT WORKER (Model Root: {MODEL_ROOT}) ===")
worker = AvatarWorker(base_model_dir=str(BASE_MODEL_DIR), avatar_model_dir=str(AVATAR_MODEL_DIR))
print("=== LONGCAT WORKER READY ===")

def handler(job):
    job_input = job.get("input", {})
    image_url = job_input.get("image_url")
    audio_url = job_input.get("audio_url")
    prompt = job_input.get("prompt", "A Chinese male presenter speaking calmly and naturally in an explainer style.")
    resolution = job_input.get("resolution", "480p")
    audio_guidance_scale = float(job_input.get("audio_guidance_scale", 2.0))
    clean_tts = bool(job_input.get("clean_tts", True))
    num_inference_steps = int(job_input.get("num_inference_steps", 8))
    num_segments = int(job_input.get("num_segments", 1))

    if not image_url or not audio_url:
        return {"error": "Missing required 'image_url' or 'audio_url' in job input"}

    try:
        video_url = worker.generate(
            job_id=job["id"],
            image_url=image_url,
            audio_url=audio_url,
            prompt=prompt,
            resolution=resolution,
            audio_guidance_scale=audio_guidance_scale,
            clean_tts=clean_tts,
            num_inference_steps=num_inference_steps,
            num_segments=num_segments
        )
        return {
            "status": "success",
            "video_url": video_url,
            "job_id": job["id"]
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": str(e),
            "job_id": job["id"]
        }

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
