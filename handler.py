import os
import sys
from pathlib import Path

# Add LongCat repository to python path
sys.path.insert(0, "/app/LongCat-Video")

import runpod

CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")

def resolve_model_directories() -> tuple[Path, Path]:
    """Dynamically resolve Base Model and Avatar Model directories from RunPod Cached Model."""
    print(f"[*] Scanning cache directory: {CACHE_ROOT}")
    
    # 1. Search for any cached models in RunPod cache volume
    if CACHE_ROOT.exists():
        for model_dir in CACHE_ROOT.glob("models--*"):
            print(f"  [+] Found cached repo: {model_dir.name}")
            snapshots = list((model_dir / "snapshots").glob("*"))
            if snapshots:
                snap_dir = snapshots[0]
                print(f"  [+] Active snapshot directory: {snap_dir}")
                
                # Check merged structure
                if (snap_dir / "LongCat-Video").exists() and (snap_dir / "LongCat-Video-Avatar-1.5").exists():
                    return snap_dir / "LongCat-Video", snap_dir / "LongCat-Video-Avatar-1.5"
                
                # Check standalone Avatar 1.5 structure
                if (snap_dir / "base_model_int8").exists() or (snap_dir / "lora").exists():
                    # Check if base model is sibling or inside
                    return snap_dir, snap_dir
                    
                return snap_dir, snap_dir

    # 2. Check local fallback
    fallback = Path("/workspace/LongCat-Video/weights")
    if fallback.exists():
        print(f"[*] Using local fallback directory: {fallback}")
        base = fallback / "LongCat-Video" if (fallback / "LongCat-Video").exists() else fallback
        avatar = fallback / "LongCat-Video-Avatar-1.5" if (fallback / "LongCat-Video-Avatar-1.5").exists() else fallback
        return base, avatar

    # 3. Default to huggingface model hub downloads if cache not present
    print("[!] Cached volume not mounted, falling back to Hugging Face Hub IDs")
    return Path("meituan-longcat/LongCat-Video"), Path("meituan-longcat/LongCat-Video-Avatar-1.5")

BASE_MODEL_DIR, AVATAR_MODEL_DIR = resolve_model_directories()
print(f"[*] Base Model Path: {BASE_MODEL_DIR}")
print(f"[*] Avatar Model Path: {AVATAR_MODEL_DIR}")

# Global Singleton initialization on Worker Cold Start
print("=== INITIALIZING LONGCAT WORKER ===")
from longcat_worker import AvatarWorker
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
