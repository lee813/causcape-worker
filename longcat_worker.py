import os
import gc
import uuid
import torch
import boto3
import requests
from pathlib import Path
import PIL.Image

# Enable expandable segments to avoid CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from longcat_video.pipeline_longcat_video_avatar import LongCatVideoAvatarPipeline


class AvatarWorker:
    def __init__(self, base_model_dir: str, avatar_model_dir: str):
        self.base_model_dir = base_model_dir
        self.avatar_model_dir = avatar_model_dir
        
        # S3 / Cloudflare R2 object storage client
        self.s3_endpoint = os.getenv("S3_ENDPOINT_URL")
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.public_cdn_url = os.getenv("CDN_BASE_URL", "").rstrip("/")
        
        if self.s3_endpoint and self.bucket_name:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.s3_endpoint,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
        else:
            self.s3_client = None

        print("Building LongCat Avatar Pipeline (INT8 + 8-Step Distilled)...")
        self.pipeline = LongCatVideoAvatarPipeline.from_pretrained(
            self.base_model_dir,
            checkpoint_dir=self.avatar_model_dir,
            model_type="avatar-v1.5",
            use_int8=True,
            use_distill=True,
            torch_dtype=torch.bfloat16,
            device="cuda"
        )
        self.pipeline.enable_model_cpu_offload()
        print("LongCat Pipeline Loaded and Ready in GPU Memory.")

    def download_asset(self, url: str, target_path: str):
        resp = requests.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(target_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=16384):
                f.write(chunk)

    def upload_output(self, local_path: str, job_id: str) -> str:
        object_key = f"causcape-outputs/{job_id}.mp4"
        
        if self.s3_client and self.bucket_name:
            self.s3_client.upload_file(
                local_path,
                self.bucket_name,
                object_key,
                ExtraArgs={"ContentType": "video/mp4"}
            )
            if self.public_cdn_url:
                return f"{self.public_cdn_url}/{object_key}"
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": object_key},
                ExpiresIn=604800  # 7 days
            )
        else:
            # Fallback to lightweight temporary hosting if S3 is not configured
            import subprocess
            try:
                out = subprocess.check_output([
                    "curl", "-s", "-F", "files[]=@" + local_path, "https://uguu.se/upload.php"
                ], timeout=30).decode("utf-8")
                import json
                data = json.loads(out)
                if data.get("success") and data.get("files"):
                    return data["files"][0]["url"]
            except Exception as e:
                print(f"Fallback upload error: {e}")
            
            # Local path fallback
            return local_path

    def generate(
        self,
        job_id: str,
        image_url: str,
        audio_url: str,
        prompt: str,
        resolution: str = "480p",
        audio_guidance_scale: float = 2.0,
        clean_tts: bool = True,
        num_inference_steps: int = 8,
        num_segments: int = 1
    ) -> str:
        job_dir = Path(f"/tmp/{job_id}")
        job_dir.mkdir(parents=True, exist_ok=True)

        anchor_img_path = str(job_dir / "anchor.png")
        input_audio_path = str(job_dir / "speech.wav")
        output_video_path = str(job_dir / f"{job_id}_final.mp4")

        self.download_asset(image_url, anchor_img_path)
        self.download_asset(audio_url, input_audio_path)

        # Execute generation via pipeline
        print(f"[{job_id}] Running LongCat Avatar Generation (Guidance Scale: {audio_guidance_scale}, Clean TTS: {clean_tts})...")
        
        # Call the pipeline generate function
        result_path = self.pipeline.generate_avatar(
            prompt=prompt,
            image_path=anchor_img_path,
            audio_path=input_audio_path,
            resolution=resolution,
            num_inference_steps=num_inference_steps,
            audio_guidance_scale=audio_guidance_scale,
            skip_vocal_separator=clean_tts,
            num_segments=num_segments,
            output_dir=str(job_dir)
        )

        final_file = result_path if (result_path and os.path.exists(result_path)) else output_video_path
        
        # Upload video to CDN / S3
        video_url = self.upload_output(final_file, job_id)
        print(f"[{job_id}] Video generated successfully: {video_url}")

        # Cleanup local scratch files & empty CUDA cache
        os.system(f"rm -rf {job_dir}")
        torch.cuda.empty_cache()
        gc.collect()

        return video_url
