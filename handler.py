import runpod
import torch
from diffusers import FluxPipeline
from nunchaku import NunchakuFluxTransformer2dModel
import os
import io
import traceback
import base64
from huggingface_hub import snapshot_download, login
import logging
import gc
import subprocess
import uuid

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 전역 변수 및 초기화 ---
initialization_error = None
pipe = None
base_model_path = "weights/flux1_dev"
nunchaku_repo_id = "mit-han-lab/svdq-int4-flux.1-dev" 
nunchaku_model_path = f"weights/{nunchaku_repo_id.replace('/', '_')}"
DTYPE = torch.bfloat16

logging.info("Worker starting up...")
try:
    # --- 1. 베이스 모델(Text Encoders, VAE 등) 다운로드 ---
    if not os.path.exists(base_model_path):
        logging.info(f"Base model not found. Downloading from black-forest-labs/FLUX.1-dev...")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face Token not found in environment variables.")
        login(token=hf_token)
        snapshot_download(repo_id="black-forest-labs/FLUX.1-dev", local_dir=base_model_path, local_dir_use_symlinks=False, ignore_patterns=["*.md", "*.git*", "*.png", "*.jpg", "transformer/*"])
        logging.info("Base model components downloaded successfully.")
    else:
        logging.info("Base model already exists, skipping download.")

    # --- 2. Nunchaku 트랜스포머 리포지토리 전체 다운로드 ---
    if not os.path.exists(nunchaku_model_path):
        logging.info(f"Nunchaku model repo not found. Downloading from {nunchaku_repo_id}...")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face Token not found in environment variables.")
        login(token=hf_token)
        snapshot_download(repo_id=nunchaku_repo_id, local_dir=nunchaku_model_path, local_dir_use_symlinks=False)
        logging.info("Nunchaku model repo downloaded successfully.")
    else:
        logging.info("Nunchaku model repo already exists, skipping download.")

    # --- 3. Nunchaku를 사용하여 파이프라인 구성 ---
    logging.info("Loading Nunchaku transformer...")
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(nunchaku_model_path)

    logging.info("Loading base pipeline and injecting Nunchaku transformer...")
    pipe = FluxPipeline.from_pretrained(
        base_model_path,
        transformer=transformer,
        torch_dtype=DTYPE
    ).to("cuda")
    
    logging.info("Pipeline with Nunchaku loaded successfully.")

except Exception as e:
    initialization_error = f"Initialization failed: {e}"
    logging.error(initialization_error, exc_info=True)

# ------------------------------------

def create_prompt(input_data):
    """입력 데이터로부터 이미지 생성 프롬프트를 생성합니다."""
    metadata = input_data.get('story_metadata', {})
    audios = input_data.get('audios', [])
    title = metadata.get('title', 'A story')
    characters = metadata.get('characters', [])
    char_descriptions = ", ".join([f"{c['name']} ({c['description']})" for c in characters])
    scene_text = " ".join([a['text'] for a in audios])
    prompt = f"A scene from '{title}'. {char_descriptions}. The scene depicts: {scene_text}. cinematic, high detail, photorealistic, 8k"
    return prompt

def handler(job):
    if initialization_error:
        return {"error": initialization_error}
    
    if not pipe:
        return {"error": "Worker is not properly initialized. Check the logs for details."}

    try:
        job_input = job['input']
        prompt = create_prompt(job_input)
        logging.info(f"Generated Prompt: {prompt}")
        
        with torch.no_grad():
            generator = torch.Generator("cuda").manual_seed(job_input.get('seed', 42))
            image = pipe(
                prompt=prompt, 
                prompt_2=prompt,
                num_inference_steps=28,
                guidance_scale=7.0,
                generator=generator
            ).images[0]
        
        logging.info("Image generated. Uploading to transfer.sh...")
        
        # 이미지를 임시 파일로 저장
        temp_filename = f"/tmp/{uuid.uuid4()}.png"
        image.save(temp_filename)
        
        # curl을 사용하여 transfer.sh에 업로드
        upload_command = ["curl", "--upload-file", temp_filename, f"https://transfer.sh/{os.path.basename(temp_filename)}"]
        result = subprocess.run(upload_command, capture_output=True, text=True)
        
        # 임시 파일 삭제
        os.remove(temp_filename)
        
        if result.returncode == 0:
            image_url = result.stdout.strip()
            logging.info(f"Image uploaded successfully: {image_url}")
            return {
                "image_url": image_url,
                "image_prompt": prompt
            }
        else:
            logging.error(f"Failed to upload image: {result.stderr}")
            return {"error": "Failed to upload image to transfer.sh", "details": result.stderr}

    except Exception as e:
        logging.error(f"Error during handling job: {e}", exc_info=True)
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": f"Job failed: {e}", "trace": traceback.format_exc()}

# RunPod 핸들러 시작
runpod.serverless.start({"handler": handler})
