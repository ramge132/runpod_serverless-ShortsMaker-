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

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 전역 변수 및 초기화 ---
initialization_error = None
pipe = None
base_model_path = "weights/flux1_dev"
# INT4 버전의 Nunchaku 리포지토리로 최종 수정 (하드웨어 호환성)
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
    # 다운로드된 Nunchaku 리포지토리 디렉토리에서 트랜스포머를 로드
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
        
        logging.info("Encoding image to Base64...")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')
        
        logging.info("Image encoded successfully.")
        
        return {
            "image_base64": base64_encoded_image,
            "image_prompt": prompt
        }

    except Exception as e:
        logging.error(f"Error during handling job: {e}", exc_info=True)
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": f"Job failed: {e}", "trace": traceback.format_exc()}

# RunPod 핸들러 시작
runpod.serverless.start({"handler": handler})
