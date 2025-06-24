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
# Nunchaku는 4-bit 양자화 모델을 사용합니다.
nunchaku_repo_id = "mit-han-lab/nunchaku-flux.1-dev"
nunchaku_filename = "svdq-int4_r32-flux.1-dev.safetensors"
nunchaku_transformer_path = f"weights/nunchaku_transformer/{nunchaku_filename}"
DTYPE = torch.bfloat16 # Nunchaku 예제에서는 bfloat16을 사용합니다.

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

    # --- 2. Nunchaku 양자화 트랜스포머 다운로드 ---
    nunchaku_dir = os.path.dirname(nunchaku_transformer_path)
    if not os.path.exists(nunchaku_transformer_path):
        logging.info(f"Nunchaku transformer not found. Downloading from {nunchaku_repo_id}...")
        os.makedirs(nunchaku_dir, exist_ok=True)
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face Token not found in environment variables.")
        login(token=hf_token)
        snapshot_download(repo_id=nunchaku_repo_id, allow_patterns=[nunchaku_filename], local_dir=nunchaku_dir, local_dir_use_symlinks=False)
        # HuggingFace 라이브러리는 파일을 전체 경로에 저장하므로, 파일 위치를 직접 사용합니다.
        # os.rename은 필요 없습니다.
        logging.info("Nunchaku transformer downloaded successfully.")
    else:
        logging.info("Nunchaku transformer already exists, skipping download.")

    # --- 3. Nunchaku를 사용하여 파이프라인 구성 ---
    logging.info("Loading Nunchaku transformer...")
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(os.path.join(nunchaku_dir, nunchaku_filename))

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
