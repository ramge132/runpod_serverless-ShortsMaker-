import runpod
import torch
from diffusers import FluxPipeline
import os
from io import BytesIO
import traceback
import base64
from huggingface_hub import snapshot_download, login

# --- 초기화 (워커 시작 시 1회 실행) ---
pipe = None
initialization_error = None

print("Worker starting up...")

try:
    # --- 모델 다운로드 (실행 시점) ---
    model_path = "weights/flux1_dev"
    
    # 모델이 이미 다운로드되었는지 확인. 없다면 다운로드 실행.
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Downloading...")
        
        # 1. RunPod 환경 변수에서 Hugging Face 토큰을 읽어옵니다.
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face Token not found in environment variables. Please set HF_TOKEN.")
        
        print("Logging in to Hugging Face Hub...")
        login(token=hf_token)
        print("Login successful.")

        # 2. 모델 다운로드
        HF_REPO_ID = "black-forest-labs/FLUX.1-dev"
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=model_path,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md","*.git*","*.png","*.jpg"]
        )
        print("Model downloaded successfully.")
    else:
        print("Model already exists, skipping download.")

    # --- 파이프라인 로드 ---
    print("Loading FLUX model pipeline...")
    pipe = FluxPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16
    )
    pipe = pipe.to("cuda")
    print("Pipeline loaded successfully.")

except Exception as e:
    initialization_error = f"Initialization failed: {e}\n{traceback.format_exc()}"
    print(initialization_error)

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
    """RunPod 서버리스 핸들러 함수"""
    if initialization_error:
        return {"error": initialization_error}
    
    if not pipe:
        return {"error": "Worker is not properly initialized. Check the logs for details."}

    try:
        job_input = job['input']
        
        prompt = create_prompt(job_input)
        print(f"Generated Prompt: {prompt}")
        
        generator = torch.Generator(device="cuda").manual_seed(job_input.get('seed', 42))
        image = pipe(
            prompt=prompt, 
            num_inference_steps=28, 
            guidance_scale=7.0,
            generator=generator
        ).images[0]
        
        print("Encoding image to Base64...")
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        base64_encoded_image = base64.b64encode(img_bytes).decode('utf-8')
        
        print("Image encoded successfully.")
        
        return {
            "image_base64": base64_encoded_image,
            "image_prompt": prompt
        }

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error during handling job: {e}\n{error_trace}")
        return {"error": f"Job failed: {e}", "trace": error_trace}

# RunPod 핸들러 시작
runpod.serverless.start({"handler": handler})
