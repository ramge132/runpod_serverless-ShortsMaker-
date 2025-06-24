# import runpod
# import torch
# from diffusers import FluxPipeline
# import os
# import boto3
# from botocore.exceptions import NoCredentialsError
# from io import BytesIO

# # --- 초기화 (워커 시작 시 1회 실행) ---
# # 모델 로드
# # download.py에서 받은 모델 경로를 정확히 지정해야 합니다.
# model_path = "weights" # snapshot_download가 이 디렉토리에 모델을 저장합니다.

# # FLUX 파이프라인은 from_pretrained를 사용하여 디렉토리에서 직접 로드합니다.
# pipe = FluxPipeline.from_pretrained(
#     model_path, 
#     torch_dtype=torch.bfloat16 # FLUX 모델은 bfloat16에 최적화되어 있습니다.
# )
# pipe = pipe.to("cuda")

# # S3 클라이언트 설정 (결과 업로드용)
# # RunPod 엔드포인트의 환경 변수에서 설정된 값을 사용합니다.
# s3_client = boto3.client(
#     's3',
#     aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
#     aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
#     region_name=os.environ.get('AWS_REGION')
# )
# S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
# # ------------------------------------

# def create_prompt(input_data):
#     """입력 데이터로부터 이미지 생성 프롬프트를 생성합니다."""
#     metadata = input_data.get('story_metadata', {})
#     audios = input_data.get('audios', [])
    
#     title = metadata.get('title', 'A story')
#     characters = metadata.get('characters', [])
    
#     # 캐릭터 설명 조합
#     char_descriptions = ", ".join([f"{c['name']} ({c['description']})" for c in characters])
    
#     # 현재 씬의 텍스트 조합
#     scene_text = " ".join([a['text'] for a in audios])
    
#     # 최종 프롬프트
#     # 이 부분을 고도화하여 이미지 품질을 높일 수 있습니다.
#     prompt = f"A scene from '{title}'. {char_descriptions}. The scene depicts: {scene_text}. cinematic, high detail, photorealistic, 8k"
    
#     return prompt

# def handler(job):
#     """RunPod 서버리스 핸들러 함수"""
#     try:
#         job_input = job['input']
        
#         # 1. 프롬프트 생성
#         prompt = create_prompt(job_input)
#         print(f"Generated Prompt: {prompt}")
        
#         # 2. 이미지 생성 (FLUX 파이프라인 사용)
#         # generator를 사용하여 재현 가능한 결과를 얻을 수 있습니다.
#         generator = torch.Generator(device="cuda").manual_seed(job_input.get('seed', 42))
#         image = pipe(
#             prompt=prompt, 
#             num_inference_steps=28, 
#             guidance_scale=7.0,
#             generator=generator
#         ).images[0]
        
#         # 3. 생성된 이미지를 S3에 업로드
#         buffer = BytesIO()
#         image.save(buffer, format="PNG")
#         buffer.seek(0)
        
#         # 파일명 생성 (storyId와 sceneId 사용)
#         story_id = job_input['story_metadata']['story_id']
#         scene_id = job_input['scene_id']
#         s3_key = f"generated_images/{story_id}/{scene_id}.png"
        
#         s3_client.upload_fileobj(
#             buffer,
#             S3_BUCKET_NAME,
#             s3_key,
#             ExtraArgs={'ContentType': 'image/png'}
#         )
        
#         # 4. S3 URL 생성 및 반환
#         image_url = f"https://{S3_BUCKET_NAME}.s3.{os.environ.get('AWS_REGION')}.amazonaws.com/{s3_key}"
        
#         return {
#             "image_url": image_url,
#             "image_prompt": prompt # Java 백엔드에서 저장할 수 있도록 프롬프트도 반환
#         }

#     except Exception as e:
#         print(f"Error: {e}")
#         return {"error": str(e)}

# # RunPod 핸들러 시작
# runpod.serverless.start({"handler": handler})
import runpod
import torch
from diffusers import FluxPipeline
import os
import io
import traceback
import base64
from huggingface_hub import snapshot_download, login
import logging
import gc
import nunchaku # Nunchaku 라이브러리 임포트

# --- 메모리 단편화 방지 (스크립트 최상단) ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 전역 변수 및 초기화 ---
initialization_error = None
pipe = None
model_path = "weights/flux1_dev"
DTYPE = torch.float16

logging.info("Worker starting up...")
try:
    # --- 1. 모델 다운로드 ---
    if not os.path.exists(model_path):
        logging.info(f"Model not found at {model_path}. Downloading...")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("Hugging Face Token not found in environment variables. Please set HF_TOKEN.")
        
        logging.info("Logging in to Hugging Face Hub...")
        login(token=hf_token)
        logging.info("Login successful.")

        HF_REPO_ID = "black-forest-labs/FLUX.1-dev"
        snapshot_download(repo_id=HF_REPO_ID, local_dir=model_path, local_dir_use_symlinks=False, ignore_patterns=["*.md","*.git*","*.png","*.jpg"])
        logging.info("Model downloaded successfully.")
    else:
        logging.info("Model already exists, skipping download.")

    # --- 2. Nunchaku를 사용하여 파이프라인 로드 및 래핑 ---
    logging.info("Loading FLUX model pipeline for Nunchaku...")
    
    # 파이프라인을 CPU에 float16으로 로드
    pipe = FluxPipeline.from_pretrained(
        model_path, 
        torch_dtype=DTYPE
    )
    
    # Nunchaku로 파이프라인을 래핑하여 자동 메모리 관리 활성화
    logging.info("Wrapping pipeline with Nunchaku...")
    pipe = nunchaku.wrap(pipe)
    
    logging.info("Pipeline loaded and wrapped with Nunchaku successfully.")

except Exception as e:
    initialization_error = f"Initialization failed: {e}"
    logging.error(initialization_error, exc_info=True)

# ------------------------------------

def create_prompt(input_data):
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
        
        # Nunchaku가 래핑한 파이프라인을 직접 호출
        # 내부적으로 필요한 모듈만 GPU로 이동시켜 실행됨
        with torch.no_grad():
            generator = torch.Generator().manual_seed(job_input.get('seed', 42))
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
        # 에러 발생 후 메모리 정리
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": f"Job failed: {e}", "trace": traceback.format_exc()}

# RunPod 핸들러 시작
runpod.serverless.start({"handler": handler})
