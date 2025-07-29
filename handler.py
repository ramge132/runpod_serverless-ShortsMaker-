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
import uuid
import boto3
from botocore.exceptions import NoCredentialsError
from openai import OpenAI

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- OpenAI 클라이언트 초기화 ---
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- 전역 변수 및 초기화 ---
initialization_error = None
pipe = None
base_model_path = "weights/flux1_dev"
nunchaku_repo_id = "mit-han-lab/svdq-int4-flux.1-dev" 
nunchaku_model_path = f"weights/{nunchaku_repo_id.replace('/', '_')}"
DTYPE = torch.bfloat16

# S3 관련 설정
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION')

logging.info("Worker starting up...")
try:
    if not S3_BUCKET_NAME or not AWS_REGION:
        raise ValueError("S3_BUCKET_NAME and AWS_REGION environment variables must be set.")

    # --- 모델 다운로드 및 파이프라인 구성 ---
    if not os.path.exists(base_model_path):
        logging.info(f"Base model not found. Downloading from black-forest-labs/FLUX.1-dev...")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token: raise ValueError("HF_TOKEN not found.")
        login(token=hf_token)
        snapshot_download(repo_id="black-forest-labs/FLUX.1-dev", local_dir=base_model_path, local_dir_use_symlinks=False, ignore_patterns=["*.md", "*.git*", "*.png", "*.jpg", "transformer/*"])
        logging.info("Base model components downloaded.")
    else:
        logging.info("Base model already exists.")

    if not os.path.exists(nunchaku_model_path):
        logging.info(f"Nunchaku model repo not found. Downloading from {nunchaku_repo_id}...")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token: raise ValueError("HF_TOKEN not found.")
        login(token=hf_token)
        snapshot_download(repo_id=nunchaku_repo_id, local_dir=nunchaku_model_path, local_dir_use_symlinks=False)
        logging.info("Nunchaku model repo downloaded.")
    else:
        logging.info("Nunchaku model repo already exists.")

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

def translate_to_english(text):
    """주어진 텍스트를 영어로 번역합니다."""
    if not text:
        return ""
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates Korean text to English."},
                {"role": "user", "content": f"Translate the following Korean text to English: {text}"}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error translating text: {e}")
        return text # 번역 실패 시 원본 텍스트 반환

def create_prompts(input_data):
    """입력 데이터로부터 Positive 프롬프트를 생성합니다."""
    # Backend에서 생성된 프롬프트를 직접 사용
    positive_prompt = input_data.get('image_prompt', '')
    
    # 품질 관련 태그는 유지
    positive_quality_tags = "masterpiece, best quality, ultra-detailed, 8k, photorealistic, cinematic lighting"

    # 만약 Backend에서 프롬프트를 전달하지 않았을 경우, 기존 방식으로 프롬프트 생성 (하위 호환성)
    if not positive_prompt:
        metadata = input_data.get('story_metadata', {})
        audios = input_data.get('audios', [])
        
        characters = metadata.get('characters', [])
        gender_map = {0: "man", 1: "woman"}
        character_descriptions = []
        for char in characters:
            name = char.get('name', 'person')
            gender = gender_map.get(char.get('gender'), "")
            desc = translate_to_english(char.get('description', '')) # 영어로 번역
            character_descriptions.append(f"{name} as a {gender} ({desc})")
        
        scene_descriptions = []
        for audio in audios:
            text = audio.get('text', '')
            translated_text = translate_to_english(text) # 영어로 번역
            if audio.get('type') == 'dialogue':
                char_name = audio.get('character', '')
                emotion = audio.get('emotion', '')
                scene_descriptions.append(f"{char_name} is speaking with a {emotion} expression, saying '{translated_text}'")
            else: # narration
                scene_descriptions.append(translated_text)

        positive_prompt = f"a scene of ({', '.join(character_descriptions)}). {' '.join(scene_descriptions)}"

    final_positive_prompt = f"{positive_quality_tags}, {positive_prompt}"
    
    return final_positive_prompt

def handler(job):
    if initialization_error:
        return {"error": initialization_error}
    
    if not pipe:
        return {"error": "Worker is not properly initialized."}

    try:
        job_input = job['input']
        
        # 개선된 프롬프트 생성 함수 호출
        positive_prompt = create_prompts(job_input)
        logging.info(f"--- Final Positive Prompt for Image Generation ---\n{positive_prompt}")
        
        with torch.no_grad():
            generator = torch.Generator("cuda").manual_seed(job_input.get('seed', 42))
            image = pipe(
                prompt=positive_prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                max_sequence_length=512,
                generator=generator
            ).images[0]
        
        logging.info("Image generated. Uploading to S3...")
        
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        file_key = f"images/{uuid.uuid4()}.png"
        
        s3_client.upload_fileobj(
            buffer,
            S3_BUCKET_NAME,
            file_key,
            ExtraArgs={'ContentType': 'image/png'}
        )
        
        image_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{file_key}"
        
        logging.info(f"Image uploaded successfully: {image_url}")
        
        return {
            "image_url": image_url,
            "image_prompt": positive_prompt
        }

    except NoCredentialsError:
        error_msg = "S3 credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
        logging.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        logging.error(f"Error during handling job: {e}", exc_info=True)
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": f"Job failed: {e}", "trace": traceback.format_exc()}

# RunPod 핸들러 시작
runpod.serverless.start({"handler": handler})


# ----------------------------------------------------------------------- #
# 백업 코드 (기존)

# import runpod
# import torch
# from diffusers import FluxPipeline
# from nunchaku import NunchakuFluxTransformer2dModel
# import os
# import io
# import traceback
# import base64
# from huggingface_hub import snapshot_download, login
# import logging
# import gc
# import uuid
# import boto3
# from botocore.exceptions import NoCredentialsError

# # --- 로깅 설정 ---
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# # --- 전역 변수 및 초기화 ---
# initialization_error = None
# pipe = None
# base_model_path = "weights/flux1_dev"
# nunchaku_repo_id = "mit-han-lab/svdq-int4-flux.1-dev" 
# nunchaku_model_path = f"weights/{nunchaku_repo_id.replace('/', '_')}"
# DTYPE = torch.bfloat16

# # S3 관련 설정
# S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
# AWS_REGION = os.environ.get('AWS_REGION')

# logging.info("Worker starting up...")
# try:
#     if not S3_BUCKET_NAME or not AWS_REGION:
#         raise ValueError("S3_BUCKET_NAME and AWS_REGION environment variables must be set.")

#     # --- 모델 다운로드 및 파이프라인 구성 ---
#     if not os.path.exists(base_model_path):
#         logging.info(f"Base model not found. Downloading from black-forest-labs/FLUX.1-dev...")
#         hf_token = os.environ.get("HF_TOKEN")
#         if not hf_token: raise ValueError("HF_TOKEN not found.")
#         login(token=hf_token)
#         snapshot_download(repo_id="black-forest-labs/FLUX.1-dev", local_dir=base_model_path, local_dir_use_symlinks=False, ignore_patterns=["*.md", "*.git*", "*.png", "*.jpg", "transformer/*"])
#         logging.info("Base model components downloaded.")
#     else:
#         logging.info("Base model already exists.")

#     if not os.path.exists(nunchaku_model_path):
#         logging.info(f"Nunchaku model repo not found. Downloading from {nunchaku_repo_id}...")
#         hf_token = os.environ.get("HF_TOKEN")
#         if not hf_token: raise ValueError("HF_TOKEN not found.")
#         login(token=hf_token)
#         snapshot_download(repo_id=nunchaku_repo_id, local_dir=nunchaku_model_path, local_dir_use_symlinks=False)
#         logging.info("Nunchaku model repo downloaded.")
#     else:
#         logging.info("Nunchaku model repo already exists.")

#     logging.info("Loading Nunchaku transformer...")
#     transformer = NunchakuFluxTransformer2dModel.from_pretrained(nunchaku_model_path)

#     logging.info("Loading base pipeline and injecting Nunchaku transformer...")
#     pipe = FluxPipeline.from_pretrained(
#         base_model_path,
#         transformer=transformer,
#         torch_dtype=DTYPE
#     ).to("cuda")
    
#     logging.info("Pipeline with Nunchaku loaded successfully.")

# except Exception as e:
#     initialization_error = f"Initialization failed: {e}"
#     logging.error(initialization_error, exc_info=True)

# # ------------------------------------

# def create_prompts(input_data):
#     """입력 데이터로부터 Positive 및 Negative 프롬프트를 생성합니다."""
#     metadata = input_data.get('story_metadata', {})
#     audios = input_data.get('audios', [])
    
#     # --- 품질 및 스타일 키워드 ---
#     positive_quality_tags = "masterpiece, best quality, ultra-detailed, 8k, photorealistic, cinematic lighting"
#     negative_quality_tags = "ugly, deformed, noisy, blurry, distorted, low quality, bad anatomy, worst quality, watermark, text, signature"

#     # --- 캐릭터 정보 파싱 ---
#     characters = metadata.get('characters', [])
#     gender_map = {0: "man", 1: "woman"}
#     character_descriptions = []
#     for char in characters:
#         name = char.get('name', 'person')
#         gender = gender_map.get(char.get('gender'), "")
#         desc = char.get('description', '')
#         character_descriptions.append(f"{name} as a {gender} ({desc})")
    
#     # --- 씬 내용 파싱 ---
#     scene_descriptions = []
#     for audio in audios:
#         text = audio.get('text', '')
#         if audio.get('type') == 'dialogue':
#             char_name = audio.get('character', '')
#             emotion = audio.get('emotion', '')
#             scene_descriptions.append(f"{char_name} is speaking with a {emotion} expression, saying '{text}'")
#         else: # narration
#             scene_descriptions.append(text)

#     # --- 최종 프롬프트 조합 ---
#     final_positive_prompt = f"{positive_quality_tags}, a scene of ({', '.join(character_descriptions)}). {' '.join(scene_descriptions)}"
    
#     return final_positive_prompt, negative_quality_tags

# def handler(job):
#     if initialization_error:
#         return {"error": initialization_error}
    
#     if not pipe:
#         return {"error": "Worker is not properly initialized."}

#     try:
#         job_input = job['input']
        
#         # 개선된 프롬프트 생성 함수 호출
#         positive_prompt, negative_prompt = create_prompts(job_input)
#         logging.info(f"Positive Prompt: {positive_prompt}")
#         logging.info(f"Negative Prompt: {negative_prompt}")
        
#         with torch.no_grad():
#             generator = torch.Generator("cuda").manual_seed(job_input.get('seed', 42))
#             image = pipe(
#                 prompt=positive_prompt, 
#                 prompt_2=positive_prompt,
#                 negative_prompt=negative_prompt,
#                 negative_prompt_2=negative_prompt,
#                 num_inference_steps=28,
#                 guidance_scale=7.0,
#                 generator=generator
#             ).images[0]
        
#         logging.info("Image generated. Uploading to S3...")
        
#         buffer = io.BytesIO()
#         image.save(buffer, format="PNG")
#         buffer.seek(0)
        
#         s3_client = boto3.client('s3', region_name=AWS_REGION)
#         file_key = f"images/{uuid.uuid4()}.png"
        
#         s3_client.upload_fileobj(
#             buffer,
#             S3_BUCKET_NAME,
#             file_key,
#             ExtraArgs={'ContentType': 'image/png'}
#         )
        
#         image_url = f"https://{S3_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{file_key}"
        
#         logging.info(f"Image uploaded successfully: {image_url}")
        
#         return {
#             "image_url": image_url,
#             "image_prompt": positive_prompt
#         }

#     except NoCredentialsError:
#         error_msg = "S3 credentials not found. Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY."
#         logging.error(error_msg)
#         return {"error": error_msg}
#     except Exception as e:
#         logging.error(f"Error during handling job: {e}", exc_info=True)
#         gc.collect()
#         torch.cuda.empty_cache()
#         return {"error": f"Job failed: {e}", "trace": traceback.format_exc()}

# # RunPod 핸들러 시작
# runpod.serverless.start({"handler": handler})
