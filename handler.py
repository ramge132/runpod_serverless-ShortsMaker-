# handler.py:
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
import random # Python에서 랜덤 선택을 위해 필수

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- OpenAI 클라이언트 초기화 ---
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- 카메라 구도 와일드카드 리스트 (Python에서 사용) ---
CAMERA_ANGLES = [
    "model shoot style", "close-up shot", "medium shot", "full shot", "wide shot", 
    "eye-level shot", "low-angle shot", "high-angle shot", "from above", "from below",
    "dutch angle", "cinematic shot", "selfie", "candid photo"
]

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
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable must be set.")

    # --- 모델 다운로드 및 파이프라인 구성 ---
    if not os.path.exists(base_model_path):
        logging.info(f"Base model not found. Downloading...")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token: raise ValueError("HF_TOKEN not found.")
        login(token=hf_token)
        snapshot_download(repo_id="black-forest-labs/FLUX.1-dev", local_dir=base_model_path, local_dir_use_symlinks=False, ignore_patterns=["*.md", "*.git*", "*.png", "*.jpg", "transformer/*"])
    if not os.path.exists(nunchaku_model_path):
        logging.info(f"Nunchaku model repo not found. Downloading...")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token: raise ValueError("HF_TOKEN not found.")
        login(token=hf_token)
        snapshot_download(repo_id=nunchaku_repo_id, local_dir=nunchaku_model_path, local_dir_use_symlinks=False)
    
    transformer = NunchakuFluxTransformer2dModel.from_pretrained(nunchaku_model_path)
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

def generate_prompt_keywords(korean_context):
    """
    한국어 컨텍스트를 받아, 전문적인 스타일의 영어 프롬프트 키워드를 생성합니다.
    카메라 앵글은 이 함수에서 결정하지 않습니다.
    """
    if not korean_context:
        return ""
        
    logging.info(f"--- 프롬프트 키워드 생성을 위한 한국어 컨텍스트 ---\nInput: {korean_context}")
    
    system_prompt = f"""You are an expert prompt engineer for a photorealistic image AI. Your task is to convert a simple Korean context into a rich, detailed, comma-separated English prompt in the style of a professional photoshoot.

**Instructions:**
1.  **Analyze Subject**: Read the Korean text. If it contains character descriptions (e.g., '등장인물:'), focus the prompt on them. Use generic terms like 'a Korean girl', 'a man'. **DO NOT include character names.** If no character is present, create a symbolic or conceptual image.
2.  **Enrich the Prompt**: You MUST add stylistic keywords to make the prompt rich and detailed. Include terms related to:
    - **Quality & Style**: `masterpiece`, `high-quality`, `4k full HD photo`, `contest winner photo`
    - **Lighting**: `ecstasy of light and shadow`, `volumetric light and shadows`, `cinematic lighting`
    - **Artists for Style Blending**: `by lee jeffries`, `by greg rutkowski`, `by magali villanueve`
3.  **Output Format**: The output MUST be a single line of comma-separated English keywords. DO NOT add a camera angle; that will be handled separately.

**Example 1 (Character is present):**
Korean text: "장면: 내가 이걸 왜 해? 등장인물: 갈색 머리의 긴 생머리를 가진 한국 아이돌 느낌의 소녀"
English Keywords: masterpiece, 4k full HD photo of a Korean girl with long brown hair, idol, beautiful face, looking defiant, beautiful city in the background, ecstasy of light and shadow, volumetric light and shadows, contest winner photo by lee jeffries, greg rutkowski and magali villanueve

**Example 2 (Character is NOT present):**
Korean text: "장면: 님들 그거 앎?"
English Keywords: masterpiece, high-quality, a giant glowing question mark in the center of a dark, mysterious room, cinematic lighting, volumetric light and shadows, by greg rutkowski
"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Korean text: \"{korean_context}\""}
            ],
            temperature=0.4,
        )
        generated_keywords = response.choices[0].message.content.strip()
        logging.info(f"--- LLM이 생성한 프롬프트 키워드 ---\nOutput: {generated_keywords}")
        return generated_keywords
    except Exception as e:
        logging.error(f"Error during prompt keyword generation: {e}")
        return korean_context

def create_prompts(input_data):
    """입력 데이터로부터 최종 Positive 프롬프트를 생성합니다."""
    
    # 1. Python 코드가 직접 랜덤 카메라 구도를 선택
    random_camera_angle = random.choice(CAMERA_ANGLES)
    
    # 2. LLM에 전달할 한국어 컨텍스트 준비
    metadata = input_data.get('story_metadata', {})
    audios = input_data.get('audios', [])
    scene_text = audios[0].get('text', '') if audios else ''
    
    characters = metadata.get('characters', [])
    character_info_kr = ""
    if characters:
        char_descs = []
        for char in characters:
            desc = char.get('description', '사람') 
            char_descs.append(desc)
        character_info_kr = f"등장인물: {', '.join(char_descs)}"

    full_korean_context = f"장면: {scene_text}. {character_info_kr}".strip()
    
    # 3. LLM을 호출하여 핵심 프롬프트 키워드 생성
    prompt_keywords = generate_prompt_keywords(full_korean_context)

    # 4. 랜덤 카메라 구도와 생성된 키워드를 조합하여 최종 프롬프트 완성
    final_positive_prompt = f"{random_camera_angle}, {prompt_keywords}"
    
    return final_positive_prompt

def handler(job):
    if initialization_error:
        return {"error": initialization_error}
    
    if not pipe:
        return {"error": "Worker is not properly initialized."}

    try:
        job_input = job['input']
        
        # --- 추가된 로그 ---
        logging.info(f"--- 핸들러가 받은 전체 입력 데이터 ---\n{job_input}")
        
        positive_prompt = create_prompts(job_input)
        logging.info(f"--- 최종 생성된 Positive 프롬프트 ---\n{positive_prompt}")
        
        with torch.no_grad():
            seed = job_input.get('seed', random.randint(0, 2**32 - 1))
            generator = torch.Generator("cuda").manual_seed(seed)
            image = pipe(
                prompt=positive_prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                generator=generator
            ).images[0]
        
        logging.info("이미지 생성 완료. S3에 업로드 중...")
        
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
        logging.info(f"이미지 업로드 성공: {image_url}")
        
        return {
            "image_url": image_url,
            "image_prompt": positive_prompt
        }

    except NoCredentialsError:
        error_msg = "S3 credentials not found."
        logging.error(error_msg)
        return {"error": error_msg}
    except Exception as e:
        logging.error(f"Error during handling job: {e}", exc_info=True)
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": f"Job failed: {e}", "trace": traceback.format_exc()}

# RunPod 핸들러 시작
runpod.serverless.start({"handler": handler})