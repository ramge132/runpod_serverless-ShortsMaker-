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
import random

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- OpenAI 클라이언트 초기화 ---
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- 카메라 구도 와일드카드 리스트 ---
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
S3_BUCKET_NAME = os.environ.get('S3_BUCKET_NAME')
AWS_REGION = os.environ.get('AWS_REGION')

logging.info("--- 워커 초기화 시작 ---")
try:
    if not S3_BUCKET_NAME or not AWS_REGION:
        raise ValueError("S3_BUCKET_NAME and AWS_REGION environment variables must be set.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable must be set.")
    if not os.path.exists(base_model_path):
        logging.info("Base model not found. Downloading...")
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token: raise ValueError("HF_TOKEN not found.")
        login(token=hf_token)
        snapshot_download(repo_id="black-forest-labs/FLUX.1-dev", local_dir=base_model_path, local_dir_use_symlinks=False, ignore_patterns=["*.md", "*.git*", "*.png", "*.jpg", "transformer/*"])
    if not os.path.exists(nunchaku_model_path):
        logging.info("Nunchaku model repo not found. Downloading...")
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
    logging.info("--- 워커 및 파이프라인 초기화 성공 ---")
except Exception as e:
    initialization_error = f"Initialization failed: {e}"
    logging.error("--- 워커 초기화 실패 ---", exc_info=True)

# ------------------------------------

def generate_prompt_keywords(korean_context):
    """
    한국어 컨텍스트를 받아, 전문적인 스타일의 영어 프롬프트 키워드를 생성합니다.
    """
    if not korean_context:
        return ""
        
    logging.info(f"--- [LLM 호출] 프롬프트 키워드 생성을 위한 한국어 컨텍스트 ---\n{korean_context}")
    
    # 시스템 프롬프트 수정: 작가 이름 지시 제거, 캐릭터 이름 처리 규칙 수정
    system_prompt = f"""You are a world-class prompt engineer for photorealistic image AI. Your ONLY job is to convert a Korean context into a rich, detailed, comma-separated list of English keywords.

**CRITICAL INSTRUCTIONS:**
1.  **Analyze Subject**: Based on the Korean text, identify the main subject.
    - If the character's name (`등장인물 이름`) describes the subject itself (e.g., '자라' for a turtle, '드래곤' for a dragon), **YOU MUST USE THIS NAME** as the main subject.
    - If the character's name is a common human name (e.g., '영희'), describe the person generically using their description (e.g., 'a Korean girl with black hair').
2.  **Enrich Prompt**: You MUST add professional stylistic keywords. Always include keywords for:
    - **Quality & Style**: `masterpiece`, `high-quality`, `4k full HD photo`, `photorealistic`
    - **Lighting**: `cinematic lighting`, `volumetric light and shadows`
3.  **STRICT OUTPUT FORMAT**:
    - **ONLY** output comma-separated English keywords.
    - **DO NOT** write sentences or explanations.
    - **DO NOT** add artist names like 'by...'.

**Example 1 (Subject name is the character):**
Korean text: "장면: 자라는 자라. 등장인물: 자라: 녹색 피부, 작은 크기, 귀여운 외형"
Correct Output: `masterpiece, high-quality, 4k photo, a cute turtle named Jara, green skin, small size, cinematic lighting, volumetric light and shadows`

**Example 2 (Human character):**
Korean text: "장면: 맛있으면 또 먹지. 등장인물: 철수: 검은 머리, 검은 눈의 남자"
Correct Output: `masterpiece, high-quality, 4k photo, a Korean man with black hair and black eyes, looking satisfied while eating, cinematic lighting, volumetric light and shadows`
"""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Korean text: \"{korean_context}\""}
            ],
            temperature=0.2,
        )
        generated_keywords = response.choices[0].message.content.strip()
        logging.info(f"--- [LLM 응답] 생성된 프롬프트 키워드 ---\n{generated_keywords}")
        return generated_keywords
    except Exception as e:
        logging.error(f"Error during prompt keyword generation: {e}")
        return "error generating prompt"

def create_prompts(input_data):
    """입력 데이터로부터 최종 Positive 프롬프트를 생성합니다."""
    logging.info("--- [단계 1] 프롬프트 생성 프로세스 시작 ---")
    
    random_camera_angle = random.choice(CAMERA_ANGLES)
    logging.info(f"--- [단계 1.1] 랜덤 카메라 구도 선택: '{random_camera_angle}' ---")
    
    # LLM에 이름과 설명을 모두 전달하도록 수정
    metadata = input_data.get('story_metadata', {})
    audios = input_data.get('audios', [])
    scene_text = audios[0].get('text', '') if audios else ''
    
    characters = metadata.get('characters', [])
    character_info_kr = ""
    if characters:
        char_descs = []
        for char in characters:
            name = char.get('name', '캐릭터')
            desc = char.get('description', '묘사 없음') 
            char_descs.append(f"{name}: {desc}") # "이름: 설명" 형식으로 변경
        character_info_kr = f"등장인물: {', '.join(char_descs)}"

    full_korean_context = f"장면: {scene_text}. {character_info_kr}".strip()
    
    prompt_keywords = generate_prompt_keywords(full_korean_context)

    final_positive_prompt = f"{random_camera_angle}, {prompt_keywords}"
    logging.info("--- [단계 1.2] 최종 프롬프트 조합 완료 ---")
    
    return final_positive_prompt

def handler(job):
    if initialization_error:
        return {"error": initialization_error}
    
    if not pipe:
        return {"error": "Worker is not properly initialized."}

    try:
        job_input = job['input']
        logging.info(f"--- [시작] 핸들러가 받은 전체 입력 데이터 ---\n{job_input}")
        
        positive_prompt = create_prompts(job_input)
        logging.info(f"--- [단계 2] 최종 생성된 Positive 프롬프트 ---\n{positive_prompt}")
        
        with torch.no_grad():
            seed = job_input.get('seed', random.randint(0, 2**32 - 1))
            logging.info(f"--- [단계 3] 이미지 생성 시작 (Seed: {seed}, Steps: 50, Guidance: 3.5) ---")
            generator = torch.Generator("cuda").manual_seed(seed)
            image = pipe(
                prompt=positive_prompt,
                height=1024,
                width=1024,
                guidance_scale=3.5,
                num_inference_steps=50,
                generator=generator
            ).images[0]
        
        logging.info("--- [단계 4] 이미지 생성 완료. S3 업로드 시작... ---")
        
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
        logging.info(f"--- [단계 5] S3 업로드 완료. 이미지 URL: {image_url} ---")
        
        return {
            "image_url": image_url,
            "image_prompt": positive_prompt
        }
    except Exception as e:
        logging.error(f"--- 작업 처리 중 에러 발생 ---", exc_info=True)
        gc.collect()
        torch.cuda.empty_cache()
        return {"error": f"Job failed: {e}", "trace": traceback.format_exc()}

# RunPod 핸들러 시작
runpod.serverless.start({"handler": handler})