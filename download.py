import os
from huggingface_hub import snapshot_download, login

# --- Hugging Face 로그인 ---
# RunPod의 환경 변수에서 HF_TOKEN을 읽음
hf_token = os.environ.get("HF_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face Token not found in environment variables. Please set HF_TOKEN.")

print("Logging in to Hugging Face Hub...")
login(token=hf_token)
print("Login successful.")

# --- 모델 다운로드 ---
# 다운로드 받을 모델의 Hugging Face 리포지토리 ID
HF_REPO_ID = "black-forest-labs/FLUX.1-dev"
# 가중치를 저장할 디렉토리 (하위 디렉토리 사용)
WEIGHT_DIR = "weights/flux1_dev"

def download_model():
    """Hugging Face Hub에서 Gated 모델 스냅샷을 다운로드합니다."""
    print(f"{HF_REPO_ID} 모델 다운로드를 시작합니다...")
    
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)

    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=WEIGHT_DIR,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md","*.git*","*.png","*.jpg"]
        )
        print("모델 다운로드가 완료되었습니다.")
    except Exception as e:
        print(f"모델 다운로드 실패: {e}")
        raise e

if __name__ == "__main__":
    download_model()
