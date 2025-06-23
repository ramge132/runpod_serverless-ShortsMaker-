import os
from huggingface_hub import snapshot_download, login


def download_model_legacy():
    """
    이 함수는 더 이상 사용되지 않지만, 로직 기록을 위해 남김.
    """
    # RunPod의 환경 변수에서 HF_TOKEN을 읽어옵니다.
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("Hugging Face Token not found in environment variables. Please set HF_TOKEN.")

    print("Logging in to Hugging Face Hub...")
    login(token=hf_token)
    print("Login successful.")

    HF_REPO_ID = "black-forest-labs/FLUX.1-dev"
    WEIGHT_DIR = "weights/flux1_dev"
    
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)

    print(f"{HF_REPO_ID} 모델 다운로드를 시작합니다...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        local_dir=WEIGHT_DIR,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md","*.git*","*.png","*.jpg"]
    )
    print("모델 다운로드가 완료되었습니다.")

if __name__ == "__main__":
    print("이 스크립트는 더 이상 직접 실행되지 않습니다. 모든 로직은 handler.py에 통합되었습니다.")
