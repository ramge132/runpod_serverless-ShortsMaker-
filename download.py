import os
from huggingface_hub import snapshot_download

# 다운로드 받을 모델의 Hugging Face 리포지토리 ID
HF_REPO_ID = "black-forest-labs/FLUX.1-dev"
# 가중치를 저장할 디렉토리
WEIGHT_DIR = "weights"

def download_model():
    """Hugging Face Hub에서 모델 스냅샷을 다운로드합니다."""
    print(f"{HF_REPO_ID} 모델 다운로드를 시작합니다...")
    
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)

    try:
        # snapshot_download는 리포지토리 전체를 다운로드하여 편리합니다.
        snapshot_download(
            repo_id=HF_REPO_ID,
            local_dir=WEIGHT_DIR,
            local_dir_use_symlinks=False,
            ignore_patterns=["*.md", "*.git*", "*.png", "*.jpg"] # 불필요한 파일은 제외
        )
        print("모델 다운로드가 완료되었습니다.")
    except Exception as e:
        print(f"모델 다운로드 실패: {e}")

if __name__ == "__main__":
    download_model()
