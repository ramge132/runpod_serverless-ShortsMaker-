import os
from huggingface_hub import snapshot_download, login

def download_models():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN 환경변수 미설정")

    print("Hugging Face 로그인 시도...")
    login(token=hf_token)
    print("로그인 성공")

    # 1) Flux1.dev 베이스 모델
    repo_flux = "black-forest-labs/FLUX.1-dev"
    dir_flux  = "weights/flux1_dev"
    os.makedirs(dir_flux, exist_ok=True)
    print(f"{repo_flux} 다운로드 시작...")
    snapshot_download(
        repo_id=repo_flux,
        local_dir=dir_flux,
        local_dir_use_symlinks=False,
        ignore_patterns=["*.md","*.git*","*.png","*.jpg"]
    )
    print("Flux1.dev 모델 다운로드 완료")

    # 2) Nunchaku 트랜스포머 웨이트
    repo_nun = "mit-han-lab/nunchaku-flux.1-dev"
    dir_nun  = "weights/nunchaku_transformer"
    os.makedirs(dir_nun, exist_ok=True)
    print(f"{repo_nun} 트랜스포머 다운로드 시작...")
    snapshot_download(
        repo_id=repo_nun,
        local_dir=dir_nun,
        local_dir_use_symlinks=False,
        allow_patterns=["*.safetensors"]
    )
    print("Nunchaku 트랜스포머 다운로드 완료")

if __name__ == "__main__":
    download_models()
