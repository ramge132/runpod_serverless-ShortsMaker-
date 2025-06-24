# 1. CUDA12.4 + cuDNN8 런타임 + Ubuntu22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 2. non-interactive tzdata 설정
ENV DEBIAN_FRONTEND=noninteractive TZ=Asia/Seoul
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

# 3. Python3.11 설치 (deadsnakes PPA)
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev git \
 && ln -sf /usr/bin/python3.11 /usr/bin/python \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. 앱 의존 파일 복사
COPY requirements.txt download.py handler.py ./

# 5. 가상환경 만들고 PyTorch/CUDA-12.4, Nunchaku 등 설치
RUN python -m venv venv \
 && . venv/bin/activate \
 && pip install --upgrade pip \
 # PyTorch 2.6 + CUDA-12.4
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.2 \
 # Nunchaku wheel (Python3.11 + PyTorch2.6 + Linux x86_64)
 && pip install --no-cache-dir \
      https://huggingface.co/mit-han-lab/nunchaku/resolve/main/nunchaku-0.2.0+torch2.6-cp311-cp311-linux_x86_64.whl \
 # 나머지 requirements
 && pip install --no-cache-dir -r requirements.txt

# 6. 모델 다운로드는 런타임에 handler.py 내에서 수행되므로 여기서는 생략
#    (HF_TOKEN 은 RunPod UI / env 설정에서 런타임에 주입)

# 7. 소스 전체 복사
COPY . .

# 8. venv 경로 우선 설정 & 서버리스 핸들러
ENV PATH="/app/venv/bin:${PATH}"
CMD ["python", "handler.py"]
