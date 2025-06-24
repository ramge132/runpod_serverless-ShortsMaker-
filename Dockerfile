# 1. 베이스 이미지: CUDA 12.4 + cuDNN 런타임
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 2. 시간대 설정 및 tzdata 설치
RUN ln -fs /usr/share/zoneinfo/Asia/Seoul /etc/localtime \
 && apt-get update \
 && apt-get install -y --no-install-recommends tzdata \
 && rm -rf /var/lib/apt/lists/*

# 3. deadsnakes PPA 추가 후 Python 3.11 설치
RUN apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev git curl \
 && ln -sf /usr/bin/python3.11 /usr/bin/python \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# 4. 작업 디렉토리
WORKDIR /app

# 5. 애플리케이션 코드 복사
COPY requirements.txt download.py handler.py ./

# 6. 가상환경 생성 및 PyTorch/CUDA, Nunchaku 설치
RUN python -m venv venv \
 && . venv/bin/activate \
 && pip install --upgrade pip \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu124 \
      torch==2.6.0 \
      torchvision==0.21.0 \
      torchaudio==2.6.0+cu124 \
 && pip install --no-cache-dir \
      https://huggingface.co/mit-han-lab/nunchaku/resolve/main/\
nunchaku-0.2.0+torch2.6-cp311-cp311-linux_x86_64.whl \
 && pip install --no-cache-dir -r requirements.txt

# 7. 나머지 애플리케이션 파일 복사
COPY . .

# 8. 컨테이너 시작 커맨드
CMD ["venv/bin/python", "handler.py"]
