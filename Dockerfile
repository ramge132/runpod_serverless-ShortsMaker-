# 1. CUDA12.4 + cuDNN8 런타임 + Ubuntu22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 2. Python 3.11 설치 (deadsnakes PPA)
RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y python3.11 python3.11-venv python3.11-dev git \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 3. 의존성 리스트와 다운로드 스크립트 먼저 복사
COPY requirements.txt download.py ./

# 4. 빌드 시 HF_TOKEN 전달받도록
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# 5. 가상환경 생성 및 의존성 설치
RUN python -m venv venv \
 && . venv/bin/activate \
 && pip install --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.6.0+cu124 \
      torchvision==0.21.0+cu124 \
      torchaudio==2.6.2+cu124 \
    -f https://download.pytorch.org/whl/cu124/torch_stable.html \
 && pip install --no-cache-dir -r requirements.txt

# 6. 모델 웨이트 다운로드
RUN . venv/bin/activate && python download.py

# 7. 나머지 소스 복사
COPY . .

# 8. 핸들러 실행
ENV PATH="/app/venv/bin:$PATH"
CMD ["python", "handler.py"]
