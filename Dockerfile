# 1. CUDA12.4 + cuDNN8 런타임 + Ubuntu22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# 2. tzdata 비대화형 설치, 타임존 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 3. 타임존 링크 및 Python3.11 설치
RUN ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
 && apt-get update \
 && apt-get install -y --no-install-recommends software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update \
 && apt-get install -y --no-install-recommends python3.11 python3.11-venv python3.11-dev git \
 && ln -sf /usr/bin/python3.11 /usr/bin/python \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 4. 다운로드 스크립트와 의존성 리스트 복사
COPY requirements.txt download.py ./

# 5. HF_TOKEN 빌드 아규먼트로 받기
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# 6. 가상환경 생성 및 PyTorch, 나머지 패키지 설치
RUN python -m venv venv \
 && . venv/bin/activate \
 && pip install --upgrade pip \
 && pip install --no-cache-dir \
      torch==2.6.0+cu124 \
      torchvision==0.21.0+cu124 \
      torchaudio==2.6.2+cu124 \
    -f https://download.pytorch.org/whl/cu124/torch_stable.html \
 && pip install --no-cache-dir -r requirements.txt

# 7. 모델 웨이트 다운로드
RUN . venv/bin/activate && python download.py

# 8. 소스 복사 및 핸들러 실행 설정
COPY . .
ENV PATH="/app/venv/bin:$PATH"
CMD ["python", "handler.py"]
