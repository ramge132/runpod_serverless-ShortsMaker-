# 베이스 이미지 선택 (PyTorch 공식 이미지로 변경하여 안정성 확보)
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# apt-get이 사용자 입력을 기다리지 않도록 설정
ENV DEBIAN_FRONTEND=noninteractive

# 시스템 패키지 업데이트 및 필수 라이브러리 설치
# -y 플래그와 --no-install-recommends를 추가하여 안정성 향상
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 먼저 복사하여 Docker 캐시를 효율적으로 사용
COPY requirements.txt .

# PyTorch 핵심 라이브러리를 먼저 설치
RUN pip install --no-cache-dir \
    torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# 그 다음 requirements.txt의 나머지 패키지들을 설치 (xformers 포함)
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 소스 코드 복사
COPY . .

# 모델 다운로드 스크립트 실행 -> 이 단계는 handler.py의 시작 부분으로 이동합니다.
# RUN python download.py

# 컨테이너 실행 시 핸들러 스크립트 실행
CMD ["python", "-u", "handler.py"]
