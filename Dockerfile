# 베이스 이미지 선택 (사용할 모델에 맞는 PyTorch 및 CUDA 버전 선택)
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel

# 시스템 패키지 업데이트 및 필수 라이브러리 설치
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 Python 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 소스 코드 복사
COPY . .

# 모델 다운로드 스크립트 실행
RUN python download.py

# 컨테이너 실행 시 핸들러 스크립트 실행
CMD ["python", "-u", "handler.py"]
