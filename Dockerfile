# 베이스 이미지 선택
FROM runpod/pytorch:2.1.0-py3.10-cuda12.1.1-devel

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
# Python 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 나머지 소스 코드 복사
COPY . .

# 모델 다운로드 스크립트 실행
RUN python download.py

# 컨테이너 실행 시 핸들러 스크립트 실행
CMD ["python", "-u", "handler.py"]
