#!/usr/bin/env bash
set -e

# 1) ComfyUI 서버 실행 (포트 8188, 백그라운드 실행)
python /ComfyUI/main.py --listen 0.0.0.0 --port 8188 &

# 2) RunPod Serverless 핸들러 실행 (작업 디렉토리에서)
python /workspace/rp_handler.py