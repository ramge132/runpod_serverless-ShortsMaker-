import os
import json
import logging

from huggingface_hub import hf_hub_download
from nunchaku.models.transformers.transformer_flux import NunchakuFluxTransformer2dModel

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_model():
    """
    Hugging Face Hub에서 모델을 가져와 초기화합니다.
    repo_id와 filename을 분리하여 from_pretrained 또는 hf_hub_download 페일오버 처리.
    """
    device    = os.environ.get("DEVICE", "cuda")
    precision = os.environ.get("PRECISION", "fp4")  # int4, fp4 등
    repo_id   = "mit-han-lab/nunchaku-flux.1-dev"
    filename  = "svdq-fp4_r32-flux.1-dev.safetensors"
    token     = os.environ.get("HF_TOKEN", None)

    try:
        logger.info(f"Loading model via from_pretrained(repo_id={repo_id}, filename={filename}, precision={precision})")
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            device=device,
            precision=precision,
            token=token,
        )
    except TypeError:
        # from_pretrained 에 filename 인자가 없을 때의 페일오버
        logger.info("from_pretrained 에 filename 인자 미지원, hf_hub_download 로 weights 파일을 먼저 다운로드합니다.")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
        )
        transformer = NunchakuFluxTransformer2dModel.from_pretrained(
            local_path,
            device=device,
            precision=precision,
        )

    return transformer

# 서버리스 시작 시 단 한 번만 모델을 초기화
logger.info("모델 초기화 시작...")
transformer = initialize_model()
logger.info("모델 초기화 완료.")

def handler(event):
    """
    RunPod Serverless가 호출하는 entry point.
    event["body"] 에 JSON 형식의 페이로드가 들어옵니다.
    """
    try:
        # event.body 가 문자열일 수 있으니 파싱
        body = event.get("body", {})
        if isinstance(body, str):
            body = json.loads(body)
        input_data = body.get("input", {})

        logger.info(f"Received input: {input_data}")
        # Nunchaku Flux 모델에 inference 요청
        result = transformer.run_inference(input_data)

        return {
            "statusCode": 200,
            "body": json.dumps({"output": result}),
        }

    except Exception as e:
        logger.error("Error during inference", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)}),
        }

if __name__ == "__main__":
    # 로컬 테스트 혹은 Docker 컨테이너 내 직접 실행용
    from runpod.serverless import start
    start(handler)
