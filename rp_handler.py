import runpod
import json
import uuid
import asyncio
import websockets
import os

# ComfyUI 서버 주소 및 설정값 정의
COMFY_HOST = "127.0.0.1:8188"
COMFY_API_AVAILABLE_INTERVAL_MS = 50
COMFY_API_AVAILABLE_MAX_RETRIES = 500
WEBSOCKET_TIMEOUT = 180

# WebSocket을 통해 ComfyUI에 워크플로우 요청 및 이미지 생성 결과 수신
async def get_images_over_websocket(workflow, client_id, job):
    uri = f"ws://{COMFY_HOST}/ws"
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({"id": client_id, "workflow": workflow}))
        while True:
            message = await ws.recv()
            data = json.loads(message)
            runpod.serverless.progress_update(job, data.get("status", "processing"))
            if data.get("status") == "completed":
                return {"images": data.get("images")}

# RunPod에서 들어오는 작업(job)을 처리하는 핸들러 함수
async def handler(job):
    job_input = job.get("input")
    if not job_input:
        return {"error": "입력값이 제공되지 않았습니다."}
    if isinstance(job_input, str):
        job_input = json.loads(job_input)

    workflow_type = job_input.get("workflow_type")
    user_prompt = job_input.get("prompt")

    if not workflow_type:
        return {"error": "'workflow_type' 파라미터가 누락되었습니다."}

    workflow_path = f"/workspace/workflows/{workflow_type}.json"
    if not os.path.exists(workflow_path):
        return {"error": f"지정된 워크플로우 템플릿이 존재하지 않습니다: {workflow_type}"}

    with open(workflow_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)
        workflow = full_data.get("input", {}).get("workflow")

    if not workflow:
        return {"error": "템플릿 JSON에서 'workflow' 데이터를 찾을 수 없습니다."}

    try:
        for node_id, node in workflow.items():
            if node.get("class_type") == "CLIPTextEncode":
                node["inputs"]["text"] = user_prompt
    except Exception as e:
        return {"error": f"워크플로우 프롬프트 삽입 중 오류: {str(e)}"}

    client_id = str(uuid.uuid4())

    try:
        output = await asyncio.wait_for(
            get_images_over_websocket(workflow, client_id, job),
            timeout=WEBSOCKET_TIMEOUT
        )
    except asyncio.TimeoutError:
        return {"status": "timeout", "images": None}
    except Exception as e:
        return {"error": str(e)}

    runpod.serverless.progress_update(job, "완료. 결과 반환 중")
    return output

if __name__ == "__main__":
    runpod.serverless.start({
        "handler": handler,
        "return_aggregate_stream": True
    })