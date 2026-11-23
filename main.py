import json
import time
import uuid
import urllib.parse
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="KROT GPT Multimodal API", version="Hacker v2.4")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_tasks: Dict[str, Dict[str, Any]] = {}

# Адрес вашего собственного сервера Render (чтобы картинки отдавались через него без сбоев)
SELF_BASE_URL = "https://krot-gpt-api.onrender.com"


# --- МОДЕЛИ ДАННЫХ ---

class Message(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "openai"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    model: Optional[str] = "flux"

class VideoGenerationRequest(BaseModel):
    prompt: str


# --- СЛУЖЕБНЫЕ МАРШРУТЫ ---

@app.get("/")
@app.get("/v1")
async def health_check():
    return {
        "status": "online",
        "service": "KROT GPT API v2.4",
        "endpoints": [
            "/v1/chat/completions",
            "/v1/images/generations",
            "/v1/videos/generations"
        ]
    }

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "krot-gpt-text", "object": "model", "owned_by": "krot"},
            {"id": "krot-flux-image", "object": "model", "owned_by": "krot"},
            {"id": "krot-ltx-video", "object": "model", "owned_by": "krot"},
        ]
    }


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"]
        return " ".join(texts)
    return str(content)

def build_proxy_image_url(prompt: str, width: int = 1024, height: int = 1024, model: str = "flux") -> str:
    encoded_prompt = urllib.parse.quote(prompt)
    seed = uuid.uuid4().int % 1000000
    # Возвращаем ссылку на наш же сервер Render, который сам скачает и отдаст картинку
    return (
        f"{SELF_BASE_URL}/v1/image-proxy"
        f"?prompt={encoded_prompt}&width={width}&height={height}&seed={seed}&model={model}"
    )

async def get_free_llm_response(messages: List[Dict[str, str]]) -> str:
    last_error = ""
    pollinations_text_host = "https://text." + "pollinations" + ".ai"

    async with httpx.AsyncClient(timeout=40.0) as client:
        for model_name in ["openai", "mistral", "llama"]:
            try:
                response = await client.post(
                    f"{pollinations_text_host}/openai",
                    json={"model": model_name, "messages": messages},
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices")
                    if isinstance(choices, list) and len(choices) > 0:
                        msg = choices[0].get("message", {})
                        if msg.get("content"):
                            return msg["content"]
                last_error = f"HTTP {response.status_code}"
            except Exception as e:
                last_error = str(e)

        try:
            last_user_msg = messages[-1]["content"] if messages else "Привет"
            encoded = urllib.parse.quote(last_user_msg)
            fallback_res = await client.get(f"{pollinations_text_host}/{encoded}")
            if fallback_res.status_code == 200 and fallback_res.text.strip():
                return fallback_res.text.strip()
        except Exception as e:
            last_error = str(e)

    return f"Временная перегрузка бесплатного шлюза ({last_error}). Попробуйте повторить запрос через 5 секунд."


# --- ПРОКСИ ДЛЯ КАРТИНОК (Render сам забирает картинку и отдаёт браузеру) ---

@app.get("/v1/image-proxy")
async def image_proxy(prompt: str, width: int = 1024, height: int = 1024, seed: int = 42, model: str = "flux"):
    encoded_prompt = urllib.parse.quote(prompt)
    img_host = "https://image." + "pollinations" + ".ai"
    target_url = f"{img_host}/prompt/{encoded_prompt}?width={width}&height={height}&seed={seed}&model={model}&nologo=true"

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        try:
            res = await client.get(target_url)
            if res.status_code == 200:
                content_type = res.headers.get("content-type", "image/jpeg")
                return Response(content=res.content, media_type=content_type)
            raise HTTPException(status_code=res.status_code, detail="Не удалось сгенерировать изображение")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")


# --- 1. ТЕКСТОВЫЙ ЧАТ + МГНОВЕННОЕ РИСОВАНИЕ ---

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    try:
        formatted_messages = [
            {"role": m.role, "content": extract_text(m.content)}
            for m in req.messages
        ]
        last_text = formatted_messages[-1]["content"].strip() if formatted_messages else ""
        lower_text = last_text.lower()

        draw_triggers = ("нарисуй", "сгенерируй картинку", "сгенерируй изображение", "создай картинку", "draw ")
        if any(lower_text.startswith(trigger) for trigger in draw_triggers):
            clean_prompt = last_text
            for trigger in draw_triggers:
                if lower_text.startswith(trigger):
                    clean_prompt = last_text[len(trigger):].strip() or last_text
                    break
            img_url = build_proxy_image_url(clean_prompt)
            reply_text = (
                f"Готово! Изображение по запросу «{clean_prompt}»:\n\n"
                f"![Изображение]({img_url})\n\n"
                f"[Открыть в полном размере]({img_url})"
            )
        else:
            reply_text = await get_free_llm_response(formatted_messages)

        if req.stream:
            async def event_stream():
                chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model or "krot-gpt-text",
                    "choices": [{"index": 0, "delta": {"content": reply_text}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                done_chunk = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": req.model or "krot-gpt-text",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
                }
                yield f"data: {json.dumps(done_chunk, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(event_stream(), media_type="text/event-stream")

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model or "krot-gpt-text",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": reply_text
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 2. ПРЯМАЯ ГЕНЕРАЦИЯ КАРТИНОК ---

@app.post("/v1/images/generations")
async def generate_image(req: ImageGenerationRequest):
    try:
        width, height = 1024, 1024
        if req.size and "x" in req.size:
            parts = req.size.lower().split("x")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                width, height = int(parts[0]), int(parts[1])

        images_data = []
        count = min(max(req.n or 1, 1), 4)
        for _ in range(count):
            img_url = build_proxy_image_url(req.prompt, width, height, req.model or "flux")
            images_data.append({"url": img_url})

        return {
            "created": int(time.time()),
            "data": images_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации изображения: {str(e)}")


# --- 3. ГЕНЕРАЦИЯ ВИДЕО (Асинхронная) ---

def run_video_worker(task_id: str, prompt: str):
    try:
        from gradio_client import Client
        client = Client("Lightricks/LTX-Video")
        result = client.predict(
            prompt=prompt,
            negative_prompt="low quality, worst quality, deformed, distorted, blurry",
            api_name="/generate"
        )
        video_url = result[0] if isinstance(result, (list, tuple)) and len(result) > 0 else str(result)
        video_tasks[task_id] = {
            "status": "completed",
            "video_url": video_url
        }
    except Exception as e:
        video_tasks[task_id] = {
            "status": "failed",
            "error": f"Сервер видео перегружен: {str(e)}"
        }


@app.post("/v1/videos/generations")
async def create_video(req: VideoGenerationRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    video_tasks[task_id] = {"status": "processing"}
