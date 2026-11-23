import os
import time
import uuid
import urllib.parse
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="KROT GPT Multimodal API", version="2.0")

# Разрешаем запросы с любых доменов (обязательно для работы с Tilda)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Хранилище статусов видео-задач
video_tasks: Dict[str, Dict[str, Any]] = {}


# --- МОДЕЛИ ДАННЫХ ---

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "gpt-4o-mini"
    messages: List[Message]
    temperature: Optional[float] = 0.7

class SimpleChatRequest(BaseModel):
    message: str

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
        "service": "KROT GPT API",
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


# --- 1. ТЕКСТОВЫЙ ЧАТ (OpenAI-совместимый) ---

async def get_free_llm_response(messages: List[Dict[str, str]]) -> str:
    """
    Отправляет запрос в бесплатный текстовый шлюз Pollinations Text API
    (не требует платных ключей, поддерживает контекст диалога).
    """
    url = "https://pollinations.ai"
    payload = {
        "model": "openai",
        "messages": messages
    }
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    try:
        formatted_messages = [{"role": m.role, "content": m.content} for m in req.messages]
        reply_text = await get_free_llm_response(formatted_messages)
        
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
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
        raise HTTPException(status_code=500, detail=f"Ошибка генерации текста: {str(e)}")

# Дополнительный упрощённый маршрут (на случай если старый виджет слал на /chat)
@app.post("/chat")
async def simple_chat(req: SimpleChatRequest):
    reply_text = await get_free_llm_response([{"role": "user", "content": req.message}])
    return {"reply": reply_text, "response": reply_text}


# --- 2. ГЕНЕРАЦИЯ КАРТИНОК ---

@app.post("/v1/images/generations")
async def generate_image(req: ImageGenerationRequest):
    try:
        # Разбираем размер вида "1024x1024"
        width, height = 1024, 1024
        if req.size and "x" in req.size:
            parts = req.size.lower().split("x")
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                width, height = int(parts[0]), int(parts[1])

        encoded_prompt = urllib.parse.quote(req.prompt)
        images_data = []

        count = min(max(req.n or 1, 1), 4)
        for _ in range(count):
            seed = uuid.uuid4().int % 1000000
            image_url = (
                f"https://pollinations.ai{encoded_prompt}"
                f"?width={width}&height={height}&seed={seed}&model={req.model}&nologo=true"
            )
            images_data.append({"url": image_url})

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
        # Подключаемся к публичному бесплатному генератору видео на Hugging Face
        client = Client("Lightricks/LTX-Video")
        result = client.predict(
            prompt=prompt,
            negative_prompt="low quality, worst quality, deformed, distorted, blurry",
            api_name="/generate"
        )
        video_url = result[0] if isinstance(result, (list, tuple)) else str(result)
        video_tasks[task_id] = {
            "status": "completed",
            "video_url": video_url
        }
    except Exception as e:
        video_tasks[task_id] = {
            "status": "failed",
            "error": f"Сервер видео перегружен или недоступен: {str(e)}"
        }

@app.post("/v1/videos/generations")
async def create_video(req: VideoGenerationRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    video_tasks[task_id] = {"status": "processing"}
    background_tasks.add_task(run_video_worker, task_id, req.prompt)
    return {
        "task_id": task_id,
        "status": "processing",
        "check_url": f"/v1/videos/status/{task_id}"
    }

@app.get("/v1/videos/status/{task_id}")
async def check_video_status(task_id: str):
    task = video_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return task
