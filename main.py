import time
import uuid
import urllib.parse
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="KROT GPT Multimodal API", version="2.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_tasks: Dict[str, Dict[str, Any]] = {}


# --- МОДЕЛИ ДАННЫХ ---

class Message(BaseModel):
    role: str
    content: Any  # Any, чтобы не падало, если виджет шлет массив с файлами/картинками

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "openai"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

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
        "service": "KROT GPT API v2.2",
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

def build_image_url(prompt: str, width: int = 1024, height: int = 1024, model: str = "flux") -> str:
    encoded_prompt = urllib.parse.quote(prompt)
    seed = uuid.uuid4().int % 1000000
    return (
        f"https://pollinations.ai{encoded_prompt}"
        f"?width={width}&height={height}&seed={seed}&model={model}&nologo=true"
    )

async def get_free_llm_response(messages: List[Dict[str, str]]) -> str:
    fallback_models = ["openai", "mistral", "llama"]
    last_error = ""

    async with httpx.AsyncClient(timeout=40.0) as client:
        for model_name in fallback_models:
            try:
                response = await client.post(
                    "https://pollinations.ai",
                    json={"model": model_name, "messages": messages},
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code == 200:
                    data = response.json()
                    if isinstance(data, dict) and "choices" in data and len(data["choices"]) > 0:
                        return data["choices"][0]["message"]["content"]
                last_error = f"HTTP {response.status_code}: {response.text[:150]}"
            except Exception as e:
                last_error = str(e)

        try:
            last_user_msg = messages[-1]["content"] if messages else "Привет"
            encoded = urllib.parse.quote(last_user_msg)
            fallback_res = await client.get(f"https://pollinations.ai{encoded}")
            if fallback_res.status_code == 200 and fallback_res.text.strip():
                return fallback_res.text.strip()
        except Exception as e:
            last_error = str(e)

    raise RuntimeError(f"Шлюзы недоступны: {last_error}")


# --- 1. ТЕКСТОВЫЙ ЧАТ + УМНОЕ АВТО-РИСОВАНИЕ ---

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    try:
        formatted_messages = [
            {"role": m.role, "content": extract_text(m.content)}
            for m in req.messages
        ]
        last_text = formatted_messages[-1]["content"].strip() if formatted_messages else ""
        lower_text = last_text.lower()

        # Проверяем, просит ли пользователь нарисовать картинку прямо в обычном чате
        draw_triggers = ("нарисуй", "сгенерируй картинку", "сгенерируй изображение", "создай картинку", "draw ")
        if any(lower_text.startswith(trigger) for trigger in draw_triggers):
            # Переводим запрос на английский для лучшего качества Flux
            en_prompt = await get_free_llm_response([
                {
                    "role": "system",
                    "content": "Translate the user image request into a detailed English image generation prompt. Reply ONLY with the English prompt, no extra words."
                },
                {"role": "user", "content": last_text}
            ])
            img_url = build_image_url(en_prompt.strip())
            reply_text = f"Вот ваше изображение по запросу «{last_text}»:\n\n![Изображение]({img_url})\n\n[Открыть картинку в полном размере]({img_url})"
        else:
            reply_text = await get_free_llm_response(formatted_messages)

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


@app.post("/chat")
async def simple_chat(req: SimpleChatRequest):
    try:
        reply_text = await get_free_llm_response([{"role": "user", "content": req.message}])
        return {"reply": reply_text, "response": reply_text}
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
            img_url = build_image_url(req.prompt, width, height, req.model or "flux")
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
        video_url = result[0] if isinstance(result, (list, tuple)) else str(result)
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
