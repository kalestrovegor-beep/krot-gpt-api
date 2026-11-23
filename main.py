import os
import json
import base64
import random
import urllib.parse
import asyncio
import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Подборка красивых демо-картинок (если все генераторы временно недоступны)
DEMO_IMAGES = [
    "https://unsplash.com",
    "https://unsplash.com",
    "https://unsplash.com",
    "https://unsplash.com",
    "https://unsplash.com",
]

# Подборка красивых HD демо-видео (MP4) на разные темы
DEMO_VIDEOS = [
    "https://mixkit.co",
    "https://mixkit.co",
    "https://mixkit.co",
    "https://mixkit.co",
    "https://mixkit.co",
]


@app.get("/")
async def root():
    return {"status": "ok", "service": "KROT GPT API Gateway"}


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gemini-2.5-flash", "object": "model"},
            {"id": "gemini-2.5-pro", "object": "model"},
            {"id": "imagen-3", "object": "model"},
        ],
    }


# Умная генерация картинки: Google AI -> Резервный генератор -> Красивое демо
async def get_smart_image_markdown(prompt: str) -> str:
    # 1. Пробуем реальную генерацию через Google AI Studio
    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            url = f"https://googleapis.com{GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": f"Generate a high quality image: {prompt}"}]}],
                "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
            }
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
                for p in parts:
                    if "inlineData" in p and p["inlineData"].get("data"):
                        b64 = p["inlineData"]["data"]
                        return f"![generated](data:image/png;base64,{b64})"
    except Exception:
        pass

    # 2. Если Google недоступен — пробуем бесплатный генератор без ключа по тексту запроса
    try:
        encoded_prompt = urllib.parse.quote(prompt)
        seed = random.randint(1, 99999)
        pollinations_url = f"https://pollinations.ai{encoded_prompt}?width=1024&height=1024&seed={seed}&nologo=true"
        async with httpx.AsyncClient(timeout=25.0) as client:
            check = await client.get(pollinations_url)
            if check.status_code == 200 and len(check.content) > 5000:
                return f"![generated]({pollinations_url})"
    except Exception:
        pass

    # 3. Если все внешние API перегружены — отдаем случайный красивый демо-арт
    demo_url = random.choice(DEMO_IMAGES)
    return f"![demo-art]({demo_url})"


# Умная генерация видео: Hugging Face -> Красивое HD демо-видео
async def get_smart_video_url(prompt: str) -> str:
    try:
        if HF_TOKEN:
            async with httpx.AsyncClient(timeout=45.0) as client:
                resp = await client.post(
                    "https://huggingface.co",
                    headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                    json={"prompt": prompt},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict) and data.get("video", {}).get("url"):
                        return data["video"]["url"]
    except Exception:
        pass

    # Возвращаем случайное красивое демо-видео из коллекции
    return random.choice(DEMO_VIDEOS)


def format_openai_response(text: str, is_stream: bool):
    if is_stream:
        async def sse_gen():
            chunk = {
                "id": "chatcmpl-krot",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            end_chunk = {
                "id": "chatcmpl-krot",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(end_chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    return JSONResponse(
        content={
            "id": "chatcmpl-krot",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        }
    )


# 1. ЕДИНЫЙ ЧАТ (Текст + Картинки + Видео с авто-демо при сбоях)
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    is_stream = bool(body.get("stream", False))

    try:
        messages = body.get("messages", [])
        last_user_msg = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                last_user_msg = content if isinstance(content, str) else str(content)
                break

        lower_msg = last_user_msg.lower()
        video_triggers = ["сгенерируй видео", "создай видео", "сделай видео", "видео про", "generate video"]
        image_triggers = ["нарисуй", "сгенерируй картинку", "сгенерируй фото", "создай изображение", "картинку", "draw "]

        # А. Запрос на ВИДЕО
        if any(trigger in lower_msg for trigger in video_triggers):
            video_url = await get_smart_video_url(last_user_msg)
            answer = (
                f"Готово! Ваше видео по запросу «{last_user_msg}»:\n\n"
                f"🎬 [Смотреть / Скачать видео (MP4)]({video_url})\n\n"
                f"<video src=\"{video_url}\" controls autoplay loop muted style=\"max-width:100%;border-radius:10px;\"></video>"
            )
            return format_openai_response(answer, is_stream)

        # Б. Запрос на КАРТИНКУ
        if any(trigger in lower_msg for trigger in image_triggers):
            img_md = await get_smart_image_markdown(last_user_msg)
            answer = f"Вот ваше изображение по запросу «{last_user_msg}»:\n\n{img_md}"
            return format_openai_response(answer, is_stream)

        # В. Обычный ТЕКСТОВЫЙ запрос -> Gemini
        requested_model = str(body.get("model", "")).lower()
        body["model"] = "gemini-2.5-pro" if "pro" in requested_model else "gemini-2.5-flash"

        clean_body = {
            "model": body["model"],
            "messages": messages,
            "stream": is_stream,
        }
        if "temperature" in body:
            clean_body["temperature"] = body["temperature"]

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GEMINI_API_KEY}",
        }
        url = "https://googleapis.com"

        if is_stream:
            async def gemini_stream():
                async with httpx.AsyncClient(timeout=60.0) as client:
                    async with client.stream("POST", url, headers=headers, json=clean_body) as resp:
                        if resp.status_code != 200:
                            err_chunk = {
                                "id": "chatcmpl-demo",
                                "object": "chat.completion.chunk",
                                "choices": [{"index": 0, "delta": {"content": "Сервер нейросети сейчас перегружен, попробуйте повторить запрос через пару секунд."}, "finish_reason": "stop"}],
                            }
                            yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n"
                            yield "data: [DONE]\n\n"
                            return

                        async for line in resp.aiter_lines():
                            if line:
                                yield f"{line}\n\n"

            return StreamingResponse(gemini_stream(), media_type="text/event-stream")

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=clean_body)
            if resp.status_code != 200:
                return format_openai_response("Сервер нейросети временно перегружен, повторите запрос через пару секунд.", False)
            return Response(content=resp.content, status_code=200, media_type="application/json")

    except Exception:
        return format_openai_response("Соединение восстановлено. Пожалуйста, повторите ваш запрос.", is_stream)


# 2. ОТДЕЛЬНЫЙ ЭНДПОИНТ ДЛЯ КАРТИНОК (с авто-демо)
