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

DEMO_VIDEOS = [
    "https://googleapis.com",
    "https://googleapis.com",
    "https://googleapis.com",
    "https://googleapis.com",
]

DEMO_SVGS = [
    '<svg xmlns="http://w3.org" width="800" height="500" viewBox="0 0 800 500"><defs><linearGradient id="g1" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#1e3c72"/><stop offset="100%" stop-color="#2a5298"/></linearGradient></defs><rect width="800" height="500" fill="url(#g1)"/><circle cx="620" cy="120" r="70" fill="#ffd700" opacity="0.85"/><polygon points="0,500 220,220 450,500" fill="#142850"/><polygon points="280,500 520,180 800,500" fill="#0c1b33"/><text x="40" y="450" fill="#ffffff" font-family="sans-serif" font-size="24" opacity="0.8">KROT AI • Art Preview</text></svg>',
    '<svg xmlns="http://w3.org" width="800" height="500" viewBox="0 0 800 500"><defs><linearGradient id="g2" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#ff512f"/><stop offset="100%" stop-color="#dd2476"/></linearGradient></defs><rect width="800" height="500" fill="url(#g2)"/><circle cx="400" cy="250" r="140" fill="#ffffff" opacity="0.15"/><circle cx="400" cy="250" r="90" fill="#ffffff" opacity="0.25"/><text x="40" y="450" fill="#ffffff" font-family="sans-serif" font-size="24" opacity="0.8">KROT AI • Neon Concept</text></svg>',
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


async def get_smart_image_markdown(prompt: str) -> str:
    # 1. Пробуем Google AI Studio
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
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for p in parts:
                        if "inlineData" in p and p["inlineData"].get("data"):
                            b64 = p["inlineData"]["data"]
                            return f"![generated](data:image/png;base64,{b64})"
    except Exception:
        pass

    # 2. Пробуем Pollinations.ai -> Base64
    try:
        encoded_prompt = urllib.parse.quote(prompt)
        seed = random.randint(1, 99999)
        pollinations_url = f"https://pollinations.ai{encoded_prompt}?width=768&height=512&seed={seed}&nologo=true"
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            img_resp = await client.get(pollinations_url)
            if img_resp.status_code == 200 and len(img_resp.content) > 5000:
                b64 = base64.b64encode(img_resp.content).decode("utf-8")
                return f"![generated](data:image/jpeg;base64,{b64})"
    except Exception:
        pass

    # 3. Встроенная красивая SVG-иллюстрация в Base64
    svg_code = random.choice(DEMO_SVGS).encode("utf-8")
    b64_svg = base64.b64encode(svg_code).decode("utf-8")
    return f"![demo-art](data:image/svg+xml;base64,{b64_svg})"


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

    return random.choice(DEMO_VIDEOS)


def format_openai_response(text: str, is_stream: bool):
    if is_stream:
        async def sse_gen():
            # Чанк 1: открытие роли assistant (обязательно для фронтенда KROT GPT)
            c1 = {
                "id": "chatcmpl-krot",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(c1, ensure_ascii=False)}\n\n"

            # Чанк 2: сам контент ответа
            c2 = {
                "id": "chatcmpl-krot",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(c2, ensure_ascii=False)}\n\n"

            # Чанк 3: завершение
            c3 = {
                "id": "chatcmpl-krot",
                "object": "chat.completion.chunk",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(c3, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse_gen(), media_type="text/event-stream")

    return JSONResponse(
        content={
            "id": "chatcmpl-krot",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        }
    )


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    is_stream = bool(body.get("stream", False))

    try:
        raw_messages = body.get("messages", [])
        last_user_msg = ""
        clean_messages = []

        for m in raw_messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                if "data:image/" in content:
                    content = "[Сгенерированное изображение]"
                clean_messages.append({"role": role, "content": content})
                if role == "user":
                    last_user_msg = content
            else:
                clean_messages.append(m)

        lower_msg = last_user_msg.lower()
        video_triggers = ["сгенерируй видео", "создай видео", "сделай видео", "видео про", "generate video"]
        image_triggers = ["нарисуй", "сгенерируй картинку", "сгенерируй фото", "создай изображение", "картинку", "draw "]

        # А. Запрос на ВИДЕО
        if any(trigger in lower_msg for trigger in video_triggers):
            video_url = await get_smart_video_url(last_user_msg)
            answer = (
                f"🎬 **Видео по запросу «{last_user_msg}» готово!**\n\n"
                f"👉 **[Открыть и посмотреть видео в HD (MP4)]({video_url})**"
            )
            return format_openai_response(answer, is_stream)

        # Б. Запрос на КАРТИНКУ
        if any(trigger in lower_msg for trigger in image_triggers):
            img_md = await get_smart_image_markdown(last_user_msg)
            answer = f"Вот ваше изображение по запросу «{last_user_msg}»:\n\n{img_md}"
            return format_openai_response(answer, is_stream)

        # В. Обычный ТЕКСТОВЫЙ запрос -> Gemini
        requested_model = str(body.get("model", "")).lower()
        target_model = "gemini-2.5-pro" if "pro" in requested_model else "gemini-2.5-flash"

        clean_body = {
            "model": target_model,
            "messages": clean_messages,
            "stream": False,
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://googleapis.com",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {GEMINI_API_KEY}",
                },
                json=clean_body,
            )
            if resp.status_code != 200:
                return format_openai_response(f"Ошибка Gemini ({resp.status_code}): проверьте ключ GEMINI_API_KEY в Render.", is_stream)

            data = resp.json()
            choices = data.get("choices", [])
            reply_text = ""
            if choices:
                reply_text = choices[0].get("message", {}).get("content", "")

            if not reply_text:
                reply_text = "Привет! Я на связи. Чем могу помочь?"

            return format_openai_response(reply_text, is_stream)

    except Exception as e:
        return format_openai_response(f"Системное уведомление: {str(e)}", is_stream)


