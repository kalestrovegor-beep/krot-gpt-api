import os
import base64
import asyncio
import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse

app = FastAPI()

# Настройка CORS для работы с вашим сайтом на Tilda без блокировок браузера
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")


# Проверка работоспособности сервера при открытии ссылки в браузере
@app.get("/")
async def root():
    return {"status": "ok", "service": "KROT GPT API Gateway"}


# Список моделей для OpenAI-совместимых интерфейсов
@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gemini-2.5-flash", "object": "model"},
            {"id": "gemini-2.5-pro", "object": "model"},
            {"id": "flux-1-schnell", "object": "model"},
        ],
    }


# 1. ТЕКСТОВЫЙ ЧАТ (поддерживает обычный и потоковый stream режим)
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    requested_model = str(body.get("model", "")).lower()

    # Подменяем названия моделей из меню сайта на рабочие модели Gemini
    if "pro" in requested_model:
        body["model"] = "gemini-2.5-pro"
    else:
        body["model"] = "gemini-2.5-flash"

    # Удаляем параметры OpenAI, которые не принимает Gemini
    for unsupported in ["frequency_penalty", "presence_penalty", "logit_bias", "user", "reasoning_effort"]:
        body.pop(unsupported, None)

    is_stream = body.get("stream", False)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}",
    }
    url = "https://googleapis.com"

    if is_stream:
        async def event_stream():
            async with httpx.AsyncClient(timeout=60.0) as client:
                async with client.stream("POST", url, headers=headers, json=body) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, headers=headers, json=body)
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


# 2. ГЕНЕРАЦИЯ КАРТИНОК (Hugging Face Flux-1 Schnell)
@app.post("/v1/images/generations")
@app.post("/images/generations")
async def image_generations(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://huggingface.co",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt},
        )
        if resp.status_code != 200:
            return JSONResponse(status_code=resp.status_code, content={"error": resp.text})

        b64_img = base64.b64encode(resp.content).decode("utf-8")
        return {"data": [{"b64_json": b64_img}]}


# 3. ГЕНЕРАЦИЯ ВИДЕО (Hugging Face Text-to-Video)
@app.post("/v1/video/generations")
@app.post("/video/generations")
async def video_generations(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            "https://huggingface.co",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt},
        )
        if resp.status_code != 200:
            return JSONResponse(status_code=resp.status_code, content={"error": resp.text})

        return Response(content=resp.content, media_type="video/mp4")


# 4. ЖИВОЙ РАЗГОВОР ПО ВИДЕО И ГОЛОСУ (WebSocket -> Gemini Live API)
@app.websocket("/v1/live")
@app.websocket("/live")
async def live_video_call(client_ws: WebSocket):
    await client_ws.accept()
    gemini_url = (
        "wss://://googleapis.com"
        f"google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={GEMINI_API_KEY}"
    )

    try:
        async with websockets.connect(gemini_url) as gemini_ws:
            async def client_to_gemini():
                try:
                    while True:
                        msg = await client_ws.receive_text()
                        await gemini_ws.send(msg)
                except WebSocketDisconnect:
                    pass

            async def gemini_to_client():
                try:
                    async for msg in gemini_ws:
                        if isinstance(msg, bytes):
                            await client_ws.send_text(msg.decode("utf-8"))
                        else:
                            await client_ws.send_text(msg)
                except Exception:
                    pass

            await asyncio.gather(client_to_gemini(), gemini_to_client())
    except Exception as e:
        await client_ws.close(code=1011, reason=str(e))
