import os
import base64
import asyncio
import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

app = FastAPI()

# Разрешаем запросы с вашего сайта (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Позже замените * на https://ваш-сайт.ru
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")


# 1. ТЕКСТОВЫЙ ЧАТ (OpenAI-совместимый формат -> Gemini)
@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    body.setdefault("model", "gemini-2.5-flash")

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://googleapis.com",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GEMINI_API_KEY}",
            },
            json=body,
        )
        return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")


# 2. ГЕНЕРАЦИЯ КАРТИНОК (Hugging Face Flux-1 Schnell -> Base64 JSON)
@app.post("/v1/images/generations")
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
