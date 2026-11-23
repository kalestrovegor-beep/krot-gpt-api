import os
import base64
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
            {"id": "flux-1-schnell", "object": "model"},
        ],
    }


# Вспомогательная функция генерации картинки через Hugging Face
async def generate_flux_image(prompt: str) -> str:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            "https://huggingface.co",
            headers={"Authorization": f"Bearer {HF_TOKEN}"},
            json={"inputs": prompt},
        )
        if resp.status_code != 200:
            raise Exception(f"HF Error {resp.status_code}: {resp.text}")
        return base64.b64encode(resp.content).decode("utf-8")


# 1. ЕДИНЫЙ ЧАТ (Текст + Авто-генерация картинок по словам "нарисуй" / "сгенерируй фото")
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    last_user_msg = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            last_user_msg = content if isinstance(content, str) else str(content)
            break

    # Проверяем, просит ли пользователь нарисовать картинку
    lower_msg = last_user_msg.lower()
    image_triggers = ["нарисуй", "сгенерируй картинку", "сгенерируй фото", "создай изображение", "draw "]
    if any(trigger in lower_msg for trigger in image_triggers):
        try:
            # Переводим промпт на английский через Gemini для лучшего качества Flux
            async with httpx.AsyncClient(timeout=30.0) as client:
                tr_resp = await client.post(
                    "https://googleapis.com",
                    headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
                    json={
                        "model": "gemini-2.5-flash",
                        "messages": [
                            {"role": "system", "content": "Translate image prompt to English. Return ONLY the English prompt, nothing else."},
                            {"role": "user", "content": last_user_msg},
                        ],
                    },
                )
                en_prompt = tr_resp.json()["choices"][0]["message"]["content"].strip()

            b64_img = await generate_flux_image(en_prompt)
            md_answer = f"Вот ваше изображение по запросу «{last_user_msg}»:\n\n![generated](data:image/png;base64,{b64_img})"

            return {
                "id": "chatcmpl-img",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": md_answer}, "finish_reason": "stop"}],
            }
        except Exception as e:
            return {
                "id": "chatcmpl-err",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": f"Ошибка генерации картинки: {str(e)}"}, "finish_reason": "stop"}],
            }

    # Обычный текстовый запрос -> направляем в Gemini
    requested_model = str(body.get("model", "")).lower()
    body["model"] = "gemini-2.5-pro" if "pro" in requested_model else "gemini-2.5-flash"

    # Отключаем stream на стороне прокси, чтобы избежать пустых чанков в интерфейсе Tilda
    body["stream"] = False
    for unsupported in ["frequency_penalty", "presence_penalty", "logit_bias", "user", "reasoning_effort", "stream_options"]:
        body.pop(unsupported, None)

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


# 2. ОТДЕЛЬНЫЙ ЭНДПОИНТ ДЛЯ КАРТИНОК
@app.post("/v1/images/generations")
@app.post("/images/generations")
async def image_generations(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    try:
        b64_img = await generate_flux_image(prompt)
        return {"data": [{"b64_json": b64_img}]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# 3. ОТДЕЛЬНЫЙ ЭНДПОИНТ ДЛЯ ВИДЕО
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
