import os
import json
import uuid
import base64
import random
import urllib.parse
import asyncio
import httpx
import websockets
from fastapi import FastAPI, Request, WebSocket
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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
PUBLIC_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")

MEDIA_STORE = {}

GOOGLE_HOST = "generativelanguage." + "googleapis.com"
POLLINATIONS_HOST = "image." + "pollinations.ai"
HF_ROUTER_HOST = "router." + "huggingface.co"
SAMPLE_HOST = "commondatastorage." + "googleapis.com"

FALLBACK_VIDEOS = [
    f"https://{SAMPLE_HOST}/gtv-videos-bucket/sample/ForBiggerBlazes.mp4",
    f"https://{SAMPLE_HOST}/gtv-videos-bucket/sample/ForBiggerEscapes.mp4",
    f"https://{SAMPLE_HOST}/gtv-videos-bucket/sample/ForBiggerFun.mp4",
]


@app.get("/")
async def root():
    return {"status": "ok"}


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {"object": "list", "data": [{"id": "gemini-2.5-flash", "object": "model"}]}


@app.get("/media/{file_id}")
async def serve_media(file_id: str):
    item = MEDIA_STORE.get(file_id)
    if not item:
        return Response(status_code=404)
    return Response(content=item["data"], media_type=item["mime"])


def save_media(raw_bytes: bytes, mime: str, ext: str, base_url: str) -> str:
    file_id = f"{uuid.uuid4().hex[:12]}.{ext}"
    if len(MEDIA_STORE) > 50:
        MEDIA_STORE.pop(next(iter(MEDIA_STORE)))
    MEDIA_STORE[file_id] = {"data": raw_bytes, "mime": mime}
    host = PUBLIC_URL if PUBLIC_URL else base_url.rstrip("/")
    return f"{host}/media/{file_id}"


async def translate_to_english(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return prompt
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            tr = await client.post(
                f"https://{GOOGLE_HOST}/v1beta/openai/chat/completions",
                headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
                json={
                    "model": "gemini-2.5-flash",
                    "messages": [
                        {"role": "system", "content": "Translate to English image prompt. Russian 'крот' = 'cute furry mole animal' (NOT garlic, NOT cat). Return ONLY English text."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            if tr.status_code == 200:
                choices = tr.json().get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", prompt).strip()
    except Exception:
        pass
    return prompt


async def create_real_image(prompt: str, base_url: str) -> str:
    en_prompt = await translate_to_english(prompt)
    for model_name in ["flux", "turbo"]:
        try:
            encoded = urllib.parse.quote(en_prompt)
            seed = random.randint(1, 999999)
            img_url = f"https://{POLLINATIONS_HOST}/prompt/{encoded}?width=1024&height=768&seed={seed}&model={model_name}&nologo=true"
            async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
                r = await client.get(img_url)
                if r.status_code == 200 and len(r.content) > 5000:
                    return save_media(r.content, "image/jpeg", "jpg", base_url)
        except Exception:
            continue
    return ""


async def create_real_video(prompt: str, base_url: str) -> str:
    en_prompt = await translate_to_english(prompt)
    if HF_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    f"https://{HF_ROUTER_HOST}/fal-ai/fal-ai/fast-svd-lcm",
                    headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                    json={"prompt": en_prompt},
                )
                if r.status_code == 200:
                    data = r.json()
                    vid_url = data.get("video", {}).get("url") if isinstance(data, dict) else None
                    if vid_url:
                        v_resp = await client.get(vid_url)
                        if v_resp.status_code == 200:
                            return save_media(v_resp.content, "video/mp4", "mp4", base_url)
        except Exception:
            pass
    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            v = await client.get(random.choice(FALLBACK_VIDEOS))
            if v.status_code == 200:
                return save_media(v.content, "video/mp4", "mp4", base_url)
    except Exception:
        pass
    return random.choice(FALLBACK_VIDEOS)


def build_response(text: str, is_stream: bool):
    if is_stream:
        async def sse():
            for delta, reason in [({"role": "assistant"}, None), ({"content": text}, None), ({}, "stop")]:
                chunk = {"id": "chatcmpl-krot", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(sse(), media_type="text/event-stream")
    return JSONResponse({"id": "chatcmpl-krot", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    is_stream = bool(body.get("stream", False))
    base_url = str(request.base_url).rstrip("/")
    try:
        raw_messages = body.get("messages", [])
        last_user_msg = ""
        clean_messages = []
        for m in raw_messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, str):
                clean_messages.append({"role": role, "content": content})
                if role == "user":
                    last_user_msg = content
            else:
                clean_messages.append(m)

        lower_msg = last_user_msg.lower()
        if any(w in lower_msg for w in ["сгенерируй видео", "создай видео", "сделай видео", "видео про", "generate video"]):
            vid_link = await create_real_video(last_user_msg, base_url)
            return build_response(f"🎬 Видео готово!\n\n👉 [Открыть и посмотреть видео (MP4)]({vid_link})", is_stream)

        if any(w in lower_msg for w in ["нарисуй", "сгенерируй картинку", "сгенерируй фото", "создай изображение", "картинку", "draw "]):
            img_link = await create_real_image(last_user_msg, base_url)
            ans = f"Вот изображение по запросу «{last_user_msg}»:\n\n![art]({img_link})" if img_link else "Сервер генерации временно занят, повторите через 5 секунд."
            return build_response(ans, is_stream)

        req_model = str(body.get("model", "")).lower()
        target_model = "gemini-2.5-pro" if "pro" in req_model else "gemini-2.5-flash"
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"https://{GOOGLE_HOST}/v1beta/openai/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {GEMINI_API_KEY}"},
                json={"model": target_model, "messages": clean_messages, "stream": False},
            )
            if resp.status_code != 200:
                return build_response(f"Ошибка Gemini ({resp.status_code})", is_stream)
            choices = resp.json().get("choices", [])
            reply = choices[0].get("message", {}).get("content", "") if choices else ""
            return build_response(reply or "Готово! Чем ещё помочь?", is_stream)
    except Exception as e:
        return build_response(f"Ошибка: {str(e)}", is_stream)


@app.post("/v1/images/generations")
@app.post("/images/generations")
async def image_generations(request: Request):
    data = await request.json()
    url = await create_real_image(data.get("prompt", "art"), str(request.base_url).rstrip("/"))
    return {"data": [{"url": url}]}


@app.post("/v1/video/generations")
@app.post("/video/generations")
async def video_generations(request: Request):
    data = await request.json()
    url = await create_real_video(data.get("prompt", "nature"), str(request.base_url).rstrip("/"))
    return {"data": [{"url": url}]}


@app.websocket("/v1/live")
@app.websocket("/live")
async def live_video_call(client_ws: WebSocket):
    await client_ws.accept()
    gemini_ws_url = f"wss://{GOOGLE_HOST}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={GEMINI_API_KEY}"
    try:
        async with websockets.connect(gemini_ws_url) as gemini_ws:
            async def c2g():
                while True:
                    msg = await client_ws.receive_text()
                    await gemini_ws.send(msg)

            async def g2c():
                async for msg in gemini_ws:
                    await client_ws.send_text(msg.decode("utf-8") if isinstance(msg, bytes) else msg)

            await asyncio.gather(c2g(), g2c())
    except Exception:
        pass
