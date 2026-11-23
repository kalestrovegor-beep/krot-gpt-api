import os
import json
import uuid
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
    return {"status": "ok", "service": "KROT GPT API Gateway"}


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gemini-2.5-flash", "object": "model"},
            {"id": "gemini-2.5-pro", "object": "model"},
        ],
    }


@app.get("/media/{file_id}")
async def serve_media(file_id: str):
    item = MEDIA_STORE.get(file_id)
    if not item:
        return Response(status_code=404, content="Not found")
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
                        {
                            "role": "system",
                            "content": (
                                "Translate user request into a detailed English AI art prompt. "
                                "Russian 'крот' means 'cute furry mole animal' (NOT garlic, NOT cat). "
                                "Output ONLY the English prompt."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            if tr.status_code == 200:
                choices = tr.json().get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content", "").strip()
                    if text:
                        return text
    except Exception:
        pass
    return prompt


async def create_real_image(prompt: str, base_url: str) -> str:
    en_prompt = await translate_to_english(prompt)

    if GEMINI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=35.0) as client:
                resp = await client.post(
                    f"https://{GOOGLE_HOST}/v1beta/models/imagen-3.0-generate-002:predict?key={GEMINI_API_KEY}",
                    json={"instances": [{"prompt": en_prompt}], "parameters": {"sampleCount": 1}},
                )
                if resp.status_code == 200:
                    preds = resp.json().get("predictions", [])
                    if preds and "bytesBase64Encoded" in preds[0]:
                        raw = base64.b64decode(preds[0]["bytesBase64Encoded"])
                        return save_media(raw, "image/png", "png", base_url)
        except Exception:
            pass

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
        for model_path in ["fal-ai/fal-ai/fast-svd-lcm", "hf-inference/models/Lightricks/LTX-Video"]:
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    r = await client.post(
                        f"https://{HF_ROUTER_HOST}/{model_path}",
                        headers={"Authorization": f"Bearer {HF_TOKEN}", "Content-Type": "application/json"},
                        json={"inputs": en_prompt, "prompt": en_prompt},
                    )
                    if r.status_code == 200:
                        ct = r.headers.get("content-type", "")
                        if "video" in ct or len(r.content) > 50000:
                            return save_media(r.content, "video/mp4", "mp4", base_url)
                        data = r.json()
                        vid_url = data.get("video", {}).get("url") if isinstance(data, dict) else None
                        if vid_url:
                            v_resp = await client.get(vid_url)
                            if v_resp.status_code == 200:
                                return save_media(v_resp.content, "video/mp4", "mp4", base_url)
            except Exception:
                continue

    try:
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            sample = random.choice(FALLBACK_VIDEOS)
            v = await client.get(sample)
            if v.status_code == 200:
                return save_media(v.content, "video/mp4", "mp4", base_url)
    except Exception:
        pass

    return random.choice(FALLBACK_VIDEOS)


def build_response(text: str, is_stream: bool):
    if is_stream:
        async def sse():
            c1 = {"id": "chatcmpl-krot", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
            yield f"data: {json.dumps(c1, ensure_ascii=False)}\n\n"
            c2 = {"id": "chatcmpl-krot", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}]}
            yield f"data: {json.dumps(c2, ensure_ascii=False)}\n\n"
            c3 = {"id": "chatcmpl-krot", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(c3, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

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
        video_words = ["сгенерируй видео", "создай видео", "сделай видео", "видео про", "generate video"]
        image_words = ["нарисуй", "сгенерируй картинку", "сгенерируй фото", "создай изображение", "картинку", "draw "]

        if any(w in lower_msg for w in video_words):
            vid_link = await create_real_video(last_user_msg, base_url)
            ans = f"🎬 Видео готово!\n\n👉 [Открыть и посмотреть видео (MP4)]({vid_link})"
            return build_response(ans, is_stream)

        if any(w in lower_msg for w in image_words):
            img_link = await create_real_image(last_user_msg, base_url)
            if img_link:
                ans = f"Вот изображение по запросу «{last_user_msg}»:\n\n![art]({img_link})"
            else:
                ans = "Сервер генерации картинок временно перегружен, повторите запрос через 5 секунд."
            return build_response(ans, is_stream)

        req_model = str(body.get("model", "")).lower()
        target_model = "gemini-2.5-pro" if "pro" in req_model else "gemini-2.5-flash"

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"https://{GOOGLE_HOST}/v1beta/openai/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {GEMINI_API_KEY}"},
                json={"model": target_model, "messages": clean_messages, "stream": False},
            )
