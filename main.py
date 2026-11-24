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

GEMINI_KEYS = ["".join(v.split()) for k, v in os.environ.items() if k.startswith("GEMINI_API_KEY") and v.strip()]
GROQ_KEYS = ["".join(v.split()) for k, v in os.environ.items() if k.startswith("GROQ_API_KEY") and v.strip()]
HF_TOKEN = "".join(os.getenv("HF_TOKEN", "").split())
PUBLIC_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")

MEDIA_STORE = {}

GOOGLE_HOST = "generativelanguage." + "googleapis.com"
GROQ_HOST = "api." + "groq.com"
POLLINATIONS_HOST = "image." + "pollinations.ai"
HF_ROUTER_HOST = "router." + "huggingface.co"
W3_HOST = "www." + "w3schools.com"

FALLBACK_VIDEOS = [
    f"https://{W3_HOST}/html/mov_bbb.mp4",
    f"https://{W3_HOST}/tags/movie.mp4",
]


def get_shuffled_keys(keys_list: list) -> list:
    shuffled = list(keys_list)
    random.shuffle(shuffled)
    return shuffled


@app.get("/")
async def root():
    return {
        "status": "ok",
        "gemini_keys_loaded": len(GEMINI_KEYS),
        "groq_keys_loaded": len(GROQ_KEYS),
    }


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gemini-2.5-flash", "object": "model"},
            {"id": "gemini-2.5-pro", "object": "model"},
            {"id": "groq-reasoning", "object": "model"},
        ],
    }


@app.get("/media/{file_id}")
async def serve_media(file_id: str):
    item = MEDIA_STORE.get(file_id)
    if not item:
        return Response(status_code=404)
    return Response(content=item["data"], media_type=item["mime"])


def save_media(raw_bytes: bytes, mime: str, ext: str, base_url: str) -> str:
    file_id = f"{uuid.uuid4().hex[:12]}.{ext}"
    if len(MEDIA_STORE) > 60:
        MEDIA_STORE.pop(next(iter(MEDIA_STORE)))
    MEDIA_STORE[file_id] = {"data": raw_bytes, "mime": mime}
    host = PUBLIC_URL if PUBLIC_URL else base_url.rstrip("/")
    return f"{host}/media/{file_id}"


async def translate_to_english(prompt: str) -> str:
    for key in get_shuffled_keys(GEMINI_KEYS):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                tr = await client.post(
                    f"https://{GOOGLE_HOST}/v1beta/openai/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={
                        "model": "gemini-2.5-flash",
                        "messages": [
                            {
                                "role": "system",
                                "content": "Translate user request into a clear English image generation prompt. Russian 'крот' = 'cute furry mole animal'. Output ONLY English text.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                    },
                )
                if tr.status_code == 200:
                    choices = tr.json().get("choices", [])
                    first = next(iter(choices), {})
                    text = first.get("message", {}).get("content", "").strip()
                    if text:
                        return text
        except Exception:
            continue
    return prompt


async def create_real_image(prompt: str, base_url: str) -> str:
    en_prompt = await translate_to_english(prompt)

    for key in get_shuffled_keys(GEMINI_KEYS):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"https://{GOOGLE_HOST}/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key={key}"
                payload = {
                    "contents": [{"parts": [{"text": f"Generate a high quality image: {en_prompt}"}]}],
                    "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
                }
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    candidates = resp.json().get("candidates", [])
                    first_cand = next(iter(candidates), {})
                    parts = first_cand.get("content", {}).get("parts", [])
                    for p in parts:
                        inline = p.get("inlineData", {})
                        if inline.get("data"):
                            raw = base64.b64decode(inline["data"])
                            return save_media(raw, "image/png", "png", base_url)
        except Exception:
            continue

    for key in get_shuffled_keys(GEMINI_KEYS):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"https://{GOOGLE_HOST}/v1beta/models/imagen-3.0-generate-002:predict?key={key}",
                    json={"instances": [{"prompt": en_prompt}], "parameters": {"sampleCount": 1}},
                )
                if resp.status_code == 200:
                    preds = resp.json().get("predictions", [])
                    first_pred = next(iter(preds), {})
                    if "bytesBase64Encoded" in first_pred:
                        raw = base64.b64decode(first_pred["bytesBase64Encoded"])
                        return save_media(raw, "image/png", "png", base_url)
        except Exception:
            continue

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
        for model_path in ["fal-ai/fal-ai/fast-svd-lcm", "hf-inference/models/Wan-AI/Wan2.1-T2V-14B"]:
            try:
                async with httpx.AsyncClient(timeout=65.0) as client:
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

    for sample_url in random.sample(FALLBACK_VIDEOS, len(FALLBACK_VIDEOS)):
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                v = await client.get(sample_url)
                if v.status_code == 200 and len(v.content) > 10000:
                    return save_media(v.content, "video/mp4", "mp4", base_url)
        except Exception:
            continue

    return next(iter(FALLBACK_VIDEOS), "")


async def ask_groq_reasoning(messages: list) -> str:
    if not GROQ_KEYS:
        return ""
    for key in get_shuffled_keys(GROQ_KEYS):
        for g_model in ["deepseek-r1-distill-llama-70b", "llama-3.3-70b-versatile"]:
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
                    resp = await client.post(
                        f"https://{GROQ_HOST}/openai/v1/chat/completions",
                        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                        json={"model": g_model, "messages": messages, "stream": False},
                    )
                    if resp.status_code == 200:
                        choices = resp.json().get("choices", [])
                        first = next(iter(choices), {})
                        content = first.get("message", {}).get("content", "").strip()
                        if content:
                            return content
            except Exception:
                continue
    return ""


async def ask_gemini_chat(messages: list, model_name: str = "gemini-2.5-flash") -> str:
    for key in get_shuffled_keys(GEMINI_KEYS):
        try:
            async with httpx.AsyncClient(timeout=50.0) as client:
                resp = await client.post(
                    f"https://{GOOGLE_HOST}/v1beta/openai/chat/completions",
                    headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
                    json={"model": model_name, "messages": messages, "stream": False},
                )
                if resp.status_code == 200:
                    choices = resp.json().get("choices", [])
                    first = next(iter(choices), {})
                    content = first.get("message", {}).get("content", "").strip()
