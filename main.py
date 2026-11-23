import asyncio
import json
import os
import time
import uuid
import urllib.parse
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, Response, Request
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="KROT GPT API", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

video_tasks: Dict[str, Dict[str, Any]] = {}
image_cache: Dict[str, tuple] = {}
img_semaphore = asyncio.Semaphore(2)

class Message(BaseModel):
    role: str
    content: Any

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "openai"
    messages: List[Message]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "768x768"
    model: Optional[str] = "flux"

class VideoGenerationRequest(BaseModel):
    prompt: str

def first_item(items: Any, default: Any = None) -> Any:
    return next(iter(items)) if isinstance(items, (list, tuple)) and len(items) > 0 else default

def last_item(items: Any, default: Any = None) -> Any:
    return next(reversed(items)) if isinstance(items, (list, tuple)) and len(items) > 0 else default

@app.get("/")
@app.get("/v1")
async def health_check(request: Request):
    return {
        "status": "online",
        "service": "KROT GPT API v4.0",
        "base_url": str(request.base_url).rstrip("/"),
        "hf_token": bool(os.getenv("HF_TOKEN")),
        "openrouter_key": bool(os.getenv("OPENROUTER_API_KEY"))
    }

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {"object": "list", "data": [{"id": "krot-gpt-text", "object": "model"}, {"id": "krot-flux-image", "object": "model"}, {"id": "krot-wan-video", "object": "model"}]}

def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "text")
    return str(content)

def make_img_url(base_url: str, prompt: str, w: int = 768, h: int = 768) -> str:
    return f"{base_url}/v1/image-proxy?prompt={urllib.parse.quote(prompt)}&width={w}&height={h}&seed={uuid.uuid4().int % 999999}"

async def ask_llm(messages: List[Dict[str, str]]) -> str:
    or_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    async with httpx.AsyncClient(timeout=40.0) as client:
        if or_key:
            for or_model in ("meta-llama/llama-3.2-3b-instruct:free", "google/gemma-2-9b-it:free", "openrouter/free"):
                try:
                    r = await client.post(
                        "https://openrouter.ai",
                        headers={"Authorization": f"Bearer {or_key}", "Content-Type": "application/json"},
                        json={"model": or_model, "messages": messages}
                    )
                    if r.status_code == 200:
                        c = first_item(r.json().get("choices"), {}).get("message", {}).get("content", "").strip()
                        if c:
                            return c
                except Exception:
                    pass

        host = "https://text." + "pollinations" + ".ai"
        for m in ("openai", "mistral", "llama"):
            try:
                r = await client.post(f"{host}/openai", json={"model": m, "messages": messages})
                if r.status_code == 200:
                    c = first_item(r.json().get("choices"), {}).get("message", {}).get("content", "").strip()
                    if c:
                        return c
            except Exception:
                pass
        try:
            q = urllib.parse.quote(last_item(messages, {}).get("content", "Привет"))
            r = await client.get(f"{host}/{q}")
            if r.status_code == 200 and r.text.strip():
                return r.text.strip()
        except Exception:
            pass
    return "Нейросеть сейчас перегружена, попробуйте повторить запрос."

def generate_via_hf_space(prompt: str, seed: int, w: int, h: int) -> bytes:
    from gradio_client import Client
    hf = os.getenv("HF_TOKEN", "").strip()
    client = Client("black-forest-labs/FLUX.1-schnell", token=hf if hf else None)
    res = client.predict(prompt=prompt, seed=seed, randomize_seed=False, width=w, height=h, num_inference_steps=4, api_name="/infer")
    img_path = first_item(res, res) if isinstance(res, (list, tuple)) else res
    if isinstance(img_path, dict):
        img_path = img_path.get("path") or img_path.get("url")
    with open(str(img_path), "rb") as f:
        return f.read()

@app.get("/v1/image-proxy")
async def image_proxy(prompt: str, width: int = 768, height: int = 768, seed: int = 42):
    cache_key = f"{prompt}_{width}_{height}_{seed}"
    if cache_key in image_cache:
        data, mtype = image_cache[cache_key]
        return Response(content=data, media_type=mtype)

    async with img_semaphore:
        if cache_key in image_cache:
            data, mtype = image_cache[cache_key]
            return Response(content=data, media_type=mtype)

        hf = os.getenv("HF_TOKEN", "").strip()
        async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
            if hf:
                try:
                    r = await client.post(
                        "https://huggingface.co",
                        headers={"Authorization": f"Bearer {hf}"},
                        json={"inputs": prompt}
                    )
                    if r.status_code == 200 and r.headers.get("content-type", "").startswith("image/"):
                        mtype = r.headers["content-type"]
                        image_cache[cache_key] = (r.content, mtype)
                        return Response(content=r.content, media_type=mtype)
                except Exception:
                    pass

            try:
                img_bytes = await asyncio.to_thread(generate_via_hf_space, prompt, seed, width, height)
                image_cache[cache_key] = (img_bytes, "image/webp")
                return Response(content=img_bytes, media_type="image/webp")
            except Exception:
                pass

            try:
                w = max(512, min(768, (width // 64) * 64))
                h = max(512, min(768, (height // 64) * 64))
                sub = await client.post(
                    "https://stablehorde.net",
                    headers={"apikey": "0000000000", "Client-Agent": "KrotGPT:4.0:contact@krot.ai"},
                    json={"prompt": prompt, "models": ["Flux.1-Schnell fp8"], "params": {"width": w, "height": h, "steps": 4, "n": 1}, "nsfw": False, "r2": True}
                )
                if sub.status_code == 202:
                    jid = sub.json().get("id")
                    for _ in range(25):
                        await asyncio.sleep(3)
                        chk = await client.get(f"https://stablehorde.net{jid}", headers={"Client-Agent": "KrotGPT:4.0:contact@krot.ai"})
                        if chk.status_code == 200 and chk.json().get("done"):
                            url = first_item(chk.json().get("generations"), {}).get("img")
                            if url:
                                img = await client.get(url)
                                if img.status_code == 200:
                                    mtype = img.headers.get("content-type", "image/webp")
                                    image_cache[cache_key] = (img.content, mtype)
                                    return Response(content=img.content, media_type=mtype)
                            break
            except Exception:
                pass

    raise HTTPException(status_code=502, detail="Генераторы картинок сейчас заняты. Повторите запрос через 10 секунд.")

def run_video(task_id: str, prompt: str, base_url: str):
    from gradio_client import Client
    hf = os.getenv("HF_TOKEN", "").strip()
    token_arg = hf if hf else None
    last_err = ""
    video_spaces = [
        ("Wan-AI/Wan2.1", "/t2v_generation"),
        ("Lightricks/ltx-video-distilled", "/generate"),
        ("zai-org/CogVideoX-5B-Space", "/generate")
    ]
    for space_name, endpoint in video_spaces:
        try:
            client = Client(space_name, token=token_arg)
            if "Wan" in space_name:
                res = client.predict(prompt=prompt, api_name=endpoint)
            else:
                res = client.predict(prompt=prompt, negative_prompt="blurry, low quality", api_name=endpoint)
            raw = first_item(res, res) if isinstance(res, (list, tuple)) else res
            if isinstance(raw, dict):
                raw = raw.get("video") or raw.get("path") or raw.get("url")
            file_path = str(raw)
            if os.path.exists(file_path):
                video_tasks[task_id] = {"status": "completed", "file_path": file_path, "video_url": f"{base_url}/v1/videos/file/{task_id}"}
                return
            elif file_path.startswith("http"):
                video_tasks[task_id] = {"status": "completed", "video_url": file_path}
                return
        except Exception as e:
            last_err = f"{space_name}: {str(e)[:140]}"
    video_tasks[task_id] = {"status": "failed", "error": f"Все видео-серверы сейчас заняты. Детали: {last_err}"}

@app.get("/v1/videos/file/{task_id}")
async def get_video_file(task_id: str):
    task = video_tasks.get(task_id)
    if not task or not task.get("file_path") or not os.path.exists(task["file_path"]):
        raise HTTPException(status_code=404, detail="Видеофайл не найден")
    return FileResponse(task["file_path"], media_type="video/mp4")
