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

app = FastAPI(title="KROT GPT API", version="4.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

video_tasks: Dict[str, Dict[str, Any]] = {}
image_cache: Dict[str, tuple] = {}
img_semaphore = asyncio.Semaphore(2)

def first_item(items: Any, default: Any = None) -> Any:
    return next(iter(items)) if isinstance(items, (list, tuple)) and len(items) > 0 else default

def last_item(items: Any, default: Any = None) -> Any:
    return next(reversed(items)) if isinstance(items, (list, tuple)) and len(items) > 0 else default

def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") in ("text", "input_text"))
    return str(content)

def strip_prefix(text: str, prefixes: tuple) -> str:
    low = text.lower()
    for p in prefixes:
        if low.startswith(p):
            rest = text[len(p):].strip(" :,-")
            return rest if rest else "cute mole on the moon"
    return text

def make_img_url(base_url: str, prompt: str, w: int = 768, h: int = 768) -> str:
    return f"{base_url}/v1/image-proxy?prompt={urllib.parse.quote(prompt)}&width={w}&height={h}&seed={uuid.uuid4().int % 999999}"

@app.get("/")
@app.get("/v1")
async def health_check(request: Request):
    return {"status": "online", "service": "KROT GPT API v4.2", "hf_token": bool(os.getenv("HF_TOKEN")), "openrouter_key": bool(os.getenv("OPENROUTER_API_KEY"))}

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {"object": "list", "data": [{"id": "krot-gpt-text", "object": "model"}, {"id": "krot-flux-image", "object": "model"}, {"id": "krot-wan-video", "object": "model"}]}

# ГЛАВНЫЙ ОБРАБОТЧИК ЧАТА В САМОМ НАЧАЛЕ ФАЙЛА (НИКОГДА НЕ ОБРЕЖЕТСЯ)
@app.post("/v1/chat/completions")
@app.post("/chat/completions")
@app.api_route("/{full_path:path}", methods=["POST"])
async def universal_chat(request: Request, bg: BackgroundTasks, full_path: str = ""):
    base_url = str(request.base_url).rstrip("/")
    if base_url.startswith("http://"):
        base_url = "https://" + base_url[7:]

    try:
        body = await request.json()
    except Exception:
        body = {}

    if full_path.endswith("images/generations"):
        prompt = body.get("prompt", "cute mole")
        return {"created": int(time.time()), "data": [{"url": make_img_url(base_url, prompt)}]}

    if full_path.endswith("videos/generations"):
        prompt = body.get("prompt", "cute mole on the moon")
        tid = str(uuid.uuid4())
        video_tasks[tid] = {"status": "processing"}
        bg.add_task(run_video, tid, prompt, base_url)
        return {"task_id": tid, "status": "processing", "check_url": f"{base_url}/v1/videos/status/{tid}"}

    raw_msgs = body.get("messages") or body.get("input") or []
    if isinstance(raw_msgs, str):
        msgs = [{"role": "user", "content": raw_msgs}]
    elif isinstance(raw_msgs, list):
        msgs = [{"role": m.get("role", "user"), "content": extract_text(m.get("content", ""))} for m in raw_msgs if isinstance(m, dict)]
    else:
        msgs = [{"role": "user", "content": str(body.get("prompt") or body.get("message") or "Привет")}]

    text = last_item(msgs, {}).get("content", "").strip()
    low = text.lower()
    stream = bool(body.get("stream", False))

    draw_keys = ("нарисуй", "сгенерируй картинку", "создай картинку", "картинка", "draw")
    video_keys = ("создай видео", "создайвидео", "сгенерируй видео", "сделай видео", "видео", "video")

    if low.startswith(draw_keys):
        clean = strip_prefix(text, draw_keys)
        en = await ask_llm([{"role": "system", "content": "Translate to English FLUX prompt. Output ONLY English words."}, {"role": "user", "content": clean}])
        url = make_img_url(base_url, en if "перегружена" not in en else clean)
        reply = f"Изображение FLUX по запросу «{clean}»:\n\n![Изображение]({url})\n\n[Открыть в полном размере]({url})"
    elif low.startswith(video_keys):
        clean = strip_prefix(text, video_keys)
        en = await ask_llm([{"role": "system", "content": "Translate to short English video prompt. Output ONLY English words."}, {"role": "user", "content": clean}])
        tid = str(uuid.uuid4())
        video_tasks[tid] = {"status": "processing"}
        bg.add_task(run_video, tid, en if "перегружена" not in en else clean, base_url)
        reply = f"Видео запущено по запросу «{clean}». Проверьте готовность через 1–2 мин:\n[Статус видео]({base_url}/v1/videos/status/{tid})"
    else:
        reply = await ask_llm(msgs)

    if stream:
        async def gen():
            cid = f"chatcmpl-{uuid.uuid4().hex}"
            ts = int(time.time())
            c1 = {"id": cid, "object": "chat.completion.chunk", "created": ts, "model": "krot", "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]}
            c2 = {"id": cid, "object": "chat.completion.chunk", "created": ts, "model": "krot", "choices": [{"index": 0, "delta": {"content": reply}, "finish_reason": None}]}
            c3 = {"id": cid, "object": "chat.completion.chunk", "created": ts, "model": "krot", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            yield f"data: {json.dumps(c1, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps(c2, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps(c3, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")

    return {"id": "chatcmpl-1", "object": "chat.completion", "created": int(time.time()), "model": "krot", "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]}

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
    return "Нейросеть сейчас перегружена, повторите запрос."

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
