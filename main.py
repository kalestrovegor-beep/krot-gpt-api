import asyncio
import json
import os
import time
import uuid
import urllib.parse
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, Response, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="KROT GPT API", version="3.2")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

video_tasks: Dict[str, Dict[str, Any]] = {}

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
        "service": "KROT GPT API v3.2",
        "base_url": str(request.base_url).rstrip("/"),
        "hf_token": bool(os.getenv("HF_TOKEN"))
    }

@app.get("/v1/models")
async def list_models():
    return {"object": "list", "data": [{"id": "krot-gpt-text", "object": "model"}, {"id": "krot-flux-image", "object": "model"}, {"id": "krot-ltx-video", "object": "model"}]}

def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "text")
    return str(content)

def make_img_url(base_url: str, prompt: str, w: int = 768, h: int = 768) -> str:
    return f"{base_url}/v1/image-proxy?prompt={urllib.parse.quote(prompt)}&width={w}&height={h}&seed={uuid.uuid4().int % 999999}"

async def ask_llm(messages: List[Dict[str, str]]) -> str:
    host = "https://text." + "pollinations" + ".ai"
    async with httpx.AsyncClient(timeout=40.0) as client:
        for m in ("openai", "mistral", "llama"):
            try:
                r = await client.post(f"{host}/openai", json={"model": m, "messages": messages})
                if r.status_code == 200:
                    c = first_item(r.json().get("choices"), {}).get("message", {}).get("content")
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
    return "Сервер перегружен, повторите запрос."

@app.get("/v1/image-proxy")
async def image_proxy(prompt: str, width: int = 768, height: int = 768, seed: int = 42):
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
                    return Response(content=r.content, media_type=r.headers["content-type"])
            except Exception:
                pass
        w, h = max(512, min(768, (width // 64) * 64)), max(512, min(768, (height // 64) * 64))
        sub = await client.post(
            "https://stablehorde.net",
            headers={"apikey": "0000000000", "Client-Agent": "KrotGPT:3.2:krot@local"},
            json={"prompt": prompt, "params": {"width": w, "height": h, "steps": 20, "n": 1}, "nsfw": False, "r2": True}
        )
        if sub.status_code != 202:
            raise HTTPException(status_code=502, detail="Ошибка очереди генерации")
        jid = sub.json().get("id")
        for _ in range(20):
            await asyncio.sleep(3)
            chk = await client.get(f"https://stablehorde.net{jid}", headers={"Client-Agent": "KrotGPT:3.2:krot@local"})
            if chk.status_code == 200 and chk.json().get("done"):
                url = first_item(chk.json().get("generations"), {}).get("img")
                if url:
                    img = await client.get(url)
                    if img.status_code == 200:
                        return Response(content=img.content, media_type=img.headers.get("content-type", "image/webp"))
                break
        raise HTTPException(status_code=504, detail="Таймаут генерации")

def run_video(task_id: str, prompt: str):
    try:
        from gradio_client import Client
        res = Client("Lightricks/LTX-Video").predict(prompt=prompt, negative_prompt="blurry, low quality", api_name="/generate")
        video_tasks[task_id] = {"status": "completed", "video_url": str(first_item(res, res) if isinstance(res, (list, tuple)) else res)}
    except Exception as e:
        video_tasks[task_id] = {"status": "failed", "error": str(e)}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request, bg: BackgroundTasks):
    base_url = str(request.base_url).rstrip("/")
    if base_url.startswith("http://"):
        base_url = "https://" + base_url[7:]

    msgs = [{"role": m.role, "content": extract_text(m.content)} for m in req.messages]
    text = last_item(msgs, {}).get("content", "").strip()
    low = text.lower()

    if low.startswith(("нарисуй", "сгенерируй картинку", "создай картинку", "draw ")):
        clean = last_item(text.split(" ", 1), text)
        en = await ask_llm([{"role": "system", "content": "Translate to English FLUX prompt. Output ONLY English words."}, {"role": "user", "content": clean}])
        url = make_img_url(base_url, en if "перегружен" not in en else clean)
        reply = f"Изображение FLUX по запросу «{clean}»:\n\n![Изображение]({url})\n\n[Открыть в полном размере]({url})"
    elif low.startswith(("создай видео", "сгенерируй видео", "сделай видео", "video ")):
        clean = last_item(text.split(" ", 1), text)
        en = await ask_llm([{"role": "system", "content": "Translate to short English video prompt. Output ONLY English words."}, {"role": "user", "content": clean}])
        tid = str(uuid.uuid4())
        video_tasks[tid] = {"status": "processing"}
        bg.add_task(run_video, tid, en if "перегружен" not in en else clean)
        reply = f"Видео запущено по запросу «{clean}». Проверьте готовность через 1–2 мин:\n[Статус видео]({base_url}/v1/videos/status/{tid})"
    else:
        reply = await ask_llm(msgs)

    if req.stream:
        async def gen():
            yield f"data: {json.dumps({'id': 'chatcmpl-1', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'krot', 'choices': [{'index': 0, 'delta': {'content': reply}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'id': 'chatcmpl-1', 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': 'krot', 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream")
    return {"id": "chatcmpl-1", "object": "chat.completion", "created": int(time.time()), "model": "krot", "choices": [{"index": 0, "message": {"role": "assistant", "content": reply}, "finish_reason": "stop"}]}

@app.post("/v1/images/generations")
async def gen_image(req: ImageGenerationRequest, request: Request):
    base_url = str(request.base_url).rstrip("/")
    if base_url.startswith("http://"):
        base_url = "https://" + base_url[7:]
    return {"created": int(time.time()), "data": [{"url": make_img_url(base_url, req.prompt)}]}

@app.post("/v1/videos/generations")
async def gen_video(req: VideoGenerationRequest, request: Request, bg: BackgroundTasks):
    base_url = str(request.base_url).rstrip("/")
    if base_url.startswith("http://"):
        base_url = "https://" + base_url[7:]
    tid = str(uuid.uuid4())
    video_tasks[tid] = {"status": "processing"}
    bg.add_task(run_video, tid, req.prompt)
    return {"task_id": tid, "status": "processing", "check_url": f"{base_url}/v1/videos/status/{tid}"}

@app.get("/v1/videos/status/{task_id}")
async def video_status(task_id: str):
    if task_id not in video_tasks:
        raise HTTPException(status_code=404, detail="Not found")
    return video_tasks[task_id]
