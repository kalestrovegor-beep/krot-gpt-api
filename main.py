import os, io, uuid, time, json, base64, asyncio, httpx
from PIL import Image, ImageFilter
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

app = FastAPI(title="KROT GPT Studio API v4.5")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
TASKS: Dict[str, Dict[str, Any]] = {}


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: Optional[str] = "gpt-6-astra"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


class ImageProxyRequest(BaseModel):
    prompt: str
    width: Optional[int] = 1024
    height: Optional[int] = 1024
    upscale_4k: Optional[bool] = True


@app.get("/")
async def root():
    return {"status": "ok", "version": "4.5", "groq": bool(GROQ_API_KEY), "hf": bool(HF_TOKEN)}


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-6-astra", "object": "model", "owned_by": "krot"},
            {"id": "flux-4k", "object": "model", "owned_by": "krot"},
            {"id": "krot-song", "object": "model", "owned_by": "krot"},
            {"id": "krot-video", "object": "model", "owned_by": "krot"},
        ],
    }


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "text")
    return str(content)


async def translate_to_en(text: str) -> str:
    if not any("а" <= c.lower() <= "я" for c in text) or not GROQ_API_KEY:
        return text
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                "https://groq.com" + "/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={
                    "model": "llama-3.1-8b-instant",
                    "messages": [
                        {"role": "system", "content": "Translate to English prompt for AI generation. Return ONLY English text."},
                        {"role": "user", "content": text},
                    ],
                },
            )
            if r.status_code == 200:
                return r.json()["choices"]["message"]["content"].strip()
    except Exception:
        pass
    return text


def upscale_to_4k_b64(raw_bytes: bytes, make_4k: bool = True) -> str:
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    if make_4k:
        w, h = img.size
        scale = 3840 / max(w, h)
        if scale > 1:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
            img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


async def generate_flux_image(prompt: str, make_4k: bool = True) -> Optional[str]:
    en_prompt = await translate_to_en(prompt)
    if HF_TOKEN:
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        models = ["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev"]
        async with httpx.AsyncClient(timeout=60.0) as client:
            for model_id in models:
                url = "https://huggingface.co" + f"/hf-inference/models/{model_id}"
                try:
                    r = await client.post(url, headers=headers, json={"inputs": en_prompt})
                    if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
                        return await asyncio.to_thread(upscale_to_4k_b64, r.content, make_4k)
                except Exception:
                    continue

    def _space_fallback():
        from gradio_client import Client
        client = Client("black-forest-labs/FLUX.1-schnell", hf_token=HF_TOKEN or None)
        res = client.predict(prompt=en_prompt, seed=0, randomize_seed=True, width=1024, height=1024, num_inference_steps=4, api_name="/infer")
        p = res if isinstance(res, (list, tuple)) else res
        if isinstance(p, dict):
            p = p.get("path") or p.get("url")
        if p and os.path.exists(str(p)):
            with open(p, "rb") as f:
                return upscale_to_4k_b64(f.read(), make_4k)
        return None

    try:
        return await asyncio.to_thread(_space_fallback)
    except Exception:
        return None


@app.post("/v1/image-proxy")
@app.post("/v1/images/generations")
async def image_endpoint(req: ImageProxyRequest):
    img_url = await generate_flux_image(req.prompt, req.upscale_4k)
    if not img_url:
        raise HTTPException(status_code=502, detail="Не удалось создать изображение через FLUX.")
    return {"url": img_url, "data": [{"url": img_url}]}


def run_song_sync(tags: str, lyrics: str) -> str:
    from gradio_client import Client
    client = Client("ACE-Step/ACE-Step", hf_token=HF_TOKEN or None)
    res = client.predict(
        audio_duration=90, prompt=tags, lyrics=lyrics, infer_step=40, guidance_scale=15.0,
        scheduler_type="euler", cfg_type="apg", omega_scale=10.0, manual_seeds=None,
        guidance_interval=0.5, guidance_interval_decay=0.0, min_guidance_scale=3.0,
        use_erg_tag=True, use_erg_lyric=False, use_erg_diffusion=True, oss_steps=None,
        guidance_scale_text=0.0, guidance_scale_lyric=0.0, api_name="/__call__",
    )
    p = res if isinstance(res, (list, tuple)) else res
    if isinstance(p, dict):
        p = p.get("path") or p.get("url")
    if p and os.path.exists(str(p)):
        with open(p, "rb") as f:
            return "data:audio/mp3;base64," + base64.b64encode(f.read()).decode("utf-8")
    raise RuntimeError("Аудиофайл не создан")


async def write_song_lyrics(topic: str) -> tuple[str, str]:
    if not GROQ_API_KEY:
        return "pop, synth, vocals", f"[verse]\n{topic}\n[chorus]\n{topic}"
    async with httpx.AsyncClient(timeout=25.0) as client:
        r = await client.post(
            "https://groq.com" + "/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": "Напиши текст песни с тегами [verse], [chorus]. В 1-й строке напиши: STYLE: pop, synth, vocals. Со 2-й строки только текст."},
                    {"role": "user", "content": topic},
                ],
            },
        )
        if r.status_code == 200:
            full = r.json()["choices"]["message"]["content"].strip()
            lines = full.splitlines()
            if lines and lines.upper().startswith("STYLE:"):
                return lines.split(":", 1).strip(), "\n".join(lines[1:]).strip()
            return "pop, synthwave, melodic vocals", full
    return "pop, synth, vocals", topic


async def song_worker(task_id: str, topic: str):
    TASKS[task_id]["status"] = "processing"
    try:
        tags, lyrics = await write_song_lyrics(topic)
        TASKS[task_id]["lyrics"] = lyrics
        url = await asyncio.to_thread(run_song_sync, tags, lyrics)
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["media_url"] = url
    except Exception as e:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(e)


def run_video_sync(en_prompt: str) -> str:
    from gradio_client import Client
    spaces = [
        ("Lightricks/ltx-video-distilled", "/text_to_video", {"prompt": en_prompt, "negative_prompt": "low quality", "height": 512, "width": 704, "num_frames": 65, "seed": 42}),
        ("Wan-AI/Wan2.1", "/t2v_generation_async", {"prompt": en_prompt, "size": "832*480", "watermark_flag": False, "seed": -1}),
    ]
    errors = []
    for space_id, api_name, kwargs in spaces:
        try:
            client = Client(space_id, hf_token=HF_TOKEN or None)
            res = client.predict(**kwargs, api_name=api_name)
            p = res if isinstance(res, (list, tuple)) else res
            if isinstance(p, dict):
                p = p.get("video") or p.get("path") or p.get("url")
            if p and os.path.exists(str(p)):
                with open(p, "rb") as f:
                    return "data:video/mp4;base64," + base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            errors.append(f"{space_id}: {str(e)}")
    raise RuntimeError(" | ".join(errors))


async def video_worker(task_id: str, prompt: str):
    TASKS[task_id]["status"] = "processing"
    try:
        en_prompt = await translate_to_en(prompt)
        url = await asyncio.to_thread(run_video_sync, en_prompt)
        TASKS[task_id]["status"] = "completed"
        TASKS[task_id]["media_url"] = url
    except Exception as e:
        TASKS[task_id]["status"] = "failed"
        TASKS[task_id]["error"] = str(e)


def format_task_reply(tid: str) -> str:
    task = TASKS.get(tid)
    if task is None:
        return f"Задача `{tid}` не найдена."
    st = task.get("status")
    if st == "completed":
        media = task.get("media_url", "")
        if task.get("type") == "song":
            return f"🎵 **Песня готова!**\n\n[Слушать или скачать MP3]({media})\n\n**Текст:**\n{task.get('lyrics', '')}"
        return f"🎬 **Видео готово!**\n\n[Открыть или скачать MP4]({media})"
    if st == "failed":
        return f"❌ Ошибка генерации: {task.get('error')}"
    return f"⏳ Задача `{tid}` ещё выполняется (`{st}`). Проверь через 30 секунд: `статус {tid}`"


