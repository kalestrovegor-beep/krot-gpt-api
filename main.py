import os, io, uuid, time, json, base64, asyncio, httpx
from PIL import Image, ImageFilter
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

app = FastAPI(title="KROT GPT Studio API v4.6")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

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
    upscale_4k: Optional[bool] = True

@app.get("/")
async def root():
    return {"status": "ok", "version": "4.6", "groq": bool(GROQ_API_KEY), "hf": bool(HF_TOKEN)}

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {"object": "list", "data": [{"id": "gpt-6-astra", "object": "model", "owned_by": "krot"}]}

def extract_text(c: Any) -> str:
    if isinstance(c, str): return c
    if isinstance(c, list): return "\n".join(i.get("text", "") for i in c if isinstance(i, dict) and i.get("type") == "text")
    return str(c)

async def to_en(text: str) -> str:
    if not any("а" <= ch.lower() <= "я" for ch in text) or not GROQ_API_KEY: return text
    try:
        async with httpx.AsyncClient(timeout=15.0) as cl:
            r = await cl.post("https://groq.com" + "/openai/v1/chat/completions", headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, json={"model": "llama-3.1-8b-instant", "messages": [{"role": "system", "content": "Translate to English prompt. Output ONLY English."}, {"role": "user", "content": text}]})
            if r.status_code == 200: return r.json()["choices"]["message"]["content"].strip()
    except Exception: pass
    return text

def make_4k(raw: bytes) -> str:
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    w, h = img.size
    sc = 3840 / max(w, h)
    if sc > 1:
        img = img.resize((int(w * sc), int(h * sc)), Image.Resampling.LANCZOS).filter(ImageFilter.UnsharpMask(1.5, 120, 3))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

async def gen_flux(prompt: str) -> Optional[str]:
    en = await to_en(prompt)
    if HF_TOKEN:
        async with httpx.AsyncClient(timeout=60.0) as cl:
            for m in ["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev"]:
                try:
                    r = await cl.post("https://huggingface.co" + f"/hf-inference/models/{m}", headers={"Authorization": f"Bearer {HF_TOKEN}"}, json={"inputs": en})
                    if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
                        return await asyncio.to_thread(make_4k, r.content)
                except Exception: pass
    def _sp():
        from gradio_client import Client
        res = Client("black-forest-labs/FLUX.1-schnell", hf_token=HF_TOKEN or None).predict(prompt=en, seed=0, randomize_seed=True, width=1024, height=1024, num_inference_steps=4, api_name="/infer")
        p = res["path"] if isinstance(res, dict) else (res if isinstance(res, (list, tuple)) else res)
        if p and os.path.exists(str(p)):
            with open(p, "rb") as f: return make_4k(f.read())
        return None
    try: return await asyncio.to_thread(_sp)
    except Exception: return None

@app.post("/v1/image-proxy")
@app.post("/v1/images/generations")
async def img_ep(req: ImageProxyRequest):
    u = await gen_flux(req.prompt)
    if not u: raise HTTPException(502, "Ошибка FLUX")
    return {"url": u, "data": [{"url": u}]}

def _song_sync(tags: str, lyrics: str) -> str:
    from gradio_client import Client
    res = Client("ACE-Step/ACE-Step", hf_token=HF_TOKEN or None).predict(audio_duration=90, prompt=tags, lyrics=lyrics, infer_step=40, guidance_scale=15.0, scheduler_type="euler", cfg_type="apg", omega_scale=10.0, manual_seeds=None, guidance_interval=0.5, guidance_interval_decay=0.0, min_guidance_scale=3.0, use_erg_tag=True, use_erg_lyric=False, use_erg_diffusion=True, oss_steps=None, guidance_scale_text=0.0, guidance_scale_lyric=0.0, api_name="/__call__")
    p = res["path"] if isinstance(res, dict) else (res if isinstance(res, (list, tuple)) else res)
    if p and os.path.exists(str(p)):
        with open(p, "rb") as f: return "data:audio/mp3;base64," + base64.b64encode(f.read()).decode()
    raise RuntimeError("Нет аудио")

async def _song_job(tid: str, topic: str):
    TASKS[tid]["status"] = "processing"
    try:
        tags, lyrics = "pop, synth, vocals", topic
        if GROQ_API_KEY:
            async with httpx.AsyncClient(timeout=25.0) as cl:
                r = await cl.post("https://groq.com" + "/openai/v1/chat/completions", headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "system", "content": "Напиши песню [verse][chorus]. В 1-й строке: STYLE: pop, synth, vocals. Далее только текст."}, {"role": "user", "content": topic}]})
                if r.status_code == 200:
                    lines = r.json()["choices"]["message"]["content"].strip().splitlines()
                    if lines and lines.upper().startswith("STYLE:"): tags, lyrics = lines.split(":", 1).strip(), "\n".join(lines[1:]).strip()
                    else: lyrics = "\n".join(lines)
        TASKS[tid]["lyrics"] = lyrics
        TASKS[tid]["media"] = await asyncio.to_thread(_song_sync, tags, lyrics)
        TASKS[tid]["status"] = "completed"
    except Exception as e:
        TASKS[tid]["status"], TASKS[tid]["error"] = "failed", str(e)

def _vid_sync(en: str) -> str:
    from gradio_client import Client
    for sp, ep, kw in [("Lightricks/ltx-video-distilled", "/text_to_video", {"prompt": en, "negative_prompt": "low quality", "height": 512, "width": 704, "num_frames": 65, "seed": 42}), ("Wan-AI/Wan2.1", "/t2v_generation_async", {"prompt": en, "size": "832*480", "watermark_flag": False, "seed": -1})]:
        try:
            res = Client(sp, hf_token=HF_TOKEN or None).predict(**kw, api_name=ep)
            p = res.get("video") or res.get("path") if isinstance(res, dict) else (res if isinstance(res, (list, tuple)) else res)
            if p and os.path.exists(str(p)):
                with open(p, "rb") as f: return "data:video/mp4;base64," + base64.b64encode(f.read()).decode()
        except Exception: pass
    raise RuntimeError("GPU видео-серверы заняты")

async def _vid_job(tid: str, prompt: str):
    TASKS[tid]["status"] = "processing"
    try:
        TASKS[tid]["media"] = await asyncio.to_thread(_vid_sync, await to_en(prompt))
        TASKS[tid]["status"] = "completed"
    except Exception as e:
        TASKS[tid]["status"], TASKS[tid]["error"] = "failed", str(e)

async def ask_groq(msgs: List[Dict[str, str]], temp: float) -> tuple[str, str]:
    enh = [{"role": "system", "content": "Ты глубоко мыслящий ИИ. Проведи длинный, детальный анализ по шагам перед ответом."}] + msgs
    hdr, url = {"Authorization": f"Bearer {GROQ_API_KEY}"}, "https://groq.com" + "/openai/v1/chat/completions"
    async with httpx.AsyncClient(timeout=60.0) as cl:
        r = await cl.post(url, headers=hdr, json={"model": "openai/gpt-oss-120b", "messages": enh, "reasoning_format": "parsed", "reasoning_effort": "high", "temperature": temp})
        if r.status_code == 200:
            m = r.json()["choices"]["message"]
            return m.get("reasoning", ""), m.get("content", "")
        r2 = await cl.post(url, headers=hdr, json={"model": "llama-3.3-70b-versatile", "messages": enh, "temperature": temp})
        if r2.status_code == 200: return "", r2.json()["choices"]["message"].get("content", "")
        return "", f"Ошибка Groq ({r.status_code}): {r.text}"

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_ep(req: ChatRequest):
    msgs = [{"role": m.role, "content": extract_text(m.content)} for m in req.messages]
    last = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
    low, reply, reason = last.lower(), "", ""
    if "статус" in low:
        t = TASKS.get(last.replace("/", " ").split()[-1])
        if not t: reply = "Задача не найдена."
        elif t["status"] == "completed": reply = f"🎵 **Готово!** [Открыть / скачать MP3]({t['media']})\n\n{t.get('lyrics','')}" if t["type"] == "song" else f"🎬 **Готово!** [Открыть / скачать MP4]({t['media']})"
        elif t["status"] == "failed": reply = f"❌ Ошибка: {t.get('error')}"
        else: reply = f"⏳ В процессе (`{t['status']}`). Проверь через 30 сек."
    elif any(k in low for k in ["напиши песню", "создай песню", "сгенерируй песню", "сочини песню", "сделай трек"]):
        tid = uuid.uuid4().hex[:8]
        TASKS[tid] = {"type": "song", "status": "queued"}
        asyncio.create_task(_song_job(tid, last))
        reply = f"🎤 **Создаю песню!** Через 1–2 минуты напиши: **`статус {tid}`**"
    elif any(k in low for k in ["сгенерируй видео", "создай видео", "сделай видео", "сними видео"]):
        tid = uuid.uuid4().hex[:8]
        TASKS[tid] = {"type": "video", "status": "queued"}
        asyncio.create_task(_vid_job(tid, last))
        reply = f"🎬 **Создаю видео!** Через 1–2 минуты напиши: **`статус {tid}`**"
    elif any(k in low for k in ["нарисуй", "сгенерируй картинку", "создай изображение", "сделай фото"]):
        img = await gen_flux(last)
        reply = f"✨ **Готово (FLUX 4K):**\n\n![FLUX 4K]({img})" if img else "❌ Не удалось создать картинку."
    else:
        reason, reply = await ask_groq(msgs, req.temperature or 0.7) if GROQ_API_KEY else ("", "⚠️ Нет GROQ_API_KEY")

