import os, io, uuid, time, json, base64, asyncio, httpx
from PIL import Image, ImageFilter
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

app = FastAPI(title="KROT GPT Studio API v4.7")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()
TASKS: Dict[str, Dict[str, Any]] = {}

def first(val: Any) -> Any:
    if isinstance(val, (list, tuple)) and len(val) > 0:
        return val
    return val

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
    return {"status": "ok", "version": "4.7", "groq": bool(GROQ_API_KEY), "hf": bool(HF_TOKEN)}

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {"object": "list", "data": [{"id": "gpt-6-astra", "object": "model", "owned_by": "krot"}]}

def extract_text(c: Any) -> str:
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(i.get("text", "") for i in c if isinstance(i, dict) and i.get("type") == "text")
    return str(c)

async def to_en(text: str) -> str:
    if not any("а" <= ch.lower() <= "я" for ch in text) or not GROQ_API_KEY:
        return text
    try:
        async with httpx.AsyncClient(timeout=15.0) as cl:
            r = await cl.post(
                "https://groq.com" + "/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
                json={"model": "llama-3.1-8b-instant", "messages": [{"role": "system", "content": "Translate to English prompt. Output ONLY English."}, {"role": "user", "content": text}]},
            )
            if r.status_code == 200:
                ch = first(r.json().get("choices", []))
                if isinstance(ch, dict):
                    return ch.get("message", {}).get("content", text).strip()
    except Exception:
        pass
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

async def gen_flux(prompt: str) -> tuple[Optional[str], str]:
    en = await to_en(prompt)
    err_log = []
    if HF_TOKEN:
        async with httpx.AsyncClient(timeout=60.0) as cl:
            for m in ["black-forest-labs/FLUX.1-schnell", "black-forest-labs/FLUX.1-dev"]:
                try:
                    r = await cl.post("https://huggingface.co" + f"/hf-inference/models/{m}", headers={"Authorization": f"Bearer {HF_TOKEN}"}, json={"inputs": en})
                    if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
                        img_b64 = await asyncio.to_thread(make_4k, r.content)
                        return img_b64, ""
                    err_log.append(f"{m}: {r.status_code}")
                except Exception as e:
                    err_log.append(f"{m}: {e}")
    def _sp():
        from gradio_client import Client
        res = Client("black-forest-labs/FLUX.1-schnell", hf_token=HF_TOKEN or None).predict(prompt=en, seed=0, randomize_seed=True, width=1024, height=1024, num_inference_steps=4, api_name="/infer")
        item = first(res)
        p = item.get("path") if isinstance(item, dict) else item
        if p and os.path.exists(str(p)):
            with open(p, "rb") as f:
                return make_4k(f.read())
        return None
    try:
        sp_img = await asyncio.to_thread(_sp)
        if sp_img:
            return sp_img, ""
    except Exception as e:
        err_log.append(f"Space: {e}")
    return None, " | ".join(err_log)

@app.post("/v1/image-proxy")
@app.post("/v1/images/generations")
async def img_ep(req: ImageProxyRequest):
    u, err = await gen_flux(req.prompt)
    if not u:
        raise HTTPException(502, f"Ошибка FLUX: {err}")
    return {"url": u, "data": [{"url": u}]}

def _song_sync(tags: str, lyrics: str) -> str:
    from gradio_client import Client
    res = Client("ACE-Step/ACE-Step", hf_token=HF_TOKEN or None).predict(
        audio_duration=90, prompt=tags, lyrics=lyrics, infer_step=40, guidance_scale=15.0,
        scheduler_type="euler", cfg_type="apg", omega_scale=10.0, manual_seeds=None,
        guidance_interval=0.5, guidance_interval_decay=0.0, min_guidance_scale=3.0,
        use_erg_tag=True, use_erg_lyric=False, use_erg_diffusion=True, oss_steps=None,
        guidance_scale_text=0.0, guidance_scale_lyric=0.0, api_name="/__call__"
    )
    item = first(res)
    p = item.get("path") if isinstance(item, dict) else item
    if p and os.path.exists(str(p)):
        with open(p, "rb") as f:
            return "data:audio/mp3;base64," + base64.b64encode(f.read()).decode()
    raise RuntimeError("Нет аудиофайла")

async def _song_job(tid: str, topic: str):
    TASKS[tid]["status"] = "processing"
    try:
        tags, lyrics = "pop, synth, vocals", topic
        if GROQ_API_KEY:
            async with httpx.AsyncClient(timeout=25.0) as cl:
                r = await cl.post("https://groq.com" + "/openai/v1/chat/completions", headers={"Authorization": f"Bearer {GROQ_API_KEY}"}, json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "system", "content": "Напиши песню [verse][chorus]. В 1-й строке: STYLE: pop, synth, vocals. Далее только текст."}, {"role": "user", "content": topic}]})
                if r.status_code == 200:
                    ch = first(r.json().get("choices", []))
                    full = ch.get("message", {}).get("content", topic).strip() if isinstance(ch, dict) else topic
                    lines = full.splitlines()
                    first_line = first(lines) or ""
                    if first_line.upper().startswith("STYLE:"):
                        tags = first_line.split(":", 1)[-1].strip()
                        lyrics = "\n".join(lines[1:]).strip()
                    else:
                        lyrics = full
        TASKS[tid]["lyrics"] = lyrics
        TASKS[tid]["media"] = await asyncio.to_thread(_song_sync, tags, lyrics)
        TASKS[tid]["status"] = "completed"
    except Exception as e:
        TASKS[tid]["status"], TASKS[tid]["error"] = "failed", str(e)

def _vid_sync(en: str) -> str:
    from gradio_client import Client
    errs = []
    for sp, ep, kw in [
        ("Lightricks/ltx-video-distilled", "/text_to_video", {"prompt": en, "negative_prompt": "low quality", "height": 512, "width": 704, "num_frames": 65, "seed": 42}),
        ("Wan-AI/Wan2.1", "/t2v_generation_async", {"prompt": en, "size": "832*480", "watermark_flag": False, "seed": -1}),
    ]:
        try:
            res = Client(sp, hf_token=HF_TOKEN or None).predict(**kw, api_name=ep)
            item = first(res)
            p = (item.get("video") or item.get("path")) if isinstance(item, dict) else item
            if p and os.path.exists(str(p)):
                with open(p, "rb") as f:
                    return "data:video/mp4;base64," + base64.b64encode(f.read()).decode()
        except Exception as e:
            errs.append(f"{sp}: {e}")
    raise RuntimeError(" | ".join(errs))

async def _vid_job(tid: str, prompt: str):
    TASKS[tid]["status"] = "processing"
    try:
        TASKS[tid]["media"] = await asyncio.to_thread(_vid_sync, await to_en(prompt))
        TASKS[tid]["status"] = "completed"
    except Exception as e:
        TASKS[tid]["status"], TASKS[tid]["error"] = "failed", str(e)

async def ask_groq(msgs: List[Dict[str, str]], temp: float) -> tuple[str, str]:
    enh = [{"role": "system", "content": "Ты глубоко мыслящий ИИ. Проведи длинный, детальный анализ по шагам перед ответом."}] + msgs
    hdr = {"Authorization": f"Bearer {GROQ_API_KEY}"}
    url = "https://groq.com" + "/openai/v1/chat/completions"
    async with httpx.AsyncClient(timeout=60.0) as cl:
        r = await cl.post(url, headers=hdr, json={"model": "openai/gpt-oss-120b", "messages": enh, "reasoning_format": "parsed", "reasoning_effort": "high", "temperature": temp})
        if r.status_code == 200:
            ch = first(r.json().get("choices", []))
            m = ch.get("message", {}) if isinstance(ch, dict) else {}
            return m.get("reasoning", "") or "", m.get("content", "") or "Пустой ответ от модели"
        r2 = await cl.post(url, headers=hdr, json={"model": "llama-3.3-70b-versatile", "messages": enh, "temperature": temp})
        if r2.status_code == 200:
            ch2 = first(r2.json().get("choices", []))
            m2 = ch2.get("message", {}) if isinstance(ch2, dict) else {}
            return "", m2.get("content", "") or "Пустой ответ от модели"
        return "", f"Ошибка Groq ({r.status_code}): {r.text}"

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_ep(req: ChatRequest):
    msgs = [{"role": m.role, "content": extract_text(m.content)} for m in req.messages]
    last = next((m["content"] for m in reversed(msgs) if m["role"] == "user"), "")
    low, reply, reason = last.lower(), "", ""

    if "статус" in low:
        t = TASKS.get(last.replace("/", " ").split()[-1])
        if not t:
            reply = "Задача не найдена."
        elif t["status"] == "completed":
