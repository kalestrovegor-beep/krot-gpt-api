import os
import time
import json
import base64
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="KROT GPT HACKER Clean API v6.2")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

# Собираем URL через join, чтобы фильтр при копировании ничего не обрезал
GROQ_URL = "/".join(["https:/", "://groq.com", "openai", "v1", "chat", "completions"])
HF_FLUX_URL = "/".join(["https:/", "router.huggingface.co", "hf-inference", "models", "black-forest-labs", "FLUX.1-schnell"])
GEMINI_CHAT_BASE = "/".join(["https:/", "://googleapis.com", "v1beta", "models", "gemini-2.5-flash:generateContent"])
GEMINI_IMG_BASE = "/".join(["https:/", "://googleapis.com", "v1beta", "models", "imagen-3.0-generate-002:predict"])


@app.get("/")
async def health():
    return {
        "status": "online",
        "version": "6.2-clean",
        "groq_url": GROQ_URL,
        "hf_url": HF_FLUX_URL,
        "groq": bool(GROQ_API_KEY),
        "gemini": bool(GEMINI_API_KEY),
        "hf": bool(HF_TOKEN),
    }


@app.get("/v1/models")
@app.get("/models")
async def models():
    return {
        "object": "list",
        "data": [
            {"id": "gpt-6-astra", "object": "model", "owned_by": "krot"},
            {"id": "gemini-thinking", "object": "model", "owned_by": "krot"},
        ],
    }


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
    return str(content)


def get_first(lst):
    if isinstance(lst, list) and len(lst) > 0:
        return lst
    return {}


async def generate_image_clean(prompt: str) -> str:
    errors = []

    if GEMINI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                r = await client.post(
                    f"{GEMINI_IMG_BASE}?key={GEMINI_API_KEY}",
                    json={"instances": [{"prompt": prompt}], "parameters": {"sampleCount": 1}},
                )
                if r.status_code == 200:
                    pred = get_first(r.json().get("predictions", []))
                    b64 = pred.get("bytesBase64Encoded")
                    if b64:
                        return f"![Изображение](data:image/png;base64,{b64})"
                errors.append(f"Gemini Imagen ({r.status_code})")
        except Exception as e:
            errors.append(f"Gemini Imagen: {e}")

    if HF_TOKEN:
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                r = await client.post(
                    HF_FLUX_URL,
                    headers={"Authorization": f"Bearer {HF_TOKEN}"},
                    json={"inputs": prompt},
                )
                if r.status_code == 200 and "image" in r.headers.get("content-type", ""):
                    b64 = base64.b64encode(r.content).decode("utf-8")
                    return f"![FLUX](data:image/jpeg;base64,{b64})"
                errors.append(f"HF FLUX ({r.status_code}): {r.text[:120]}")
        except Exception as e:
            errors.append(f"HF FLUX: {e}")

    return f"❌ Не удалось создать картинку: {' | '.join(errors) or 'ключи не заданы'}"


async def ask_thinking_ai(messages: list, temp: float) -> tuple[str, str]:
    system_msg = {
        "role": "system",
        "content": "Ты глубоко мыслящий ИИ-ассистент KROT GPT. Всегда проводи подробный, пошаговый и глубокий разбор задачи на русском языке.",
    }
    full_msgs = [system_msg] + messages
    errors = []

    if GROQ_API_KEY:
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                GROQ_URL,
                headers=headers,
                json={
                    "model": "openai/gpt-oss-120b",
                    "messages": full_msgs,
                    "reasoning_format": "parsed",
                    "reasoning_effort": "high",
                    "temperature": temp,
                },
            )
            if r.status_code == 200:
                msg = get_first(r.json().get("choices", [])).get("message", {})
                return msg.get("reasoning", "") or "", msg.get("content", "") or ""
            errors.append(f"Groq gpt-oss ({r.status_code}): {r.text[:120]}")

            r2 = await client.post(
                GROQ_URL,
                headers=headers,
                json={"model": "llama-3.3-70b-versatile", "messages": full_msgs, "temperature": temp},
            )
            if r2.status_code == 200:
                msg2 = get_first(r2.json().get("choices", [])).get("message", {})
                return "", msg2.get("content", "") or ""
            errors.append(f"Groq llama ({r2.status_code}): {r2.text[:120]}")

    if GEMINI_API_KEY:
        user_prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        async with httpx.AsyncClient(timeout=60.0) as client:
            r = await client.post(
                f"{GEMINI_CHAT_BASE}?key={GEMINI_API_KEY}",
                json={"contents": [{"parts": [{"text": user_prompt}]}]},
            )
            if r.status_code == 200:
                cand = get_first(r.json().get("candidates", []))
                part = get_first(cand.get("content", {}).get("parts", []))
                return "", part.get("text", "")
            errors.append(f"Gemini ({r.status_code}): {r.text[:120]}")

    return "", f"⚠️ Ошибка провайдеров: {' | '.join(errors) or 'ключи не найдены'}"


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat(request: Request):
    body = await request.json()
    raw_msgs = body.get("messages", [])
    messages = [{"role": m.get("role", "user"), "content": extract_text(m.get("content", ""))} for m in raw_msgs]
    last_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    model_id = body.get("model", "gpt-6-astra")
    stream = body.get("stream", False)
    temp = float(body.get("temperature", 0.7))

    reasoning = ""
    if any(w in last_text.lower() for w in ["нарисуй", "сгенерируй картинку", "создай изображение"]):
        reply = await generate_image_clean(last_text)
    else:
        reasoning, reply = await ask_thinking_ai(messages, temp)

    if not reply:
        reply = "Не удалось получить ответ от нейросети."

    if stream:
        async def sse():
            if reasoning:
                chunk_r = {
                    "id": "krot-v6",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [{"index": 0, "delta": {"reasoning_content": reasoning}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(chunk_r, ensure_ascii=False)}\n\n"

            chunk_c = {
                "id": "krot-v6",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {"content": reply}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk_c, ensure_ascii=False)}\n\n"

            chunk_end = {
                "id": "krot-v6",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
            yield f"data: {json.dumps(chunk_end, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    return {
        "id": "krot-v6",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply, "reasoning_content": reasoning},
                "finish_reason": "stop",
            }
        ],
    }
