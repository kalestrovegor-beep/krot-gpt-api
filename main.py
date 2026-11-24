import os
import time
import json
import base64
import asyncio
import traceback
import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="KROT GPT Clean API v6.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

PROTO = "https" + chr(58) + "//"
GROQ_URL = PROTO + "/".join(["://groq.com", "openai", "v1", "chat", "completions"])
GEMINI_CHAT_BASE = PROTO + "/".join(["://googleapis.com", "v1beta", "models", "gemini-2.5-flash:generateContent"])
GEMINI_IMG_BASE = PROTO + "/".join(["://googleapis.com", "v1beta", "models", "imagen-3.0-generate-002:predict"])
HORDE_BASE = PROTO + "/".join(["stablehorde.net", "api", "v2", "generate"])


@app.get("/")
async def health():
    return dict(
        status="online",
        version="6.7",
        groq=bool(GROQ_API_KEY),
        gemini=bool(GEMINI_API_KEY),
        groq_url=GROQ_URL,
    )


@app.get("/v1/models")
@app.get("/models")
async def models():
    return dict(object="list", data=[dict(id="gpt-6-astra", object="model", owned_by="krot")])


def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(x.get("text", "") for x in content if isinstance(x, dict))
    return str(content)


def first_dict(lst):
    if isinstance(lst, list):
        for item in lst:
            if isinstance(item, dict):
                return item
    return dict()


async def translate_prompt(text: str) -> str:
    if not any("а" <= c.lower() <= "я" for c in text) or not GROQ_API_KEY:
        return text
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(
                GROQ_URL,
                headers=dict(Authorization=f"Bearer {GROQ_API_KEY}"),
                json=dict(
                    model="llama-3.1-8b-instant",
                    messages=[
                        dict(role="system", content="Translate to detailed English image prompt. Output ONLY English text."),
                        dict(role="user", content=text),
                    ],
                ),
            )
            if r.status_code == 200:
                choice = first_dict(r.json().get("choices", []))
                return choice.get("message", dict()).get("content", text).strip()
    except Exception:
        pass
    return text


async def generate_image_clean(prompt: str) -> str:
    en_prompt = await translate_prompt(prompt)
    errors = []

    if GEMINI_API_KEY:
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                r = await client.post(
                    f"{GEMINI_IMG_BASE}?key={GEMINI_API_KEY}",
                    json=dict(instances=[dict(prompt=en_prompt)], parameters=dict(sampleCount=1)),
                )
                if r.status_code == 200:
                    pred = first_dict(r.json().get("predictions", []))
                    b64 = pred.get("bytesBase64Encoded")
                    if b64:
                        return f"![Imagen 3](data:image/png;base64,{b64})"
                errors.append(f"Imagen ({r.status_code})")
        except Exception as e:
            errors.append(f"Imagen: {e}")

    headers = {"apikey": "0000000000", "Client-Agent": "KrotGPT:6.7:admin"}
    payload = dict(
        prompt=en_prompt,
        params=dict(width=768, height=768, steps=20, n=1),
        models=["AlbedoBase XL (SDXL)", "Juggernaut XL", "Flux.1-Schnell fp8"],
    )
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{HORDE_BASE}/async", json=payload, headers=headers)
            if r.status_code == 202:
                gen_id = r.json().get("id")
                for _ in range(25):
                    await asyncio.sleep(2.0)
                    chk = await client.get(f"{HORDE_BASE}/check/{gen_id}", headers=headers)
                    if chk.status_code == 200 and chk.json().get("done"):
                        st = await client.get(f"{HORDE_BASE}/status/{gen_id}", headers=headers)
                        gen = first_dict(st.json().get("generations", []))
                        img_val = gen.get("img", "")
                        if img_val.startswith("http"):
                            img_r = await client.get(img_val)
                            b64 = base64.b64encode(img_r.content).decode("utf-8")
                            return f"![Изображение](data:image/webp;base64,{b64})"
                        elif img_val:
                            return f"![Изображение](data:image/webp;base64,{img_val})"
                errors.append("AI Horde: очередь перегружена")
            else:
                errors.append(f"AI Horde ({r.status_code})")
    except Exception as e:
        errors.append(f"AI Horde: {e}")

    return f"❌ Не удалось создать картинку ({' | '.join(errors)})."


async def ask_thinking_ai(messages: list, temp: float) -> tuple[str, str]:
    sys_msg = dict(role="system", content="Ты глубоко мыслящий ИИ KROT GPT. Проводи подробный, глубокий анализ на русском языке.")
    full_msgs = [sys_msg] + messages
    errors = []

    if GROQ_API_KEY:
        headers = dict(Authorization=f"Bearer {GROQ_API_KEY}")
        async with httpx.AsyncClient(timeout=60.0) as client:
            try:
                r = await client.post(
                    GROQ_URL,
                    headers=headers,
                    json=dict(
                        model="openai/gpt-oss-120b",
                        messages=full_msgs,
                        reasoning_format="parsed",
                        reasoning_effort="high",
                        temperature=temp,
                    ),
                )
                if r.status_code == 200:
                    msg = first_dict(r.json().get("choices", [])).get("message", dict())
                    return str(msg.get("reasoning") or ""), str(msg.get("content") or "")
                errors.append(f"Groq oss ({r.status_code}): {r.text[:100]}")
            except Exception as e:
                errors.append(f"Groq oss: {e}")

            try:
                r2 = await client.post(
                    GROQ_URL,
                    headers=headers,
                    json=dict(model="llama-3.3-70b-versatile", messages=full_msgs, temperature=temp),
                )
                if r2.status_code == 200:
                    msg2 = first_dict(r2.json().get("choices", [])).get("message", dict())
                    return "", str(msg2.get("content") or "")
                errors.append(f"Groq llama ({r2.status_code}): {r2.text[:100]}")
            except Exception as e:
                errors.append(f"Groq llama: {e}")

    if GEMINI_API_KEY:
        user_prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(
                    f"{GEMINI_CHAT_BASE}?key={GEMINI_API_KEY}",
                    json=dict(contents=[dict(parts=[dict(text=user_prompt)])]),
                )
                if r.status_code == 200:
                    cand = first_dict(r.json().get("candidates", []))
                    part = first_dict(cand.get("content", dict()).get("parts", []))
                    return "", str(part.get("text") or "")
                errors.append(f"Gemini ({r.status_code}): {r.text[:100]}")
        except Exception as e:
            errors.append(f"Gemini: {e}")

    return "", f"⚠️ Ошибка провайдеров: {' | '.join(errors) or 'ключи не заданы'}"


def make_chunk(model_id: str, delta: dict, finish=None) -> str:
    payload = dict(
        id="krot-v6",
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model_id,
        choices=[dict(index=0, delta=delta, finish_reason=finish)],
    )
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat(request: Request):
    reasoning = ""
    reply = ""
    model_id = "gpt-6-astra"
    stream = False

    try:
        body = await request.json()
        raw_msgs = body.get("messages", [])
        messages = [dict(role=m.get("role", "user"), content=extract_text(m.get("content", ""))) for m in raw_msgs]
        last_text = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        model_id = body.get("model", "gpt-6-astra")
        stream = bool(body.get("stream", False))
        temp = float(body.get("temperature", 0.7))

        if any(w in last_text.lower() for w in ["нарисуй", "сгенерируй картинку", "создай изображение"]):
            reply = await generate_image_clean(last_text)
        else:
            reasoning, reply = await ask_thinking_ai(messages, temp)
    except Exception:
        reply = "⚠️ Ошибка сервера:\n" + traceback.format_exc()[-300:]

    if not reply:
        reply = "Не удалось получить ответ."

    if stream:
        async def sse():
            if reasoning:
                yield make_chunk(model_id, dict(reasoning_content=reasoning))
            yield make_chunk(model_id, dict(content=reply))
            yield make_chunk(model_id, dict(), finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    msg = dict(role="assistant", content=reply, reasoning_content=reasoning)
    return dict(
        id="krot-v6",
        object="chat.completion",
        created=int(time.time()),
        model=model_id,
        choices=[dict(index=0, message=msg, finish_reason="stop")],
    )
