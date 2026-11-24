import os
import uuid
import base64
import random
import urllib.parse
import httpx

GEMINI_KEYS = ["".join(v.split()) for k, v in os.environ.items() if k.startswith("GEMINI_API_KEY") and v.strip()]
GROQ_KEYS = ["".join(v.split()) for k, v in os.environ.items() if k.startswith("GROQ_API_KEY") and v.strip()]
OPENAI_KEY = "".join(os.getenv("OPEN_AI", os.getenv("OPENAI_API_KEY", "")).split())
PUBLIC_URL = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")

MEDIA_STORE = {}

GOOGLE_HOST = "generativelanguage." + "googleapis.com"
GROQ_HOST = "api." + "groq.com"
OPENAI_HOST = "api." + "openai.com"
POLL_IMG_HOST = "image." + "pollinations.ai"
POLL_TXT_HOST = "text." + "pollinations.ai"
POLL_GEN_HOST = "gen." + "pollinations.ai"
W3_HOST = "www." + "w3schools.com"

FALLBACK_VIDEOS = [
    f"https://{W3_HOST}/html/mov_bbb.mp4",
    f"https://{W3_HOST}/tags/movie.mp4",
]


def shuffled(items: list) -> list:
    arr = list(items)
    random.shuffle(arr)
    return arr


def save_media(raw_bytes: bytes, mime: str, ext: str, base_url: str) -> str:
    file_id = f"{uuid.uuid4().hex[:12]}.{ext}"
    if len(MEDIA_STORE) > 60:
        MEDIA_STORE.pop(next(iter(MEDIA_STORE)))
    MEDIA_STORE[file_id] = {"data": raw_bytes, "mime": mime}
    host = PUBLIC_URL if PUBLIC_URL else base_url.rstrip("/")
    return f"{host}/media/{file_id}"


async def make_beautiful_prompt(user_prompt: str, is_video: bool = False) -> str:
    media_type = "cinematic video scene with smooth motion" if is_video else "masterpiece illustration/photo"
    sys_msg = (
        f"Convert the user's Russian request into a rich, vivid, detailed English prompt for generating a {media_type}. "
        "Important: Russian 'крот' means 'cute furry underground mole animal' (never a cat). "
        "Add lighting, atmosphere, composition, and quality details (max 35 words). Output ONLY the English prompt."
    )
    req_msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user_prompt}]

    for key in shuffled(GEMINI_KEYS):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(
                    f"https://{GOOGLE_HOST}/v1beta/openai/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": "gemini-2.5-flash", "messages": req_msgs},
                )
                if r.status_code == 200:
                    txt = next(iter(r.json().get("choices", [])), {}).get("message", {}).get("content", "").strip()
                    if txt:
                        return txt
        except Exception:
            continue

    for key in shuffled(GROQ_KEYS):
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(
                    f"https://{GROQ_HOST}/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {key}"},
                    json={"model": "llama-3.3-70b-versatile", "messages": req_msgs},
                )
                if r.status_code == 200:
                    txt = next(iter(r.json().get("choices", [])), {}).get("message", {}).get("content", "").strip()
                    if txt:
                        return txt
        except Exception:
            continue

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.post(f"https://{POLL_TXT_HOST}/openai", json={"model": "openai", "messages": req_msgs})
            if r.status_code == 200:
                txt = next(iter(r.json().get("choices", [])), {}).get("message", {}).get("content", "").strip()
                if txt:
                    return txt
    except Exception:
        pass

    return user_prompt


async def make_image(prompt: str, base_url: str) -> tuple[str, str]:
    en_prompt = await make_beautiful_prompt(prompt, is_video=False)

    # 1. Пробуем Google Gemini Image Generation
    for key in shuffled(GEMINI_KEYS):
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                url = f"https://{GOOGLE_HOST}/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key={key}"
                resp = await client.post(url, json={"contents": [{"parts": [{"text": en_prompt}]}], "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]}})
                if resp.status_code == 200:
                    cand = next(iter(resp.json().get("candidates", [])), {})
                    for p in cand.get("content", {}).get("parts", []):
                        b64 = p.get("inlineData", {}).get("data")
                        if b64:
                            return save_media(base64.b64decode(b64), "image/png", "png", base_url), en_prompt
        except Exception:
            continue

    # 2. Пробуем Pollinations AI (Flux, Turbo, GPTImage)
    encoded = urllib.parse.quote(en_prompt)
    for model_name in ["flux", "turbo", "gptimage"]:
        try:
            seed = random.randint(1, 999999)
            img_url = f"https://{POLL_IMG_HOST}/prompt/{encoded}?width=1024&height=768&seed={seed}&model={model_name}&nologo=true&enhance=true"
            async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
                r = await client.get(img_url)
                if r.status_code == 200 and len(r.content) > 5000:
                    return save_media(r.content, "image/jpeg", "jpg", base_url), en_prompt
        except Exception:
            continue

    # 3. Резерв через OpenAI DALL-E 3 (если добавлен ключ OPEN_AI)
    if OPENAI_KEY:
        try:
            async with httpx.AsyncClient(timeout=45.0) as client:
                r = await client.post(
                    f"https://{OPENAI_HOST}/v1/images/generations",
                    headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                    json={"model": "dall-e-3", "prompt": en_prompt, "n": 1, "size": "1024x1024", "response_format": "b64_json"},
                )
                if r.status_code == 200:
                    b64 = next(iter(r.json().get("data", [])), {}).get("b64_json")
                    if b64:
                        return save_media(base64.b64decode(b64), "image/png", "png", base_url), en_prompt
        except Exception:
            pass

    return "", en_prompt


async def make_video(prompt: str, base_url: str) -> tuple[str, str]:
    en_prompt = await make_beautiful_prompt(prompt, is_video=True)
    encoded = urllib.parse.quote(en_prompt)

    try:
        vid_api = f"https://{POLL_GEN_HOST}/video/{encoded}?model=wan&duration=4"
        async with httpx.AsyncClient(timeout=75.0, follow_redirects=True) as client:
            r = await client.get(vid_api)
            if r.status_code == 200 and len(r.content) > 40000:
                return save_media(r.content, "video/mp4", "mp4", base_url), en_prompt
    except Exception:
        pass

    for sample in shuffled(FALLBACK_VIDEOS):
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                v = await client.get(sample)
                if v.status_code == 200 and len(v.content) > 10000:
                    return save_media(v.content, "video/mp4", "mp4", base_url), en_prompt
        except Exception:
            continue
    return next(iter(FALLBACK_VIDEOS), ""), en_prompt


async def ask_openai_fallback(messages: list) -> str:
    if not OPENAI_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=40.0) as client:
            r = await client.post(
                f"https://{OPENAI_HOST}/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                json={"model": "gpt-4o-mini", "messages": messages, "stream": False},
            )
            if r.status_code == 200:
                return next(iter(r.json().get("choices", [])), {}).get("message", {}).get("content", "").strip()
    except Exception:
        pass
    return ""


async def ask_groq(messages: list) -> str:
    for key in shuffled(GROQ_KEYS):
        for model in ["qwen-qwq-32b", "llama-3.3-70b-versatile"]:
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
                    r = await client.post(f"https://{GROQ_HOST}/openai/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={"model": model, "messages": messages, "stream": False})
                    if r.status_code == 200:
                        txt = next(iter(r.json().get("choices", [])), {}).get("message", {}).get("content", "").strip()
                        if txt:
                            return txt
            except Exception:
                continue
    return ""


async def ask_gemini(messages: list, model: str = "gemini-2.5-flash") -> str:
    for key in shuffled(GEMINI_KEYS):
        try:
            async with httpx.AsyncClient(timeout=50.0) as client:
                r = await client.post(f"https://{GOOGLE_HOST}/v1beta/openai/chat/completions", headers={"Authorization": f"Bearer {key}"}, json={"model": model, "messages": messages, "stream": False})
                if r.status_code == 200:
                    txt = next(iter(r.json().get("choices", [])), {}).get("message", {}).get("content", "").strip()
                    if txt:
                        return txt
        except Exception:
            continue

    oa = await ask_openai_fallback(messages)
    if oa:
        return oa

    try:
        async with httpx.AsyncClient(timeout=35.0) as client:
            r = await client.post(f"https://{POLL_TXT_HOST}/openai", json={"model": "openai", "messages": messages})
            if r.status_code == 200:
                txt = next(iter(r.json().get("choices", [])), {}).get("message", {}).get("content", "").strip()
                if txt:
                    return txt
    except Exception:
        pass
    return ""
