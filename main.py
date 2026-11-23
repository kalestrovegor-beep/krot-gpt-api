import asyncio
import json
import os
import time
import uuid
import urllib.parse
from typing import List, Optional, Dict, Any

import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="KROT GPT Multimodal API", version="2.7")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

video_tasks: Dict[str, Dict[str, Any]] = {}
SELF_BASE_URL = "https://onrender.com"


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
    size: Optional[str] = "1024x1024"
    model: Optional[str] = "flux"


class VideoGenerationRequest(BaseModel):
    prompt: str


@app.get("/")
@app.get("/v1")
async def health_check():
    return {
        "status": "online",
        "service": "KROT GPT API v2.7",
        "hf_token_configured": bool(os.getenv("HF_TOKEN")),
        "endpoints": [
            "/v1/chat/completions",
            "/v1/images/generations",
            "/v1/videos/generations"
        ]
    }


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {"id": "krot-gpt-text", "object": "model", "owned_by": "krot"},
            {"id": "krot-flux-image", "object": "model", "owned_by": "krot"},
            {"id": "krot-ltx-video", "object": "model", "owned_by": "krot"},
        ]
    }


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(texts)
    return str(content)


def build_proxy_image_url(prompt: str, width: int = 768, height: int = 768, model: str = "flux") -> str:
    encoded_prompt = urllib.parse.quote(prompt)
    seed = uuid.uuid4().int % 1000000
    return (
        f"{SELF_BASE_URL}/v1/image-proxy"
        f"?prompt={encoded_prompt}&width={width}&height={height}&seed={seed}&model={model}"
    )


async def get_free_llm_response(messages: List[Dict[str, str]]) -> str:
    last_error = ""
    text_host = "https://text." + "pollinations" + ".ai"

    async with httpx.AsyncClient(timeout=40.0) as client:
        for model_name in ["openai", "mistral", "llama"]:
            try:
                response = await client.post(
                    f"{text_host}/openai",
                    json={"model": model_name, "messages": messages},
                    headers={"Content-Type": "application/json"}
                )
                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices")
                    if isinstance(choices, list) and len(choices) > 0:
                        msg = choices.get("message", {})
                        if msg.get("content"):
                            return msg["content"]
                last_error = f"HTTP {response.status_code}"
            except Exception as e:
                last_error = str(e)

        try:
            last_user_msg = messages[-1]["content"] if messages else "Привет"
            encoded = urllib.parse.quote(last_user_msg)
            fallback_res = await client.get(f"{text_host}/{encoded}")
            if fallback_res.status_code == 200 and fallback_res.text.strip():
                return fallback_res.text.strip()
        except Exception as e:
            last_error = str(e)

    return f"Временная перегрузка текстового шлюза ({last_error}). Попробуйте повторить запрос."


@app.get("/v1/image-proxy")
async def image_proxy(prompt: str, width: int = 768, height: int = 768, seed: int = 42, model: str = "flux"):
    hf_token = os.getenv("HF_TOKEN", "").strip()

    async with httpx.AsyncClient(timeout=90.0, follow_redirects=True) as client:
        if hf_token:
            try:
                hf_res = await client.post(
                    "https://huggingface.co",
                    headers={"Authorization": f"Bearer {hf_token}"},
                    json={"inputs": prompt}
                )
                if hf_res.status_code == 200 and hf_res.headers.get("content-type", "").startswith("image/"):
                    return Response(content=hf_res.content, media_type=hf_res.headers["content-type"])
            except Exception:
                pass

        try:
            w = max(512, min(768, (width // 64) * 64))
            h = max(512, min(768, (height // 64) * 64))

            submit_res = await client.post(
                "https://stablehorde.net",
                headers={"apikey": "0000000000", "Client-Agent": "KrotGPT:2.7:contact@krot.local"},
                json={
                    "prompt": prompt,
                    "params": {"width": w, "height": h, "steps": 20, "n": 1},
                    "nsfw": False,
                    "r2": True
                }
            )
            if submit_res.status_code != 202:
                raise HTTPException(status_code=502, detail=f"AI Horde отклонил запрос: {submit_res.text}")

            job_id = submit_res.json().get("id")
            if not job_id:
                raise HTTPException(status_code=502, detail="Не получен ID задачи генерации")

            for _ in range(20):
                await asyncio.sleep(3)
                check_res = await client.get(
                    f"https://stablehorde.net{job_id}",
                    headers={"Client-Agent": "KrotGPT:2.7:contact@krot.local"}
                )
                if check_res.status_code == 200:
                    status_data = check_res.json()
                    if status_data.get("done"):
                        generations = status_data.get("generations", [])
                        if isinstance(generations, list) and len(generations) > 0 and generations.get("img"):
                            img_url = generations["img"]
                            img_file = await client.get(img_url)
                            if img_file.status_code == 200:
                                return Response(
                                    content=img_file.content,
                                    media_type=img_file.headers.get("content-type", "image/webp")
                                )
                        break

            raise HTTPException(status_code=504, detail="Очередь бесплатной генерации перегружена, попробуйте ещё раз")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Ошибка генерации: {str(e)}")


def run_video_worker(task_id: str, prompt: str):
    try:
        from gradio_client import Client
        client = Client("Lightricks/LTX-Video")
        result = client.predict(
            prompt=prompt,
            negative_prompt="low quality, worst quality, deformed, distorted, blurry",
            api_name="/generate"
        )
        video_url = result if isinstance(result, (list, tuple)) and len(result) > 0 else str(result)
        video_tasks[task_id] = {
            "status": "completed",
            "video_url": video_url
        }
    except Exception as e:
        video_tasks[task_id] = {
            "status": "failed",
            "error": f"Бесплатный сервер видео перегружен: {str(e)}"
        }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, background_tasks: BackgroundTasks):
    try:
        formatted_messages = [
            {"role": m.role, "content": extract_text(m.content)}
            for m in req.messages
        ]
        last_text = formatted_messages[-1]["content"].strip() if formatted_messages else ""
        lower_text = last_text.lower()

        draw_triggers = ("нарисуй", "сгенерируй картинку", "сгенерируй изображение", "создай картинку", "draw ")
        video_triggers = ("создай видео", "сгенерируй видео", "сделай видео", "video ")

        if any(lower_text.startswith(trigger) for trigger in draw_triggers):
            clean_prompt = last_text
            for trigger in draw_triggers:
                if lower_text.startswith(trigger):
                    clean_prompt = last_text[len(trigger):].strip() or last_text
                    break

            en_prompt = await get_free_llm_response([
                {
                    "role": "system",
                    "content": "Translate the image description into a detailed English prompt for FLUX image generator. Output ONLY the English prompt, no quotes or extra text."
                },
                {"role": "user", "content": clean_prompt}
            ])
            final_prompt = en_prompt.strip() if "перегрузка" not in en_prompt.lower() else clean_prompt

            img_url = build_proxy_image_url(final_prompt)
            reply_text = (
                f"Изображение FLUX по запросу «{clean_prompt}»:\n\n"
                f"![Изображение]({img_url})\n\n"
                f"[Открыть в полном размере]({img_url})"
            )

        elif any(lower_text.startswith(trigger) for trigger in video_triggers):
            clean_prompt = last_text
            for trigger in video_triggers:
                if lower_text.startswith(trigger):
                    clean_prompt = last_text[len(trigger):].strip() or last_text
                    break

            en_prompt = await get_free_llm_response([
                {
                    "role": "system",
