import json
import asyncio
import websockets
from fastapi import FastAPI, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from ai_providers import (
    GEMINI_KEYS,
    GROQ_KEYS,
    MEDIA_STORE,
    GOOGLE_HOST,
    shuffled,
    make_image,
    make_video,
    ask_groq,
    ask_gemini,
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "gemini_keys": len(GEMINI_KEYS), "groq_keys": len(GROQ_KEYS)}


@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {"object": "list", "data": [{"id": "gemini-2.5-flash", "object": "model"}, {"id": "groq-reasoning", "object": "model"}]}


@app.get("/media/{file_id}")
async def serve_media(file_id: str):
    item = MEDIA_STORE.get(file_id)
    if not item:
        return Response(status_code=404)
    return Response(content=item["data"], media_type=item["mime"])


async def sse_stream(text: str):
    for delta, reason in [({"role": "assistant"}, None), ({"content": text}, None), ({}, "stop")]:
        chunk = {"id": "chatcmpl-krot", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def send_reply(text: str, is_stream: bool):
    if is_stream:
        return StreamingResponse(sse_stream(text), media_type="text/event-stream")
    return JSONResponse({"id": "chatcmpl-krot", "object": "chat.completion", "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]})


@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    is_stream = bool(body.get("stream", False))
    base_url = str(request.base_url).rstrip("/")
    try:
        msgs = [m for m in body.get("messages", []) if isinstance(m.get("content"), str)]
        last_msg = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
        low = last_msg.lower()
        req_model = str(body.get("model", "")).lower()

        if any(w in low for w in ["сгенерируй видео", "создай видео", "сделай видео", "видео про", "generate video"]):
            vid, en_p = await make_video(last_msg, base_url)
            return send_reply(f"🎬 **Видео готово!**\n*Промпт:* `{en_p}`\n\n👉 [Открыть и посмотреть видео (MP4)]({vid})", is_stream)

        if any(w in low for w in ["нарисуй", "сгенерируй картинку", "сгенерируй фото", "создай изображение", "картинку", "draw "]):
            img, en_p = await make_image(last_msg, base_url)
            ans = f"🎨 **Промпт:** `{en_p}`\n\n![art]({img})" if img else "Не удалось создать изображение, повторите запрос."
            return send_reply(ans, is_stream)

        if any(k in req_model for k in ["reason", "think", "r1", "pro", "o1"]) or any(w in low for w in ["подумай", "поразмышляй", "реши подробно"]):
            rep = await ask_groq(msgs) or await ask_gemini(msgs, "gemini-2.5-pro") or await ask_gemini(msgs, "gemini-2.5-flash")
            return send_reply(rep or "Готово!", is_stream)

        rep = await ask_gemini(msgs, "gemini-2.5-flash") or await ask_groq(msgs)
        return send_reply(rep or "Привет! Чем помочь?", is_stream)
    except Exception as e:
        return send_reply(f"Ошибка: {str(e)}", is_stream)


@app.post("/v1/images/generations")
@app.post("/images/generations")
async def image_generations(request: Request):
    data = await request.json()
    url, _ = await make_image(data.get("prompt", "art"), str(request.base_url).rstrip("/"))
    return {"data": [{"url": url}]}


@app.post("/v1/video/generations")
@app.post("/video/generations")
async def video_generations(request: Request):
    data = await request.json()
    url, _ = await make_video(data.get("prompt", "nature"), str(request.base_url).rstrip("/"))
    return {"data": [{"url": url}]}


@app.websocket("/v1/live")
@app.websocket("/live")
async def live_video_call(client_ws: WebSocket):
    await client_ws.accept()
    for key in shuffled(GEMINI_KEYS):
        ws_url = f"wss://{GOOGLE_HOST}/ws/google.ai.generativelanguage.v1alpha.GenerativeService.BidiGenerateContent?key={key}"
        try:
            async with websockets.connect(ws_url) as gemini_ws:
                async def c2g():
                    while True:
                        await gemini_ws.send(await client_ws.receive_text())

                async def g2c():
                    async for msg in gemini_ws:
                        await client_ws.send_text(msg.decode("utf-8") if isinstance(msg, bytes) else msg)

                await asyncio.gather(c2g(), g2c())
                return
        except Exception:
            continue
