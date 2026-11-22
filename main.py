from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI
import time
import json
import os

app = FastAPI(title="KROT GPT Backend API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

ai_client = AsyncOpenAI(
    base_url="https://openrouter.ai" + "/api/v1",
    api_key=OPENROUTER_API_KEY,
)

FREE_MODEL = "openrouter/free"

@app.get("/")
async def root():
    return {"status": "ok", "service": "KROT GPT API is running 24/7"}

@app.post("/v1/chat/completions")
@app.post("/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    model_name = body.get("model", "gpt-6-astra")
    is_stream = body.get("stream", False)

    if is_stream:
        async def event_generator():
            try:
                stream = await ai_client.chat.completions.create(
                    model=FREE_MODEL,
                    messages=messages,
                    stream=True
                )
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                        text_piece = chunk.choices[0].delta.content
                        data = {
                            "id": "chatcmpl-krot-live",
                            "object": "chat.completion.chunk",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": text_piece}, "finish_reason": None}]
                        }
                        yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
            except Exception as e:
                error_data = {
                    "id": "chatcmpl-krot-err",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"content": f"Ошибка сервера: {str(e)}"}, "finish_reason": None}]
                }
                yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    response = await ai_client.chat.completions.create(
        model=FREE_MODEL,
        messages=messages,
        stream=False
    )
    return {
        "id": "chatcmpl-krot-live",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": response.choices[0].message.content},
            "finish_reason": "stop"
        }]
    }

@app.get("/v1/models")
@app.get("/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "gpt-6-astra", "object": "model", "owned_by": "krot"}]
    }

books_db = [{"id": 1, "title": "Приключения Крота в космосе", "author": "KROT Team"}]

@app.get("/v1/books")
@app.get("/books")
async def get_books():
    return books_db

@app.post("/v1/books")
@app.post("/books")
async def add_book(request: Request):
    data = await request.json()
    new_book = {"id": len(books_db) + 1, "title": data.get("title", ""), "author": data.get("author", "")}
    books_db.append(new_book)
    return new_book
