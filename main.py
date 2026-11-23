async def translate_to_english(prompt: str) -> str:
    if not GEMINI_API_KEY:
        return prompt
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            tr = await client.post(
                f"https://{GOOGLE_HOST}/v1beta/openai/chat/completions",
                headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
                json={
                    "model": "gemini-2.5-flash",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Translate the user request into a clear, detailed English prompt for an AI image generator. Output ONLY the English translation, nothing else."
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            if tr.status_code == 200:
                choices = tr.json().get("choices", [])
                first_choice = next(iter(choices), {})
                translated = first_choice.get("message", {}).get("content", "").strip()
                if translated:
                    return translated
    except Exception:
        pass
    return prompt
