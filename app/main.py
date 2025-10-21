from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
import os

# --- Env & OpenAI ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env (and don't commit it).")
client = OpenAI(api_key=OPENAI_API_KEY)

# --- FastAPI ---
app = FastAPI(title="LLM Chat API", version="1.0.0")

# --- Prometheus counters (labeled by model) ---
llm_tokens_prompt = Counter("llm_tokens_prompt", "Prompt tokens", ["model"])
llm_tokens_completion = Counter("llm_tokens_completion", "Completion tokens", ["model"])
llm_tokens_total = Counter("llm_tokens_total", "Total tokens", ["model"])

# ✅ Pre-warm for gpt-5-nano so it appears in /metrics right away
for m in ["gpt-5-nano"]:
    llm_tokens_prompt.labels(model=m).inc(0)
    llm_tokens_completion.labels(model=m).inc(0)
    llm_tokens_total.labels(model=m).inc(0)

class PromptRequest(BaseModel):
    model: str
    prompt: str

@app.post("/chat")
async def chat(body: PromptRequest):
    try:
        resp = client.chat.completions.create(
            model=body.model,
            messages=[{"role": "user", "content": body.prompt}],
            temperature=0.2,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            llm_tokens_prompt.labels(model=body.model).inc(getattr(usage, "prompt_tokens", 0) or 0)
            llm_tokens_completion.labels(model=body.model).inc(getattr(usage, "completion_tokens", 0) or 0)
            llm_tokens_total.labels(model=body.model).inc(getattr(usage, "total_tokens", 0) or 0)

        content = resp.choices[0].message.content if resp.choices else ""
        return {
            "response": content,
            "usage": {
                "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
                "completion_tokens": getattr(usage, "completion_tokens", None) if usage else None,
                "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            },
            "model": body.model,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"/chat error: {e}")

@app.get("/health/openai")
def health_openai():
    try:
        client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "ping"}],
        )
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
