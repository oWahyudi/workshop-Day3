from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
from openai import OpenAI

# Prefer v3 get_client(); fall back gracefully if not available
try:
    from langfuse import get_client  # v3
except ImportError:
    get_client = None

# ------------------------------
# Load environment variables
# ------------------------------
if os.path.exists(".env"):
    load_dotenv(".env")
elif os.path.exists(".env.template"):
    load_dotenv(".env.template")

# ------------------------------
# Initialize Langfuse client
# ------------------------------
langfuse = None
if get_client:
    # v3 client: reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (/ LANGFUSE_HOST) from env
    langfuse = get_client()

# ------------------------------
# Initialize OpenAI client
# ------------------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ------------------------------
# Initialize Flask app
# ------------------------------
app = Flask(__name__)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message")
    user_id = data.get("user_id", "anonymous")

    if not message:
        return jsonify({"error": "message is required"}), 400

    try:
        # If Langfuse v3 is available, create a root span/trace
        if langfuse is not None:
            with langfuse.start_as_current_span(name="chat-endpoint") as span:
                # attach user + input to the trace/span
                try:
                    span.update_trace(user_id=user_id)
                except Exception:
                    # older clients may not support update_trace
                    pass
                try:
                    span.update(input={"message": message})
                except Exception:
                    pass

                # Generation span for the LLM call
                with span.start_as_current_generation(
                    name="openai-completion",
                    model="gpt-4o-mini",  # use a valid model name you have access to
                ) as gen:
                    resp = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": message}],
                    )
                    answer = resp.choices[0].message.content or ""
                    try:
                        gen.update(output=answer)
                    except Exception:
                        pass

                # Try to record a score in v3; if not supported, just skip
                try:
                    if hasattr(langfuse, "score"):
                        langfuse.score(
                            trace_id=span.trace_id,
                            name="response_length",
                            value=len(answer),
                        )
                except Exception:
                    # Some installs expose no .score or have different signature — ignore
                    pass

                # Ensure telemetry is sent for short-lived web requests
                try:
                    langfuse.flush()
                except Exception:
                    pass

                return jsonify({"response": answer, "trace_id": getattr(span, "trace_id", None)})

        # ------- If Langfuse not available, just run OpenAI and return -------
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": message}],
        )
        answer = resp.choices[0].message.content or ""
        return jsonify({"response": answer, "trace_id": None})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(host="0.0.0.0", port=4000, debug=debug_mode)
