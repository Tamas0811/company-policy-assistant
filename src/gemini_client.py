"""Thin wrapper around the Gemini API with model fallback and token counting."""
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError, ServerError

load_dotenv()

# Override with a comma-separated GEMINI_MODELS value in .env / Streamlit secrets
# if any of these names are not available for your key.
DEFAULT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]  # 3.5 Flash Lite: 15 RPM / 500 per day on free tier


def _models():
    env = os.getenv("GEMINI_MODELS", "")
    models = [m.strip() for m in env.split(",") if m.strip()]
    return models or DEFAULT_MODELS


class GeminiClient:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Put it in .env locally or in the app's Secrets on Streamlit Cloud."
            )
        self.client = genai.Client(api_key=api_key)
        self.models = _models()

    def generate(self, prompt: str):
        """Return (text, token_usage dict, model name). Raises RuntimeError if every model fails."""
        last_error = None
        for model_name in self.models:
            for _ in range(2):
                try:
                    response = self.client.models.generate_content(model=model_name, contents=prompt)
                    usage = response.usage_metadata
                    token_usage = {
                        "prompt_tokens": (usage.prompt_token_count or 0) if usage else 0,
                        "completion_tokens": (usage.candidates_token_count or 0) if usage else 0,
                        "total_tokens": (usage.total_token_count or 0) if usage else 0,
                    }
                    return response.text or "", token_usage, model_name
                except ServerError as e:  # 5xx, e.g. 503 high demand -> retry once
                    last_error = e
                    time.sleep(1)
                except APIError as e:  # 4xx, e.g. unknown model -> try the next model
                    last_error = e
                    break
        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


EMPTY_USAGE = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
