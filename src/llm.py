# src/llm.py
# ---------------------------------------------------------------------------
# The ONE place that talks to the text language model.
# Everything else calls ask() / ask_json() / ask_groq() from here.
#
# All text generation runs on Groq (llama-3.3-70b): it is fast (~0.25s/call),
# supports JSON mode, and has generous rate limits — so the whole pipeline runs in
# well under a few minutes. (PDF reading is the one job that needs a multimodal
# model; that lives in pdf_reader.py on Gemini and is cached to disk.)
#
# A light throttle + 429 retry/backoff is our robust-failure handling: a throttled
# call waits and retries instead of failing or pretending it succeeded.
# ---------------------------------------------------------------------------

import os
import re
import json
import time
import urllib.request
from dotenv import load_dotenv
from groq import Groq
from google import genai
from google.genai import types

load_dotenv()                                   # read API keys from .env
_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
_gemini = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Models, chosen for their job:
#   FAST_MODEL  (Groq) - the agent's many small controller decisions, and the Part 2
#                 reviewer/summariser/formatter calls. 8b-instant has very high
#                 throughput, so the loop's decision steps run in well under a second each.
#   SMART_MODEL (Groq) - higher-quality Groq model, kept as a configurable option.
#   EXTRACT_MODEL (Gemini) - the ONE batched extraction call per patient. Gemini's large
#                 request limit ingests a full scanned record (~9k+ tokens) that Groq's
#                 free-tier per-request cap rejects, and it is accurate on nuanced fields.
FAST_MODEL    = os.getenv("GROQ_FAST_MODEL",  "llama-3.1-8b-instant")
SMART_MODEL   = os.getenv("GROQ_SMART_MODEL", "llama-3.3-70b-versatile")
EXTRACT_MODEL = os.getenv("GEMINI_EXTRACT_MODEL", "gemini-2.5-flash-lite")
GROQ_MODEL    = SMART_MODEL                      # back-compat alias

MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "0.0"))  # optional spacing between
# calls. Groq is fast with generous limits, so 0 by default; raise it via .env if you
# hit rate limits. The 429 retry/backoff below absorbs occasional throttling.
_last_call = 0.0                                # time of the previous call

# Hard per-request timeout. A stalled API call (no error, just hanging) must never
# freeze the run — it aborts and is retried/fallen-back instead. This is what keeps
# every run (and the demo) bounded in time.
REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "30"))


def _throttle():
    """Sleep just enough to keep calls at least MIN_INTERVAL seconds apart."""
    global _last_call
    wait = MIN_INTERVAL - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)
    _last_call = time.time()


def _is_rate_limit(err) -> bool:
    text = str(err)
    return "RESOURCE_EXHAUSTED" in text or "429" in text or "rate_limit" in text.lower()


def _is_retryable(err) -> bool:
    """Rate-limit OR timeout: both mean 'back off and try again', not 'give up'."""
    t = str(err).lower()
    return _is_rate_limit(err) or "timeout" in t or "timed out" in t


def _retry_seconds(err, fallback) -> float:
    """Pull the server's suggested retry delay out of the error, else use fallback."""
    match = re.search(r"(?:retryDelay|retry-after|try again in)['\"]?[:\s]+'?([\d.]+)", str(err))
    return float(match.group(1)) if match else fallback


def _chat(messages, retries: int = 5, json_mode: bool = False, model: str = None,
          timeout: float = None) -> str:
    """Call Groq with light pacing + a hard per-request timeout + backoff.

    A throttled OR stalled call waits and retries instead of hanging the run; only if
    every retry fails does it raise (so the caller can fall back / flag). `timeout`
    overrides the default per-request limit (used for cheap calls that should fail fast).
    """
    kwargs = {"model": model or FAST_MODEL, "temperature": 0, "messages": messages,
              "timeout": timeout or REQUEST_TIMEOUT}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(retries):
        _throttle()
        try:
            resp = _groq.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if _is_retryable(e) and attempt < retries - 1:
                time.sleep(_retry_seconds(e, fallback=4) + 1)
                continue
            raise


def ask(prompt: str) -> str:
    """Send a prompt, get plain text back."""
    return _chat([{"role": "user", "content": prompt}])


def ask_json(prompt: str, smart: bool = False) -> dict:
    """Send a prompt to Groq, get a parsed JSON object back (used by the controller)."""
    model = SMART_MODEL if smart else FAST_MODEL
    return json.loads(_chat([{"role": "user", "content": prompt}], json_mode=True, model=model))


# --- Gemini path: ONE big-context extraction call per patient ----------------
_GEMINI_MIN_INTERVAL = float(os.getenv("GEMINI_MIN_INTERVAL", "1.0"))  # light pacing
_gemini_last = 0.0


def extract_json(prompt: str, retries: int = 4) -> dict:
    """Send an extraction prompt to Gemini and parse JSON back.

    Gemini's large request limit ingests a full record that Groq rejects. Lightly
    paced + 429-backoff so a throttled call waits and retries instead of failing.
    """
    global _gemini_last
    for attempt in range(retries):
        wait = _GEMINI_MIN_INTERVAL - (time.time() - _gemini_last)
        if wait > 0:
            time.sleep(wait)
        _gemini_last = time.time()
        try:
            resp = _gemini.models.generate_content(
                model=EXTRACT_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0, response_mime_type="application/json",
                    http_options=types.HttpOptions(timeout=int(REQUEST_TIMEOUT * 1000)),
                ),
            )
            return json.loads(resp.text)
        except Exception as e:
            if _is_retryable(e) and attempt < retries - 1:
                time.sleep(_retry_seconds(e, fallback=10) + 1)
                continue
            raise


def ask_groq(system: str, user: str, retries: int = 5, json_mode: bool = False,
             model: str = None, timeout: float = None) -> str:
    """Send a system + user prompt to Groq, get text (or JSON text) back.

    Used by the Part 2 rule summariser. `model`/`timeout` let cheap calls use the fast
    model and fail fast. Retries on rate-limit/timeout so a throttled or stalled call
    waits and retries instead of hanging.
    """
    return _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        retries=retries, json_mode=json_mode, model=model or SMART_MODEL, timeout=timeout,
    )


# --- OpenRouter: an alternate free provider, used for the Part 2 rule summariser ----
# Kept deliberately tiny (stdlib urllib, no extra SDK). Lets Part 2's one learning call
# live on a different platform/key from Part 1, per the project setup.
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")


def ask_openrouter(system: str, user: str, json_mode: bool = False,
                   timeout: float = 20) -> str:
    """Send system+user to OpenRouter, return text. Raises on any failure so the caller
    can fall back (the Part 2 summariser falls back to Groq)."""
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")
    body = {"model": OPENROUTER_MODEL, "temperature": 0,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}]}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.load(resp)
    return out["choices"][0]["message"]["content"].strip()
