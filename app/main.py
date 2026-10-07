import asyncio
import json
import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.audio import AudioValidationError, decode_audio

# TODO(auth): public demo runs without API keys. To re-enable, restore
# `auth=Depends(verify_api_key)` on the POST routes (see app/auth.py).
from app.inference import predict_with_meta
from app.schemas import DetectRequest, DetectResponse

app = FastAPI(title="AI Generated Voice Detection API")


def _client_key(request: Request) -> str:
    """Rate-limit key that respects proxies (HF Spaces fronts with one)."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_client_key, default_limits=["30/minute"])
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    # Same envelope as every other error; Retry-After lets the UI count down.
    return JSONResponse(
        status_code=429,
        headers={"Retry-After": "60"},
        content={
            "status": "error",
            "message": "Scan budget exhausted — try again in a minute.",
        },
    )


# Harmless same-origin setup today; required if the console ever splits
# hosting (e.g. Vercel frontend + Spaces API).
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


def _load_serving_config() -> dict:
    """Read training-time config (window length, suggested threshold).

    Falls back to safe defaults when the file is absent (e.g. a deploy that
    ships only detector.pt), so serving never crashes on a missing file.
    """
    for candidate in (
        os.getenv("DETECTOR_CONFIG_PATH", "model/detector_config.json"),
        "./model/detector_config.json",
    ):
        try:
            with open(candidate) as f:
                cfg = json.load(f)
            if isinstance(cfg, dict):
                return cfg
        except (OSError, ValueError):
            continue
    return {}


_SERVING_CONFIG = _load_serving_config()

# Per-language decision thresholds. The acoustic model itself is
# language-agnostic, so thresholds are the honest place where `language`
# affects the outcome. Base value comes from training
# (model/detector_config.json's Youden-J suggestion, 0.85 for the current
# weights — the model is recall-heavy on AI, so 0.5 would over-flag humans);
# each language can still be overridden via THRESHOLD_* env vars. Tune per
# language on a held-out set (see training/evaluate.py).
DEFAULT_THRESHOLD = float(
    os.getenv("DETECTOR_THRESHOLD", _SERVING_CONFIG.get("suggested_threshold", 0.5))
)
LANGUAGE_THRESHOLDS = {
    "Tamil": float(os.getenv("THRESHOLD_TAMIL", DEFAULT_THRESHOLD)),
    "English": float(os.getenv("THRESHOLD_ENGLISH", DEFAULT_THRESHOLD)),
    "Hindi": float(os.getenv("THRESHOLD_HINDI", DEFAULT_THRESHOLD)),
    "Malayalam": float(os.getenv("THRESHOLD_MALAYALAM", DEFAULT_THRESHOLD)),
    "Telugu": float(os.getenv("THRESHOLD_TELUGU", DEFAULT_THRESHOLD)),
}
MODEL_VERSION = os.getenv("MODEL_VERSION", "detector.pt")


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "platform": "huggingface_spaces",
        "version": "1.2.0",
        "model": MODEL_VERSION,
        "threshold": DEFAULT_THRESHOLD,
        "best_val_acc": _SERVING_CONFIG.get("best_val_acc"),
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Surface the real pydantic message instead of a generic failure.
    try:
        detail = exc.errors()[0]
        message = f"Invalid request: {'.'.join(map(str, detail['loc']))}: {detail['msg']}"
    except Exception:
        message = "Malformed request"
    return JSONResponse(status_code=400, content={"status": "error", "message": message})


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Never leak internal exception names/details to callers.
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"},
    )


async def _classify(language: str, audio_b64: str) -> DetectResponse:
    try:
        audio = decode_audio(audio_b64)
    except AudioValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Torch releases the GIL during inference; run off the event loop so the
    # server stays responsive under concurrent requests.
    meta = await asyncio.to_thread(predict_with_meta, audio)
    fake_prob: float = meta["fake_prob"]

    threshold = LANGUAGE_THRESHOLDS.get(language, DEFAULT_THRESHOLD)
    is_fake = fake_prob >= threshold
    label = "AI_GENERATED" if is_fake else "HUMAN"
    display_confidence = fake_prob if is_fake else (1.0 - fake_prob)

    if is_fake:
        explanation = (
            "Detected synthetic spectral patterns consistent with AI voice generation."
        )
    else:
        explanation = "Natural speech patterns and physiological micro-tremors detected."
    if meta["num_windows"] > 1:
        explanation += f" (averaged over {meta['num_windows']} segments)"

    return DetectResponse(
        status="success",
        language=language,
        classification=label,
        confidenceScore=round(display_confidence, 4),
        explanation=explanation,
    )


# Primary endpoint (no auth in public-demo mode; rate-limited instead)
@app.post("/api/voice-detection", response_model=DetectResponse)
@limiter.limit("5/minute")
async def detect_voice(request: Request, payload: DetectRequest):
    return await _classify(payload.language, payload.audioBase64)


# Serve the React console when a production build exists. Vite emits
# index.html + favicon.svg at root and hashed files under assets/, so mount
# assets directly and fall back to index.html for every other non-API path
# (BrowserRouter needs /model to resolve client-side). API routes above keep
# precedence because they are registered first.
_DIST = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_DIST):
    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(_DIST, "assets")),
        name="console-assets",
    )

    def _spa_file(name: str) -> FileResponse:
        return FileResponse(os.path.join(_DIST, name))

    @app.get("/", include_in_schema=False)
    async def _console_root():
        return _spa_file("index.html")

    @app.get("/favicon.svg", include_in_schema=False)
    async def _console_icon():
        return _spa_file("favicon.svg")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def _console_fallback(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not found")
        return _spa_file("index.html")
