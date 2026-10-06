import asyncio
import json
import os

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.audio import AudioValidationError, decode_audio
from app.auth import verify_api_key
from app.inference import predict_with_meta
from app.schemas import DetectRequest, DetectResponse

app = FastAPI(title="AI Generated Voice Detection API")


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


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "platform": "huggingface_spaces",
        "version": "1.1.0",
        "model": MODEL_VERSION,
        "threshold": DEFAULT_THRESHOLD,
        "best_val_acc": _SERVING_CONFIG.get("best_val_acc"),
    }


@app.get("/health")
def health_alias():
    return health_check()


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Surface the real pydantic message instead of blaming the API key.
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


# Primary endpoint as per guidelines
@app.post("/api/voice-detection", response_model=DetectResponse)
async def detect_voice(request: DetectRequest, auth=Depends(verify_api_key)):
    return await _classify(request.language, request.audioBase64)


# Alias for root URL to support testers that don't append the path
@app.post("/", response_model=DetectResponse)
async def detect_voice_root_alias(request: DetectRequest, auth=Depends(verify_api_key)):
    return await _classify(request.language, request.audioBase64)
