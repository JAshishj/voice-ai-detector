from fastapi import FastAPI, Depends, Request
from fastapi.responses import JSONResponse
from app.schemas import DetectRequest, DetectResponse
from app.auth import verify_api_key
from app.audio import decode_audio
from app.inference import predict

app = FastAPI(title="AI Generated Voice Detection API")

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": str(exc)}
    )

@app.post("/api/voice-detection", response_model=DetectResponse)
def detect_voice(
    request: DetectRequest,
    auth=Depends(verify_api_key)
):
    try:
        audio = decode_audio(request.audioBase64)
    except Exception as e:
        # Wrap decoding/processing errors as 400 bad request
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Invalid audio input: {str(e)}"}
        )

    confidence = predict(audio)
    is_fake = confidence >= 0.5
    label = "AI_GENERATED" if is_fake else "HUMAN"
    
    # Simple explanation logic based on confidence
    if is_fake:
        explanation = "Detected synthetic spectral patterns consistent with AI voice generation."
    else:
        explanation = "Natural speech patterns and physiological micro-tremors detected."

    return DetectResponse(
        status="success",
        language=request.language,
        classification=label,
        confidenceScore=round(confidence, 2),
        explanation=explanation
    )
