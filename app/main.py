from fastapi import FastAPI, Depends, HTTPException
from app.schemas import DetectRequest, DetectResponse
from app.auth import verify_api_key
from app.audio import decode_audio
from app.inference import predict

app = FastAPI(title="AI Generated Voice Detection API")

@app.post("/detect-voice", response_model=DetectResponse)
def detect_voice(
    request: DetectRequest,
    auth=Depends(verify_api_key)
):
    try:
        audio = decode_audio(request.audio_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid audio input")

    confidence = predict(audio)
    label = "AI_GENERATED" if confidence >= 0.5 else "HUMAN"

    return DetectResponse(
        classification=label,
        confidence=round(confidence, 3),
        explanation={
            "spectral_artifacts": round(confidence, 3),
            "temporal_consistency": round(1 - confidence, 3)
        }
    )
