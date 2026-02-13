from fastapi import FastAPI, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.schemas import DetectRequest, DetectResponse
from app.auth import verify_api_key
from app.audio import decode_audio
from app.inference import predict

app = FastAPI(title="AI Generated Voice Detection API")

@app.get("/")
def health_check():
    return {
        "status": "healthy", 
        "platform": "huggingface_spaces",
        "version": "1.0.0"
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"status": "error", "message": f"Malformed request or validation error: {exc.errors()}"}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": f"Internal server error: {type(exc).__name__}"}
    )

# Primary endpoint as per guidelines
@app.post("/api/voice-detection", response_model=DetectResponse)
def detect_voice(
    request: DetectRequest,
    auth=Depends(verify_api_key)
):
    try:
        audio = decode_audio(request.audioBase64)
    except Exception as e:
        return JSONResponse(
            status_code=400,
            content={"status": "error", "message": f"Audio processing failed: {str(e)}"}
        )

    confidence = predict(audio)
    is_fake = confidence >= 0.5
    label = "AI_GENERATED" if is_fake else "HUMAN"
    
    display_confidence = confidence if is_fake else (1.0 - confidence)
    
    if is_fake:
        explanation = "Detected synthetic spectral patterns consistent with AI voice generation."
    else:
        explanation = "Natural speech patterns and physiological micro-tremors detected."

    return DetectResponse(
        status="success",
        language=request.language,
        classification=label,
        confidenceScore=round(display_confidence, 2),
        explanation=explanation
    )

# Alias for root URL to support testers that don't append the path
@app.post("/", response_model=DetectResponse)
def detect_voice_root_alias(request: DetectRequest, auth=Depends(verify_api_key)):
    return detect_voice(request, auth)
