from pydantic import BaseModel, Field

class DetectRequest(BaseModel):
    audio_base64: str
    language: str = Field(..., regex="^(ta|en|hi|ml|te)$")

class DetectResponse(BaseModel):
    classification: str
    confidence: float
    explanation: dict
