from pydantic import BaseModel, Field, validator
from typing import Union

class DetectRequest(BaseModel):
    language: str = Field(..., description="Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)")
    audioFormat: str = Field("mp3", description="Audio format, must be mp3")
    audioBase64: str = Field(..., description="Base64 encoded MP3 audio")

    @validator("language")
    def validate_language(cls, v):
        allowed = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        if v not in allowed:
            raise ValueError(f"Language must be one of {allowed}")
        return v

    @validator("audioFormat")
    def validate_audio_format(cls, v):
        if v.lower() != "mp3":
            raise ValueError("audioFormat must be 'mp3'")
        return "mp3"

class DetectResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str
