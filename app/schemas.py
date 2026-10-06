from pydantic import BaseModel, Field, field_validator


class DetectRequest(BaseModel):
    language: str = Field(..., description="Language of the audio (Tamil, English, Hindi, Malayalam, Telugu)")
    audioFormat: str = Field("mp3", description="Audio format: mp3, wav, or flac")
    audioBase64: str = Field(..., description="Base64 encoded audio")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        allowed = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
        if v not in allowed:
            raise ValueError(f"Language must be one of {allowed}")
        return v

    @field_validator("audioFormat")
    @classmethod
    def validate_audio_format(cls, v: str) -> str:
        allowed = {"mp3", "wav", "flac"}
        if v.lower() not in allowed:
            raise ValueError(f"audioFormat must be one of {sorted(allowed)}")
        return v.lower()

    @field_validator("audioBase64")
    @classmethod
    def validate_audio_size(cls, v: str) -> str:
        # Import here to avoid a hard dependency cycle at import time.
        from app.audio import MAX_BASE64_CHARS

        if not v:
            raise ValueError("audioBase64 must not be empty")
        if len(v) > MAX_BASE64_CHARS:
            raise ValueError(
                f"audio payload too large ({len(v)} chars, max {MAX_BASE64_CHARS})"
            )
        return v


class DetectResponse(BaseModel):
    status: str
    language: str
    classification: str
    confidenceScore: float
    explanation: str
