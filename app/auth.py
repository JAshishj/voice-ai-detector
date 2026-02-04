import os
from fastapi import Header, HTTPException

API_KEY = os.getenv("API_KEY")

def verify_api_key(x_api_key: str = Header(..., alias="x-api-key")):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not set on server")

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key or malformed request"
        )
