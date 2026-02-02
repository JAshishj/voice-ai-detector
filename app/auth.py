import os
from fastapi import Header, HTTPException

API_KEY = os.getenv("API_KEY")

def verify_api_key(authorization: str = Header(None)):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not set")

    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")
