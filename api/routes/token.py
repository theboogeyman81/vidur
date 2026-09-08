from __future__ import annotations

import os

from fastapi import APIRouter, HTTPException
from livekit.api import AccessToken, VideoGrants

router = APIRouter()


@router.get("/token")
async def get_token(room: str, identity: str) -> dict:
    if not room or not identity:
        raise HTTPException(status_code=400, detail="room and identity required")
    token = (
        AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_name(identity)
        .with_grants(VideoGrants(room_join=True, room=room))
        .to_jwt()
    )
    return {"token": token, "url": os.environ["LIVEKIT_URL"]}
