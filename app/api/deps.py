from fastapi import Header, HTTPException, Request

from app.config import get_settings
from app.gateway.dispatcher import Dispatcher


def verify_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != get_settings().gateway_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_dispatcher(request: Request) -> Dispatcher:
    return request.app.state.dispatcher
