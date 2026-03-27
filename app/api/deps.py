from fastapi import Header, HTTPException, Request

from app.config import get_settings
from app.gateway.dispatcher import Dispatcher
from app.providers.base import Provider


def verify_api_key(x_api_key: str = Header(...)) -> None:
    if x_api_key != get_settings().gateway_api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")


def get_providers(request: Request) -> dict[str, Provider]:
    return request.app.state.providers


def get_dispatcher(request: Request) -> Dispatcher:
    return Dispatcher(providers=get_providers(request), settings=get_settings())
