from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.deps import get_dispatcher, verify_api_key
from app.gateway.dispatcher import Dispatcher
from app.schemas.request import CompletionRequest

router = APIRouter()


@router.post("/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(
    body: CompletionRequest,
    request: Request,
    dispatcher: Dispatcher = Depends(get_dispatcher),
) -> StreamingResponse:
    request_id = request.state.request_id
    return StreamingResponse(
        dispatcher.stream(body, request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
