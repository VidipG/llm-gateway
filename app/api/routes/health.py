from fastapi import APIRouter, Request

router = APIRouter()


@router.get("")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/providers")
async def provider_health(request: Request) -> dict:
    providers = request.app.state.providers
    results = {}
    for name, provider in providers.items():
        ok = await provider.health_check()
        results[name] = "ok" if ok else "unreachable"
    return results
