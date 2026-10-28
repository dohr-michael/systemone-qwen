"""HTTP API: System One decisions over the OpenRouter / TypeSafe wire format."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import __version__
from .backend import Backend, BackendError
from .config import Settings, load_settings
from .decide import UnsupportedRequest, decide
from .schema import DecisionRequest


def error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "code": status}})


def create_app(settings: Settings | None = None, backends: dict[str, Backend] | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings or load_settings()
        app.state.backends = backends or {name: Backend(cfg) for name, cfg in app.state.settings.models.items()}
        yield
        for backend in app.state.backends.values():
            await backend.close()

    app = FastAPI(title="systemone-qwen", version=__version__, lifespan=lifespan)

    # OpenRouter SDKs call {server_url}/systemone with server_url ending in /api/v1; TypeSafe's
    # own path is /v1/systemone; /api/alpha/decisions is OpenRouter's earlier alpha path.
    @app.post("/api/v1/systemone")
    @app.post("/v1/systemone")
    @app.post("/api/alpha/decisions")
    async def systemone(payload: DecisionRequest, request: Request):
        try:
            request.app.state.settings.resolve(payload.model)
        except KeyError:
            return error(404, f"unknown model {payload.model!r}; see GET /v1/models")
        try:
            return await decide(request.app.state.settings, request.app.state.backends, payload)
        except UnsupportedRequest as unsupported:
            return error(422, str(unsupported))
        except BackendError as failure:
            return error(502, f"model backend failed: {failure}")

    @app.get("/v1/models")
    @app.get("/api/v1/models")
    async def models(request: Request):
        settings: Settings = request.app.state.settings
        data = [{"id": name, "object": "model", "owned_by": settings.provider} for name in settings.models]
        data += [{"id": alias, "object": "model", "owned_by": settings.provider, "alias_of": target}
                 for alias, target in settings.aliases.items()]
        return {"object": "list", "data": data}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/ready")
    async def ready(request: Request):
        status = {name: await backend.healthy() for name, backend in request.app.state.backends.items()}
        return JSONResponse(status_code=200 if all(status.values()) else 503, content={"backends": status})

    return app


def main() -> None:
    uvicorn.run(
        create_app(),
        host=os.environ.get("SYSTEMONE_HOST", "0.0.0.0"),
        port=int(os.environ.get("SYSTEMONE_PORT", "8000")),
        log_level=os.environ.get("SYSTEMONE_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
