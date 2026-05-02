import asyncio
import logging
from contextlib import asynccontextmanager
from contextlib import suppress

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.model import (
    RatingPredictor,
    build_arabic_predictor,
    build_primary_predictor,
)
from app.schemas import PredictionRequest, PredictionResponse, RatingStatusResponse


logger = logging.getLogger(__name__)
settings = get_settings()


async def _load_predictor_state(
    app: FastAPI,
    *,
    predictor_attr: str,
    state_prefix: str,
    log_label: str,
) -> None:
    predictor: RatingPredictor = getattr(app.state, predictor_attr)

    setattr(app.state, f"{state_prefix}_loading", True)
    setattr(app.state, f"{state_prefix}_load_error", None)

    try:
        await asyncio.to_thread(predictor.load)
        setattr(app.state, f"{state_prefix}_loaded", predictor.is_loaded)
        logger.info("%s model loaded from %s", log_label, predictor.model_source)
    except Exception as exc:  # pragma: no cover - startup fallback path
        setattr(app.state, f"{state_prefix}_loaded", False)
        setattr(app.state, f"{state_prefix}_load_error", str(exc))
        logger.exception("Failed to load %s model", log_label)
    finally:
        setattr(app.state, f"{state_prefix}_loading", False)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = build_primary_predictor(settings)
    arabic_predictor = build_arabic_predictor(settings)
    app.state.predictor = predictor
    app.state.arabic_predictor = arabic_predictor
    app.state.model_loaded = False
    app.state.model_loading = True
    app.state.model_load_error = None
    app.state.arabic_model_loaded = False
    app.state.arabic_model_loading = True
    app.state.arabic_model_load_error = None

    primary_task = asyncio.create_task(
        _load_predictor_state(
            app,
            predictor_attr="predictor",
            state_prefix="model",
            log_label="primary",
        )
    )
    arabic_task = asyncio.create_task(
        _load_predictor_state(
            app,
            predictor_attr="arabic_predictor",
            state_prefix="arabic_model",
            log_label="arabic",
        )
    )

    yield

    for task in (primary_task, arabic_task):
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(title=settings.service_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict[str, object]:
    return {
        "service": settings.service_name,
        "docs_url": "/docs",
        "health_url": "/health",
        "predict_url": "/predict",
        "predict_ar_url": "/predict-ar",
    }


@app.get("/health", response_model=RatingStatusResponse)
async def rating_status(request: Request) -> RatingStatusResponse:
    predictor: RatingPredictor | None = getattr(request.app.state, "predictor", None)
    model_loaded = bool(request.app.state.model_loaded)
    model_loading = bool(getattr(request.app.state, "model_loading", False))
    arabic_model_loaded = bool(request.app.state.arabic_model_loaded)
    arabic_model_loading = bool(
        getattr(request.app.state, "arabic_model_loading", False)
    )

    return RatingStatusResponse(
        status=(
            "ok"
            if model_loaded and arabic_model_loaded
            else "loading"
            if model_loading or arabic_model_loading
            else "degraded"
        ),
        model_loaded=model_loaded,
        model_loading=model_loading,
        model_source=getattr(predictor, "model_source", None),
        model_load_error=getattr(request.app.state, "model_load_error", None),
        arabic_model_loaded=arabic_model_loaded,
        arabic_model_loading=arabic_model_loading,
        arabic_model_source=getattr(
            getattr(request.app.state, "arabic_predictor", None),
            "model_source",
            None,
        ),
        arabic_model_load_error=getattr(
            request.app.state,
            "arabic_model_load_error",
            None,
        ),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    predictor: RatingPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=getattr(request.app.state, "model_load_error", None)
            or "model is still loading",
        )

    try:
        prediction = await asyncio.to_thread(predictor.predict, payload.review)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"inference failed: {exc}",
        ) from exc

    if payload.forward_to is not None:
        try:
            async with httpx.AsyncClient(
                timeout=settings.forward_timeout_seconds
            ) as client:
                await client.post(str(payload.forward_to), json=prediction)
        except httpx.HTTPError as exc:
            logger.exception("Forwarding prediction failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"forwarding failed: {exc}",
            ) from exc

    return PredictionResponse(**prediction)


@app.post("/predict-ar", response_model=PredictionResponse)
async def predict_arabic(
    payload: PredictionRequest,
    request: Request,
) -> PredictionResponse:
    predictor: RatingPredictor | None = getattr(
        request.app.state,
        "arabic_predictor",
        None,
    )
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=getattr(request.app.state, "arabic_model_load_error", None)
            or "arabic model is still loading",
        )

    try:
        prediction = await asyncio.to_thread(predictor.predict, payload.review)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Arabic inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"arabic inference failed: {exc}",
        ) from exc

    if payload.forward_to is not None:
        try:
            async with httpx.AsyncClient(
                timeout=settings.forward_timeout_seconds
            ) as client:
                await client.post(str(payload.forward_to), json=prediction)
        except httpx.HTTPError as exc:
            logger.exception("Forwarding Arabic prediction failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"forwarding failed: {exc}",
            ) from exc

    return PredictionResponse(**prediction)
