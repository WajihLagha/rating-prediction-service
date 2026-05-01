import asyncio
import logging
from contextlib import asynccontextmanager

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor = build_primary_predictor(settings)
    arabic_predictor = build_arabic_predictor(settings)
    app.state.predictor = predictor
    app.state.arabic_predictor = arabic_predictor
    app.state.model_loaded = False
    app.state.arabic_model_loaded = False
    app.state.model_load_error = None
    app.state.arabic_model_load_error = None

    try:
        await asyncio.to_thread(predictor.load)
        app.state.model_loaded = predictor.is_loaded
        logger.info("Model loaded from %s", predictor.model_source)
    except Exception as exc:  # pragma: no cover - startup fallback path
        app.state.model_loaded = False
        app.state.model_load_error = str(exc)
        logger.exception("Failed to load model")

    try:
        await asyncio.to_thread(arabic_predictor.load)
        app.state.arabic_model_loaded = arabic_predictor.is_loaded
        logger.info("Arabic model loaded from %s", arabic_predictor.model_source)
    except Exception as exc:  # pragma: no cover - startup fallback path
        app.state.arabic_model_loaded = False
        app.state.arabic_model_load_error = str(exc)
        logger.exception("Failed to load Arabic model")

    yield


app = FastAPI(title=settings.service_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_origin_regex=settings.cors_allow_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=RatingStatusResponse)
async def rating_status(request: Request) -> RatingStatusResponse:
    predictor: RatingPredictor | None = getattr(request.app.state, "predictor", None)
    return RatingStatusResponse(
        status=(
            "ok"
            if request.app.state.model_loaded and request.app.state.arabic_model_loaded
            else "degraded"
        ),
        model_loaded=bool(request.app.state.model_loaded),
        model_source=getattr(predictor, "model_source", None),
        arabic_model_loaded=bool(request.app.state.arabic_model_loaded),
        arabic_model_source=getattr(
            getattr(request.app.state, "arabic_predictor", None),
            "model_source",
            None,
        ),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    predictor: RatingPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="model is not loaded",
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
            detail="arabic model is not loaded",
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
