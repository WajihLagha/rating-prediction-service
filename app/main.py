import logging
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.model import (
    HuggingFaceInferenceError,
    HuggingFaceServiceUnavailable,
    RatingPredictor,
    build_arabic_predictor,
    build_primary_predictor,
)
from app.schemas import PredictionRequest, PredictionResponse, RatingStatusResponse


logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = build_primary_predictor(settings)
    app.state.arabic_predictor = build_arabic_predictor(settings)
    app.state.forward_client = httpx.AsyncClient(
        timeout=settings.forward_timeout_seconds,
        trust_env=False,
    )
    app.state.inference_client = httpx.AsyncClient(
        timeout=settings.inference_timeout_seconds,
        trust_env=False,
    )

    yield

    await app.state.forward_client.aclose()
    await app.state.inference_client.aclose()


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
    arabic_predictor: RatingPredictor | None = getattr(
        request.app.state,
        "arabic_predictor",
        None,
    )
    model_loaded = bool(predictor and predictor.is_configured)
    arabic_model_loaded = bool(arabic_predictor and arabic_predictor.is_configured)
    missing_token_error = (
        None if settings.hf_token else "NLP_RATING_HF_TOKEN is not configured"
    )

    return RatingStatusResponse(
        status="ok" if model_loaded and arabic_model_loaded else "degraded",
        model_loaded=model_loaded,
        model_loading=False,
        model_source=getattr(predictor, "model_source", None) if predictor else None,
        model_load_error=missing_token_error,
        arabic_model_loaded=arabic_model_loaded,
        arabic_model_loading=False,
        arabic_model_source=getattr(
            arabic_predictor,
            "model_source",
            None,
        ),
        arabic_model_load_error=missing_token_error,
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    predictor: RatingPredictor | None = getattr(request.app.state, "predictor", None)
    if predictor is None or not predictor.is_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hugging Face token is not configured",
        )

    try:
        prediction = await predictor.predict(
            payload.review,
            client=request.app.state.inference_client,
        )
    except HuggingFaceServiceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except HuggingFaceInferenceError as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"inference failed: {exc}",
        ) from exc

    if payload.forward_to is not None:
        try:
            await request.app.state.forward_client.post(
                str(payload.forward_to),
                json=prediction,
            )
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
    if predictor is None or not predictor.is_configured:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Hugging Face token is not configured",
        )

    try:
        prediction = await predictor.predict(
            payload.review,
            client=request.app.state.inference_client,
        )
    except HuggingFaceServiceUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except HuggingFaceInferenceError as exc:
        logger.exception("Arabic inference failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Arabic inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"arabic inference failed: {exc}",
        ) from exc

    if payload.forward_to is not None:
        try:
            await request.app.state.forward_client.post(
                str(payload.forward_to),
                json=prediction,
            )
        except httpx.HTTPError as exc:
            logger.exception("Forwarding Arabic prediction failed")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"forwarding failed: {exc}",
            ) from exc

    return PredictionResponse(**prediction)
