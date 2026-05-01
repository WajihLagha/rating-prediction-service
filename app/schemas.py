from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator

from app.config import get_settings


settings = get_settings()


class PredictionRequest(BaseModel):
    review: str = Field(..., description="Raw review text to score.")
    forward_to: AnyHttpUrl | None = Field(
        default=None,
        description="Optional Spring Boot endpoint to receive the prediction result.",
    )

    @field_validator("review")
    @classmethod
    def validate_review(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("review must not be empty")
        if len(stripped) > settings.max_review_chars:
            raise ValueError(
                f"review must not exceed {settings.max_review_chars} characters"
            )
        return stripped


class PredictionResponse(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    confidence: float = Field(..., ge=0.0, le=1.0)
    label_scores: dict[str, float]


class RatingStatusResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_source: str | None = None
    arabic_model_loaded: bool = False
    arabic_model_source: str | None = None
