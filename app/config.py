from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="NLP_RATING_",
        env_file=".env",
        extra="ignore",
    )

    service_name: str = "nlp-rating-service"
    hf_token: str | None = Field(default=None, repr=False)
    hf_inference_base_url: str = "https://router.huggingface.co/hf-inference/models"
    hf_model_name: str = "nhull/distilbert-sentiment-model"
    arabic_hf_model_name: str = "mohres/Arabic-Book-Review-Sentiment-Assessment"
    inference_timeout_seconds: float = 60.0
    expected_num_labels: int = 5
    arabic_expected_num_labels: int = 5
    max_review_chars: int = 1000
    forward_timeout_seconds: float = 10.0
    low_confidence_threshold: float = 0.0
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost",
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8080",
        ]
    )
    cors_allow_origin_regex: str = (
        r"^https?://("
        r"localhost|127\.0\.0\.1|"
        r"10(?:\.\d{1,3}){3}|"
        r"192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2}"
        r")(?::\d{1,5})?$"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
