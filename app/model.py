from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import Settings
from app.preprocessor import preprocess_review


class RatingPredictor:
    def __init__(
        self,
        settings: Settings,
        *,
        model_path: Path,
        hf_model_name: str,
        expected_num_labels: int,
        guard_untrained_distilbert: bool = False,
    ) -> None:
        self.settings = settings
        self.model_path = Path(model_path)
        self.hf_model_name = hf_model_name
        self.expected_num_labels = expected_num_labels
        self.guard_untrained_distilbert = guard_untrained_distilbert
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.model_source: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.tokenizer is not None and self.model is not None

    def load(self) -> None:
        source = self._resolve_model_source()
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self._validate_loaded_model(source)
        self.model.to(self.device)
        self.model.eval()
        self.model_source = str(source)

    def _resolve_model_source(self) -> str:
        if (self.model_path / "config.json").exists():
            return str(self.model_path)
        if (
            self.guard_untrained_distilbert
            and self.hf_model_name == "distilbert-base-uncased"
            and not self.settings.allow_untrained_base_model
        ):
            raise RuntimeError(
                "No fine-tuned model was found in the model/ directory. "
                "Refusing to serve predictions from the raw distilbert-base-uncased "
                "checkpoint because its 1-5 rating head is not trained. "
                "Fine-tune a hotel/place review model and save it into model/, "
                "or set NLP_RATING_HF_MODEL_NAME to a fine-tuned 5-label checkpoint."
            )
        return self.hf_model_name

    def _validate_loaded_model(self, source: str) -> None:
        num_labels = getattr(self.model.config, "num_labels", None)
        if num_labels != self.expected_num_labels:
            raise RuntimeError(
                f"Loaded model '{source}' exposes {num_labels} labels, "
                f"expected {self.expected_num_labels}."
            )

    def predict(self, review: str) -> dict[str, object]:
        if not self.is_loaded:
            raise RuntimeError("model is not loaded")

        prepared_review = preprocess_review(
            review=review,
            tokenizer=self.tokenizer,
            max_tokens=self.settings.max_model_tokens,
        )
        encoded = self.tokenizer(
            prepared_review,
            return_tensors="pt",
            truncation=True,
            max_length=self.settings.max_model_tokens,
            padding=False,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}

        with torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).squeeze(0)

        rating_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[rating_index].item())
        label_scores = {
            str(label + 1): float(score)
            for label, score in enumerate(probabilities.tolist())
        }

        return {
            "rating": rating_index + 1,
            "confidence": confidence,
            "label_scores": label_scores,
        }


def build_primary_predictor(settings: Settings) -> RatingPredictor:
    return RatingPredictor(
        settings,
        model_path=settings.model_path,
        hf_model_name=settings.hf_model_name,
        expected_num_labels=settings.expected_num_labels,
        guard_untrained_distilbert=True,
    )


def build_arabic_predictor(settings: Settings) -> RatingPredictor:
    return RatingPredictor(
        settings,
        model_path=settings.arabic_model_path,
        hf_model_name=settings.arabic_hf_model_name,
        expected_num_labels=settings.arabic_expected_num_labels,
    )
