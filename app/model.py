import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import Settings
from app.preprocessor import preprocess_review


class RatingPredictor:
    def __init__(
        self,
        settings: Settings,
        *,
        hf_model_name: str,
        expected_num_labels: int,
    ) -> None:
        self.settings = settings
        self.hf_model_name = hf_model_name
        self.expected_num_labels = expected_num_labels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.model_source: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.tokenizer is not None and self.model is not None

    def load(self) -> None:
        source = self.hf_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(source)
        self.model = AutoModelForSequenceClassification.from_pretrained(source)
        self._validate_loaded_model(source)
        self.model.to(self.device)
        self.model.eval()
        self.model_source = str(source)

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
        hf_model_name=settings.hf_model_name,
        expected_num_labels=settings.expected_num_labels,
    )


def build_arabic_predictor(settings: Settings) -> RatingPredictor:
    return RatingPredictor(
        settings,
        hf_model_name=settings.arabic_hf_model_name,
        expected_num_labels=settings.arabic_expected_num_labels,
    )
