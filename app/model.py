import httpx

from app.config import Settings
from app.preprocessor import clean_text


class HuggingFaceInferenceError(RuntimeError):
    """Raised when Hugging Face inference fails."""


class HuggingFaceServiceUnavailable(HuggingFaceInferenceError):
    """Raised when the remote model is still cold-starting or unavailable."""


class RatingPredictor:
    def __init__(
        self,
        settings: Settings,
        *,
        hf_model_name: str,
        expected_num_labels: int,
        label_aliases: dict[str, int],
    ) -> None:
        self.settings = settings
        self.hf_model_name = hf_model_name
        self.expected_num_labels = expected_num_labels
        self.label_aliases = label_aliases

    @property
    def is_configured(self) -> bool:
        return bool(self.settings.hf_token)

    @property
    def model_source(self) -> str:
        base_url = self.settings.hf_inference_api_base_url.rstrip("/")
        return f"{base_url}/{self.hf_model_name}"

    async def predict(
        self,
        review: str,
        *,
        client: httpx.AsyncClient,
    ) -> dict[str, object]:
        if not self.is_configured:
            raise HuggingFaceInferenceError("Hugging Face token is not configured")

        prepared_review = clean_text(review)
        payload = await self._request_scores(client, prepared_review)
        scores = self._normalize_scores(payload)
        rating = max(scores, key=scores.get)
        confidence = float(scores[rating])

        return {
            "rating": int(rating),
            "confidence": confidence,
            "label_scores": scores,
        }

    async def _request_scores(
        self,
        client: httpx.AsyncClient,
        prepared_review: str,
    ) -> object:
        try:
            response = await client.post(
                self.model_source,
                headers={
                    "Authorization": f"Bearer {self.settings.hf_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "inputs": prepared_review,
                    "parameters": {
                        "top_k": self.expected_num_labels,
                        "function_to_apply": "softmax",
                    },
                },
            )
        except httpx.TimeoutException as exc:
            raise HuggingFaceServiceUnavailable("Model is warming up") from exc
        except httpx.HTTPError as exc:
            raise HuggingFaceInferenceError(f"Inference API request failed: {exc}") from exc

        return self._parse_payload(response)

    def _parse_payload(self, response: httpx.Response) -> object:
        try:
            payload = response.json()
        except ValueError as exc:
            raise HuggingFaceInferenceError("Inference API returned invalid JSON") from exc

        if response.status_code == 503:
            detail = self._extract_error_message(payload) or "Model is warming up"
            raise HuggingFaceServiceUnavailable(detail)

        if response.status_code >= 400:
            detail = self._extract_error_message(payload) or response.text
            raise HuggingFaceInferenceError(
                f"Inference API request failed with status {response.status_code}: {detail}"
            )

        if isinstance(payload, dict) and payload.get("error"):
            raise HuggingFaceInferenceError(str(payload["error"]))

        return payload

    def _normalize_scores(self, payload: object) -> dict[str, float]:
        rows = payload
        if isinstance(rows, list) and rows and isinstance(rows[0], list):
            rows = rows[0]

        if not isinstance(rows, list):
            raise HuggingFaceInferenceError(
                "Inference API returned an unexpected response shape"
            )

        label_scores = {str(index): 0.0 for index in range(1, self.expected_num_labels + 1)}

        for item in rows:
            if isinstance(item, dict):
                label = item.get("label")
                score = item.get("score")
            else:
                label = getattr(item, "label", None)
                score = getattr(item, "score", None)
            if label is None or score is None:
                continue

            rating = self._map_label_to_rating(str(label))
            label_scores[str(rating)] = float(score)

        if not any(label_scores.values()):
            raise HuggingFaceInferenceError(
                "Inference API returned no usable label scores"
            )

        return label_scores

    def _map_label_to_rating(self, label: str) -> int:
        normalized = " ".join(
            label.strip().lower().replace("-", " ").replace("_", " ").split()
        )

        if normalized in self.label_aliases:
            return self.label_aliases[normalized]

        numeric_suffix = normalized.removeprefix("label ").strip()
        for candidate in (normalized, numeric_suffix):
            if candidate.isdigit():
                value = int(candidate)
                if 1 <= value <= self.expected_num_labels:
                    return value
                if 0 <= value < self.expected_num_labels:
                    return value + 1

        raise HuggingFaceInferenceError(f"Unsupported label returned by model: {label}")

    @staticmethod
    def _extract_error_message(payload: object) -> str | None:
        if isinstance(payload, dict):
            message = payload.get("error") or payload.get("message")
            if isinstance(message, str):
                return message
        return None


def build_primary_predictor(settings: Settings) -> RatingPredictor:
    return RatingPredictor(
        settings,
        hf_model_name=settings.hf_model_name,
        expected_num_labels=settings.expected_num_labels,
        label_aliases={
            "1": 1,
            "2": 2,
            "3": 3,
            "4": 4,
            "5": 5,
            "very negative": 1,
            "negative": 2,
            "neutral": 3,
            "positive": 4,
            "very positive": 5,
            "label 0": 1,
            "label 1": 2,
            "label 2": 3,
            "label 3": 4,
            "label 4": 5,
        },
    )


def build_arabic_predictor(settings: Settings) -> RatingPredictor:
    return RatingPredictor(
        settings,
        hf_model_name=settings.arabic_hf_model_name,
        expected_num_labels=settings.arabic_expected_num_labels,
        label_aliases={
            "0": 1,
            "1": 2,
            "2": 3,
            "3": 4,
            "4": 5,
            "very negative": 1,
            "negative": 2,
            "neutral": 3,
            "positive": 4,
            "very positive": 5,
            "label 0": 1,
            "label 1": 2,
            "label 2": 3,
            "label 3": 4,
            "label 4": 5,
            "poor": 1,
            "fair": 2,
            "good": 3,
            "very good": 4,
            "excellent": 5,
        },
    )
