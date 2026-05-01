# NLP Rating Service

`nlp-rating-service` is a FastAPI microservice for predicting hotel and place review ratings from text using a DistilBERT-based sequence classification model. It accepts a review, preprocesses the text, runs inference, and returns a predicted rating from 1 to 5 with confidence scores for every label.

The service is designed to be simple to deploy, easy to integrate with other internal services, and flexible enough to load either:

- a fine-tuned model saved in the local `model/` directory
- or a fine-tuned Hugging Face hotel-review model configured with `NLP_RATING_HF_MODEL_NAME`

The service also supports a separate Arabic review endpoint with its own model source:

- a fine-tuned Arabic model saved in the local `model_ar/` directory
- or a fallback Arabic Hugging Face review model configured with `NLP_RATING_ARABIC_HF_MODEL_NAME`

## Features

- FastAPI-based REST API
- DistilBERT sequence classification with 5 rating labels
- Single model load at startup using FastAPI lifespan
- Review text preprocessing before tokenization
- Confidence score and per-label probability output
- Optional forwarding of predictions to a Spring Boot endpoint
- Docker support with cached Hugging Face assets
- CORS enabled for local and internal network clients
- Safer startup behavior that prefers a fine-tuned hotel-review checkpoint and refuses to use an untrained base checkpoint by default

## Project Structure

```text
nlp-rating-service/
|-- app/
|   |-- main.py
|   |-- model.py
|   |-- schemas.py
|   |-- preprocessor.py
|   `-- config.py
|-- model/
|-- Dockerfile
|-- requirements.txt
`-- README.md
```

## How It Works

1. The API starts and loads the tokenizer and model once during application startup.
2. The loaded predictor is stored in `app.state`.
3. A review is submitted to `POST /predict`.
4. The text is cleaned by:
   - converting to lowercase
   - stripping HTML tags
   - removing extra whitespace
   - truncating to 512 tokens
5. The model runs inference with `torch.no_grad()`.
6. Softmax is applied to the logits.
7. The label with the highest score is mapped to a rating from `1` to `5`.
8. The API returns:
   - predicted rating
   - confidence score
   - all label probabilities
9. If `forward_to` is provided, the prediction result is posted to that URL.

## Why the Base Model Is Not Enough

`distilbert-base-uncased` is a pretrained language model, not a trained 1-5 hotel rating model by itself. It understands English well, but it does not know how to map hotel/place reviews to ratings until it is fine-tuned on labeled examples.

Because of that, the service now refuses to serve predictions from the raw base checkpoint. You must provide one of these:

- a fine-tuned model saved in `model/`
- a fine-tuned Hugging Face model ID through `NLP_RATING_HF_MODEL_NAME`

By default, if `model/` is empty, the service uses `nhull/distilbert-sentiment-model`, which is a fine-tuned 1-5 hotel review model trained on TripAdvisor-style data.

This avoids misleading behavior such as predicting the same middle rating for many different reviews.

## Requirements

- Python 3.11
- pip
- Docker optional, for containerized deployment

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Service Locally

Start the API with Uvicorn:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The service will be available at:

```text
http://localhost:8000
```

## API Endpoints

### `GET /health`

Returns the service status and whether the model was loaded successfully.

Example response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_source": "model",
  "arabic_model_loaded": true,
  "arabic_model_source": "model_ar"
}
```

Response fields:

- `status`: service state, typically `ok` or `degraded`
- `model_loaded`: whether the model is ready for inference
- `model_source`: the model path or source used at startup
- `arabic_model_loaded`: whether the Arabic model is ready for inference
- `arabic_model_source`: the Arabic model path or source used at startup

### `POST /predict`

Predicts a review rating from 1 to 5.

Request body:

```json
{
  "review": "This product is excellent and works perfectly.",
  "forward_to": "http://localhost:8080/api/predictions"
}
```

Request fields:

- `review`: required review text
- `forward_to`: optional URL to forward the prediction result to

Example response:

```json
{
  "rating": 5,
  "confidence": 0.91,
  "label_scores": {
    "1": 0.01,
    "2": 0.02,
    "3": 0.03,
    "4": 0.03,
    "5": 0.91
  }
}
```

Response fields:

- `rating`: predicted rating from `1` to `5`
- `confidence`: highest softmax probability
- `label_scores`: probability for each rating label

### `POST /predict-ar`

Predicts a rating for Arabic review text using a separate Arabic-focused review model.

Request body:

```json
{
  "review": "الفندق رائع جداً والخدمة ممتازة والموقع مميز"
}
```

Example response:

```json
{
  "rating": 5,
  "confidence": 0.88,
  "label_scores": {
    "1": 0.01,
    "2": 0.03,
    "3": 0.05,
    "4": 0.03,
    "5": 0.88
  }
}
```

Notes:

- the default Arabic fallback model is `mohres/Arabic-Book-Review-Sentiment-Assessment`
- it is a 5-label Arabic review model, which keeps the response shape consistent with `/predict`
- many public AraBERT sentiment checkpoints are binary or 3-class only, so this fallback is a practical bootstrap until you train a dedicated Arabic hotel/place model

## Validation and Error Handling

The API returns:

- `422 Unprocessable Entity` if the review is empty
- `422 Unprocessable Entity` if the review exceeds 1000 characters before tokenization
- `503 Service Unavailable` if the model is not loaded
- `500 Internal Server Error` if inference fails
- `502 Bad Gateway` if forwarding to `forward_to` fails

## Testing with cURL

Health check:

```bash
curl http://localhost:8000/health
```

Prediction request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"review\":\"Amazing quality and fast delivery.\"}"
```

Arabic prediction request:

```bash
curl -X POST http://localhost:8000/predict-ar \
  -H "Content-Type: application/json" \
  -d "{\"review\":\"الفندق نظيف جداً والموقع ممتاز والخدمة رائعة\"}"
```

## Testing with Postman

### Health endpoint

1. Open Postman.
2. Create a new request.
3. Choose `GET`.
4. Enter `http://localhost:8000/health`.
5. Click `Send`.

### Prediction endpoint

1. Create a new request.
2. Choose `POST`.
3. Enter `http://localhost:8000/predict`.
4. Open the `Body` tab.
5. Select `raw`.
6. Choose `JSON`.
7. Paste:

```json
{
  "review": "Amazing quality and fast delivery."
}
```

8. Click `Send`.

### Prediction with forwarding

Use this body if you want to forward results to another service:

```json
{
  "review": "Amazing quality and fast delivery.",
  "forward_to": "http://localhost:8080/api/predictions"
}
```

### Arabic prediction endpoint

1. Create a new request.
2. Choose `POST`.
3. Enter `http://localhost:8000/predict-ar`.
4. Open the `Body` tab.
5. Select `raw`.
6. Choose `JSON`.
7. Paste:

```json
{
  "review": "الفندق ممتاز والغرف نظيفة جدا لكن الفطور عادي"
}
```

8. Click `Send`.

## Docker Usage

Build the image:

```bash
docker build -t nlp-rating-service .
```

Run the container:

```bash
docker run -p 8000:8000 nlp-rating-service
```

The container exposes port `8000`.

## Model Directory

Place your fine-tuned Hugging Face model files in the `model/` directory for production use. Typical files include:

- `config.json`
- `model.safetensors` or `pytorch_model.bin`
- `tokenizer.json`
- `tokenizer_config.json`
- `vocab.txt`

If a valid saved model is found in `model/`, the service loads it. Otherwise, set `NLP_RATING_HF_MODEL_NAME` to a fine-tuned 5-label checkpoint. If neither is available, the service starts in a degraded state instead of returning misleading predictions from an untrained head.

Default fallback model:

- `nhull/distilbert-sentiment-model`

For Arabic production use, place your fine-tuned Arabic checkpoint in `model_ar/`. If that folder is empty, the service falls back to:

- `mohres/Arabic-Book-Review-Sentiment-Assessment`

## Fine-Tuning for Hotel and Place Reviews

The best path for your app is to fine-tune DistilBERT on hotel/place review data. A training script is included at `scripts/train_tripadvisor.py`.

Default training target:

- dataset: `nhull/tripadvisor-split-dataset-v2`
- model backbone: `distilbert-base-uncased`
- labels: `1` to `5`

Install dependencies first:

```bash
pip install -r requirements.txt
```

Run training:

```bash
python scripts/train_tripadvisor.py
```

This saves the fine-tuned model into `model/`, which the API will load automatically on the next startup.

You can also customize training:

```bash
python scripts/train_tripadvisor.py \
  --dataset nhull/tripadvisor-split-dataset-v2 \
  --epochs 4 \
  --train-batch-size 16 \
  --eval-batch-size 16 \
  --learning-rate 2e-5
```

If you want to train on another dataset, make sure it has:

- a text column
- a rating label column
- labels mapped either as `0-4` or `1-5`

## Configuration

The service reads settings from `app/config.py` and supports environment variable overrides with the prefix `NLP_RATING_`.

Examples:

- `NLP_RATING_MODEL_PATH`
- `NLP_RATING_HF_MODEL_NAME`
- `NLP_RATING_ARABIC_MODEL_PATH`
- `NLP_RATING_ARABIC_HF_MODEL_NAME`
- `NLP_RATING_ALLOW_UNTRAINED_BASE_MODEL`
- `NLP_RATING_MAX_REVIEW_CHARS`
- `NLP_RATING_FORWARD_TIMEOUT_SECONDS`

## Troubleshooting

### Training fails with metric dependency errors

The training script computes accuracy and weighted F1 locally with NumPy, so it should not require `scikit-learn` or Hugging Face `evaluate`. If you still see an old metric-related error, make sure you are running the current version of `scripts/train_tripadvisor.py`.

### API starts in degraded mode

If the API reports `model_loaded: false` on `GET /health`, check these cases:

- `model/` is empty and the configured Hugging Face model could not be downloaded
- `NLP_RATING_HF_MODEL_NAME` points to a model that is not a 5-label classifier
- the machine has network restrictions and cannot fetch the fallback model
- the local checkpoint in `model/` is incomplete or corrupted

The model source priority is:

1. local `model/`
2. `NLP_RATING_HF_MODEL_NAME`
3. never the raw base checkpoint unless `NLP_RATING_ALLOW_UNTRAINED_BASE_MODEL=true`

The Arabic model source priority is:

1. local `model_ar/`
2. `NLP_RATING_ARABIC_HF_MODEL_NAME`

## Development Notes

- The current project scaffold is ready for a fine-tuned DistilBERT classifier.
- For meaningful hotel/place rating predictions, you should provide a trained 5-label model in `model/` or configure a fine-tuned model ID.
- If you explicitly set `NLP_RATING_ALLOW_UNTRAINED_BASE_MODEL=true`, the service can load the raw base checkpoint again, but predictions will not be reliable.
- FastAPI interactive documentation is available at:
  - `http://localhost:8000/docs`
  - `http://localhost:8000/redoc`

## License

Add your preferred license for internal or public distribution.
