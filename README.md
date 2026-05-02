# NLP Rating Service

`nlp-rating-service` is a FastAPI API that predicts 1-5 ratings for English and Arabic reviews by calling the Hugging Face Inference API. The app keeps the same REST endpoints as before, but it no longer downloads or serves transformer weights locally.

## Architecture

- `POST /predict` sends English review text to `nhull/distilbert-sentiment-model`
- `POST /predict-ar` sends Arabic review text to `mohres/Arabic-Book-Review-Sentiment-Assessment`
- Hugging Face returns label scores
- the service maps those labels into the shared `1..5` response format

## Required Environment Variables

The real token must stay out of Git. Use a local `.env` file or Azure App Settings.

Copy `.env.example` to `.env` and set:

```env
NLP_RATING_HF_TOKEN=your-huggingface-token
```

Optional overrides:

```env
NLP_RATING_HF_MODEL_NAME=nhull/distilbert-sentiment-model
NLP_RATING_ARABIC_HF_MODEL_NAME=mohres/Arabic-Book-Review-Sentiment-Assessment
NLP_RATING_INFERENCE_TIMEOUT_SECONDS=60
NLP_RATING_FORWARD_TIMEOUT_SECONDS=10
```

## Install And Run Locally

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` from `.env.example`, then start the API:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/health`
- `http://127.0.0.1:8000/docs`

## Docker

The Docker image is now lightweight because it only needs the FastAPI app and HTTP client dependencies.

Build and run:

```powershell
docker compose up --build
```

The compose file reads environment variables from `.env`.

## API Endpoints

### `GET /`

Returns basic service metadata and useful links.

### `GET /health`

Returns:

- `ok` when the Hugging Face token is configured
- `degraded` when `NLP_RATING_HF_TOKEN` is missing

### `POST /predict`

Request body:

```json
{
  "review": "Amazing hotel, clean room, great staff."
}
```

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

### `POST /predict-ar`

Request body:

```json
{
  "review": "الفندق ممتاز والخدمة رائعة"
}
```

## Azure App Service Notes

For Azure App Service, set these application settings:

```text
NLP_RATING_HF_TOKEN=your-token
SCM_DO_BUILD_DURING_DEPLOYMENT=true
```

Startup command:

```text
gunicorn -w 1 -k uvicorn.workers.UvicornWorker --timeout 600 -b 0.0.0.0:$PORT app.main:app
```

Health check path:

```text
/health
```

## Notes

- this project now depends on Hugging Face availability for inference
- the first request may be slower if the hosted model is cold
- if Hugging Face returns a temporary `503`, the API surfaces that as a `503 Service Unavailable`
