FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoTokenizer.from_pretrained('nhull/distilbert-sentiment-model', cache_dir='/opt/huggingface'); AutoModelForSequenceClassification.from_pretrained('nhull/distilbert-sentiment-model', cache_dir='/opt/huggingface')"
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; AutoTokenizer.from_pretrained('mohres/Arabic-Book-Review-Sentiment-Assessment', cache_dir='/opt/huggingface'); AutoModelForSequenceClassification.from_pretrained('mohres/Arabic-Book-Review-Sentiment-Assessment', cache_dir='/opt/huggingface')"

COPY app ./app

EXPOSE 8000

CMD ["gunicorn", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "600", "-b", "0.0.0.0:8000", "app.main:app"]
