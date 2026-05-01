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
COPY model ./model
COPY model_ar ./model_ar

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
