import argparse
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


DEFAULT_MODEL_NAME = "distilbert-base-uncased"
DEFAULT_DATASET_NAME = "nhull/tripadvisor-split-dataset-v2"
DEFAULT_TEXT_COLUMN = "review"
DEFAULT_LABEL_COLUMN = "label"
DEFAULT_OUTPUT_DIR = "model"
RATINGS = [1, 2, 3, 4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune DistilBERT for 1-5 hotel/place review rating prediction."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--text-column", default=DEFAULT_TEXT_COLUMN)
    parser.add_argument("--label-column", default=DEFAULT_LABEL_COLUMN)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def normalize_label(raw_label: int | float) -> int:
    label = int(raw_label)
    if label in range(0, 5):
        return label
    if label not in RATINGS:
        raise ValueError(
            "Expected rating label in 0-4 or 1-5 format, "
            f"got {raw_label!r}"
    )
    return label - 1


def compute_weighted_f1(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> float:
    weighted_sum = 0.0
    total_support = int(labels.size)

    for class_index in range(num_classes):
        predicted_positive = predictions == class_index
        actual_positive = labels == class_index

        true_positive = int(np.logical_and(predicted_positive, actual_positive).sum())
        false_positive = int(
            np.logical_and(predicted_positive, np.logical_not(actual_positive)).sum()
        )
        false_negative = int(
            np.logical_and(np.logical_not(predicted_positive), actual_positive).sum()
        )
        support = int(actual_positive.sum())

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )
        recall = true_positive / recall_denominator if recall_denominator > 0 else 0.0
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        weighted_sum += f1 * support

    if total_support == 0:
        return 0.0

    return weighted_sum / total_support


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    dataset = load_dataset(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    label2id = {str(rating): rating - 1 for rating in RATINGS}
    id2label = {rating - 1: str(rating) for rating in RATINGS}

    def preprocess(batch: dict[str, list]) -> dict[str, list]:
        encoded = tokenizer(
            batch[args.text_column],
            truncation=True,
            max_length=args.max_length,
        )
        encoded["labels"] = [
            normalize_label(raw_label) for raw_label in batch[args.label_column]
        ]
        return encoded

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(RATINGS),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    def compute_metrics(eval_pred) -> dict[str, float]:
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        labels = np.asarray(labels)
        accuracy = float((predictions == labels).mean()) if labels.size > 0 else 0.0
        weighted_f1 = compute_weighted_f1(
            predictions=predictions,
            labels=labels,
            num_classes=len(RATINGS),
        )
        return {
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    test_metrics = trainer.evaluate(tokenized["test"])
    print("Saved fine-tuned model to:", output_dir)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
