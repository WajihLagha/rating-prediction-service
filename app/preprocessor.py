import html
import re

from transformers import PreTrainedTokenizerBase


HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(review: str) -> str:
    normalized = html.unescape(review).lower()
    without_html = HTML_TAG_PATTERN.sub(" ", normalized)
    return WHITESPACE_PATTERN.sub(" ", without_html).strip()


def truncate_to_token_limit(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
) -> str:
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_tokens,
    )
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def preprocess_review(
    review: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
) -> str:
    cleaned = clean_text(review)
    return truncate_to_token_limit(cleaned, tokenizer=tokenizer, max_tokens=max_tokens)
