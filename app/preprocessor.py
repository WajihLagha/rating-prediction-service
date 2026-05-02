import html
import re

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(review: str) -> str:
    normalized = html.unescape(review).lower()
    without_html = HTML_TAG_PATTERN.sub(" ", normalized)
    return WHITESPACE_PATTERN.sub(" ", without_html).strip()
