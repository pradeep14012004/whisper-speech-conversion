import re
from functools import lru_cache
from typing import Any, Tuple

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

QWEN3_TRANSLATION_MODEL = "qwen3:4b"
DEFAULT_TRANSLATION_MODEL = "qwen3_4b"
OPTIONAL_TRANSLATION_MODELS = {"qwen3_4b": "Qwen3 4B (Recommended)", "nllb_600m": "NLLB-200 600M (Optional)"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _strip_code_fences(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_ollama_message(response: Any) -> Tuple[str, str]:
    if isinstance(response, dict):
        message = response.get("message") or {}
        if isinstance(message, dict):
            content = message.get("content")
            thinking = message.get("thinking")
            return (content if isinstance(content, str) else "", thinking if isinstance(thinking, str) else "")
        content = response.get("content")
        return (content if isinstance(content, str) else "", "")

    message = getattr(response, "message", None)
    if message is None:
        content = getattr(response, "content", "")
        return (content if isinstance(content, str) else "", "")

    content = getattr(message, "content", "")
    thinking = getattr(message, "thinking", "")
    return (content if isinstance(content, str) else "", thinking if isinstance(thinking, str) else "")


def _looks_like_reasoning(text: str) -> bool:
    lower = text.lower()
    markers = [
        "let's", "first,", "next part", "breaking it down", "translates to", "means", "analysis", "reasoning",
        "i'll", "i will", "i should", "i need to", "the user wants", "wait,", "for a positive statement", "good to know", "here's",
        "they specified", "they want", "the sentence is: 今", "source text", "original chinese"
    ]
    return any(marker in lower for marker in markers) or lower.startswith(("okay", "let me", "here is", "here's", "original chinese", "source text", "source:", "translation:", "translated text:"))


def _extract_english_sentence(raw_text: str) -> str:
    text = _normalize_text(_strip_code_fences(raw_text or ""))
    if not text:
        return ""

    if not re.search(r"[A-Za-z]", text):
        return ""

    English_candidates: list[str] = []
    marker_patterns = [
        r"(?i)(?:so the sentence structure is|final english translation|final translation|translation is|translated text is|answer is|result is|the sentence is)\s*[:\-]?\s*([A-Z][^.!?]*[.!?])",
        r"(?i)(?:so the sentence structure is|final english translation|final translation|translation is|translated text is|answer is|result is|the sentence is)\s*[:\-]?\s*(?:\"|“)?([A-Z][^.!?]*[.!?])",
    ]
    for pattern in marker_patterns:
        for match in re.finditer(pattern, text):
            candidate = _normalize_text(match.group(1))
            if not candidate:
                continue
            if _looks_like_reasoning(candidate):
                continue
            if len(re.findall(r"[A-Za-z]", candidate)) < max(6, int(len(candidate) * 0.55)):
                continue
            English_candidates.append(candidate)

    for quote in re.findall(r'["“](.+?)["”]', text):
        clean = _normalize_text(re.sub(r"[\u4e00-\u9fff]+", " ", quote))
        if not clean:
            continue
        if len(re.findall(r"[A-Za-z]", clean)) < max(6, int(len(clean) * 0.55)):
            continue
        if _looks_like_reasoning(clean):
            continue
        English_candidates.append(clean)

    if English_candidates:
        return max(English_candidates, key=len)

    segments = re.split(r"(?<=[.!?])\s+|\n+", text)
    for segment in segments:
        candidate = segment.strip(" -—")
        if not candidate:
            continue
        clean = _normalize_text(re.sub(r"[\u4e00-\u9fff]+", " ", candidate))
        if not clean:
            continue
        if len(re.findall(r"[A-Za-z]", clean)) < max(6, int(len(clean) * 0.55)):
            continue
        lower = clean.lower()
        if lower.startswith(("okay", "let me", "let's", "here is", "here's", "first", "next", "analysis", "reasoning", "original chinese", "source text", "translation:", "the translation should be", "the sentence is", "they specified", "the user wants")):
            continue
        if _looks_like_reasoning(clean):
            continue
        English_candidates.append(clean)

    if English_candidates:
        return English_candidates[-1]

    if not _looks_like_reasoning(text):
        return text
    return ""


@lru_cache(maxsize=1)
def get_qwen3_translation_client():
    if ollama is None:
        raise RuntimeError("The local Ollama package is not installed. Install it with: python -m pip install ollama")
    return ollama


def resolve_translation_model(choice: str) -> str:
    normalized = (choice or "").strip().lower()
    if "nllb" in normalized:
        return "nllb_600m"
    return "qwen3_4b"


def translate_with_qwen3(chinese_text: str) -> str:
    """Translate Simplified Chinese text into English using the local Qwen3 model via Ollama."""
    text = _normalize_text(chinese_text)
    if not text:
        return ""

    try:
        client = get_qwen3_translation_client()
        prompt = (
            "Translate the following Simplified Chinese text into natural, accurate English. "
            "Preserve all original meaning and details. Do not summarize or add information. "
            "Do not explain your answer. Return only the English translation.\n\n"
            f"Chinese text:\n{text}"
        )
        chat_kwargs = {
            "model": QWEN3_TRANSLATION_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.0, "top_p": 0.2, "num_predict": 256},
            "stream": False,
        }
        try:
            response = client.chat(think=False, **chat_kwargs)
        except TypeError:
            response = client.chat(**chat_kwargs)

        content, thinking = _extract_ollama_message(response)
        candidate_source = content or thinking
        if not candidate_source:
            raise ValueError("Qwen3 produced no translation text; both message.content and message.thinking were empty.")

        translated = _extract_english_sentence(candidate_source)
        if not translated:
            if content and not re.search(r"[A-Za-z]", content):
                raise ValueError("Qwen3 translation produced Chinese text instead of English.")
            if content:
                raise ValueError("Qwen3 produced no translation text.")
            raise ValueError("Qwen3 produced no translation text. The model returned reasoning in message.thinking instead of an English translation.")

        if content and not re.search(r"[A-Za-z]", content):
            raise ValueError("Qwen3 translation produced Chinese text instead of English.")
        return translated
    except Exception as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise RuntimeError(f"Qwen3 translation failed: {exc}") from exc
