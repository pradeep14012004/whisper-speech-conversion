import json
import os
import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import Any, Optional, Tuple

try:
    import ollama
except ImportError:  # pragma: no cover
    ollama = None

QWEN3_MODEL = "qwen3:4b"
QWEN3_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return " ".join(text.split())


def _strip_code_fences(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_content(container: Any) -> str:
    if isinstance(container, dict):
        message = container.get("message", {})
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str):
                return content
        if "content" in container:
            content = container["content"]
            if isinstance(content, str):
                return content
    if hasattr(container, "message"):
        message = getattr(container, "message")
        if hasattr(message, "content"):
            return str(message.content)
    if hasattr(container, "content"):
        return str(container.content)
    return str(container)


@lru_cache(maxsize=1)
def get_qwen3_client():
    if ollama is None:
        raise RuntimeError("The local Ollama package is not installed. Install it with: python -m pip install ollama")
    return ollama


def _build_prompt(
    fused_transcript: str,
    audio_transcript: Optional[str] = None,
    av_transcript: Optional[str] = None,
    large_v3_transcript: Optional[str] = None,
    confidence: Optional[float] = None,
) -> str:
    return (
        "You are a conservative Mandarin ASR correction assistant. "
        "Your task is to correct obvious recognition errors using the supplied ASR hypotheses only. "
        "Do not translate Chinese into English. Do not rewrite for style. "
        "Do not invent missing words or infer meaning from context. "
        "Preserve the original spoken meaning, names, numbers and technical terms unless the evidence clearly shows a character error. "
        "If the corrected text is clearly better and still conservative, return it even when uncertain; set uncertainty_flag to true. "
        "Only keep the original fused transcript when the correction is not supported or the text remains unclear. "
        "Return only valid JSON with the structure: {\"corrected_text\": \"...\", \"correction_made\": true, \"explanation\": \"...\", \"uncertainty_flag\": false}. "
        f"Fused transcript: {fused_transcript}. "
        f"Audio transcript: {audio_transcript or 'empty'}. "
        f"AV transcript: {av_transcript or 'empty'}. "
        f"Large-v3 transcript: {large_v3_transcript or 'empty'}. "
        f"Confidence: {confidence if confidence is not None else 'unknown'}. "
        "Reject unsupported changes, English rewrites, and any output that changes numbers or names. Keep text in Mandarin Chinese."
    )


def _parse_json_response(raw_text: str) -> dict:
    cleaned = _strip_code_fences(raw_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise ValueError("Ollama returned invalid JSON.")
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("Ollama response was not a JSON object.")
    return payload


def _validate_payload(original_text: str, payload: dict) -> bool:
    if not isinstance(payload, dict):
        return False

    required = {"corrected_text", "correction_made", "explanation", "uncertainty_flag"}
    if not required.issubset(payload.keys()):
        return False

    corrected = _normalize_text(payload.get("corrected_text", ""))
    if not corrected:
        return False

    if not re.search(r"[\u4e00-\u9fff]", corrected):
        return False

    if not isinstance(payload.get("correction_made"), bool):
        return False
    if not isinstance(payload.get("uncertainty_flag"), bool):
        return False

    if corrected == original_text:
        return True

    if re.search(r"[A-Za-z]{6,}", corrected) and not re.search(r"[A-Za-z]{6,}", original_text):
        return False

    if re.search(r"[A-Za-z]{3,}", corrected) and sum(1 for ch in corrected if ch.isascii() and ch.isalpha()) > max(2, len(corrected) * 0.2):
        return False

    original_digits = set(re.findall(r"\d+", original_text))
    corrected_digits = set(re.findall(r"\d+", corrected))
    if original_digits and original_digits != corrected_digits:
        return False

    return True


def correct_mandarin_transcription_with_metadata(
    fused_transcript: str,
    audio_transcript: Optional[str] = None,
    av_transcript: Optional[str] = None,
    large_v3_transcript: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Tuple[str, str, str, bool]:
    """Return (original_transcript, corrected_transcript, explanation, uncertainty_flag)."""
    original = _normalize_text(fused_transcript)
    if not original:
        return "", "", "", False

    try:
        client = get_qwen3_client()
        prompt = _build_prompt(
            fused_transcript=original,
            audio_transcript=audio_transcript,
            av_transcript=av_transcript,
            large_v3_transcript=large_v3_transcript,
            confidence=confidence,
        )
        response = client.chat(
            model=QWEN3_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options={"temperature": 0.0, "top_p": 0.2, "num_predict": 256},
        )
        payload = _parse_json_response(_extract_content(response))
        if not _validate_payload(original, payload):
            return original, original, _normalize_text(payload.get("explanation", "")) if isinstance(payload, dict) else "", bool(payload.get("uncertainty_flag", False)) if isinstance(payload, dict) else False

        corrected = _normalize_text(payload.get("corrected_text", ""))
        explanation = _normalize_text(payload.get("explanation", ""))
        uncertainty = bool(payload.get("uncertainty_flag", False))
        if not corrected:
            return original, original, explanation, uncertainty

        if corrected == original:
            return original, original, explanation, uncertainty

        return original, corrected, explanation, uncertainty
    except Exception:
        return original, original, "", False


def correct_mandarin_transcription(
    fused_transcript: str,
    audio_transcript: Optional[str] = None,
    av_transcript: Optional[str] = None,
    large_v3_transcript: Optional[str] = None,
    confidence: Optional[float] = None,
) -> Tuple[str, str]:
    """Return (original_transcript, corrected_transcript). The corrected result is kept conservative."""
    original, corrected, _, _ = correct_mandarin_transcription_with_metadata(
        fused_transcript=fused_transcript,
        audio_transcript=audio_transcript,
        av_transcript=av_transcript,
        large_v3_transcript=large_v3_transcript,
        confidence=confidence,
    )
    return original, corrected
