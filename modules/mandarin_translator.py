import re
from functools import lru_cache
from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

NLLB_MODEL_NAME = "facebook/nllb-200-distilled-600M"
NLLB_SOURCE_LANG = "zho_Hans"
NLLB_TARGET_LANG = "eng_Latn"
NLLB_MAX_CHARS_PER_CHUNK = 350


@lru_cache(maxsize=1)
def get_nllb_model():
    """Load and cache the local NLLB model and tokenizer."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL_NAME, src_lang=NLLB_SOURCE_LANG)
        model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_NAME)
    except Exception as exc:  # pragma: no cover - exercised in tests via monkeypatch
        raise RuntimeError(
            f"Unable to load NLLB model '{NLLB_MODEL_NAME}'. "
            "Ensure the model is available locally or can be downloaded from Hugging Face. "
            f"Original error: {exc}"
        ) from exc

    tokenizer.set_src_lang_special_tokens(NLLB_SOURCE_LANG)

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        model = model.to(device)
    else:
        device = "cpu"
        model = model.to(device)

    model.eval()
    return device, tokenizer, model


def _normalize_input_text(chinese_text: str) -> str:
    if chinese_text is None:
        return ""
    text = str(chinese_text).strip()
    text = text.replace("\u3000", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _split_translation_chunks(text: str, max_chars: int = NLLB_MAX_CHARS_PER_CHUNK) -> List[str]:
    cleaned = _normalize_input_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= max_chars:
        return [cleaned]

    sentence_parts = re.split(r"(?<=[。！？.!?；;])\s*", cleaned)
    if len(sentence_parts) <= 1:
        return [cleaned[i : i + max_chars] for i in range(0, len(cleaned), max_chars) if cleaned[i : i + max_chars]]

    chunks: List[str] = []
    current = ""
    for part in sentence_parts:
        part = part.strip()
        if not part:
            continue
        if not current:
            current = part
            continue
        if len(current) + 1 + len(part) <= max_chars:
            current = f"{current} {part}"
        else:
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks or [cleaned]


def _nllb_generation_kwargs(tokenizer) -> dict:
    target_token_id = tokenizer.convert_tokens_to_ids(NLLB_TARGET_LANG)
    if target_token_id is None or target_token_id == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"NLLB target language token not found: {NLLB_TARGET_LANG}")
    return {
        "forced_bos_token_id": target_token_id,
        "max_new_tokens": 256,
        "num_beams": 4,
        "do_sample": False,
        "early_stopping": True,
    }


def translate_mandarin_to_english(chinese_text: str) -> str:
    """Translate Mandarin Chinese to English using Meta NLLB-200 distilled 600M."""
    text = _normalize_input_text(chinese_text)
    if not text:
        return ""

    device, tokenizer, model = get_nllb_model()
    chunks = _split_translation_chunks(text)
    translated_chunks: List[str] = []

    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.inference_mode():
            generated_tokens = model.generate(
                **inputs,
                **_nllb_generation_kwargs(tokenizer),
            )

        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0].strip()
        if not decoded:
            raise ValueError("NLLB returned an empty translation for a non-empty Mandarin chunk.")
        translated_chunks.append(decoded)

    return " ".join(part.strip() for part in translated_chunks if part.strip())
