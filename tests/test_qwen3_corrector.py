import json
from types import SimpleNamespace

import modules.qwen3_corrector as qwen3


def _mock_ollama_response(payload):
    return {"message": {"content": json.dumps(payload)}}


def test_returns_original_when_ollama_unavailable(monkeypatch):
    monkeypatch.setattr(qwen3, "ollama", None)
    qwen3.get_qwen3_client.cache_clear()
    original, corrected = qwen3.correct_mandarin_transcription("今天的天气很好")
    assert original == "今天的天气很好"
    assert corrected == "今天的天气很好"


def test_accepts_valid_correction(monkeypatch):
    def fake_chat(**kwargs):
        return _mock_ollama_response({
            "corrected_text": "今天的天气很好",
            "correction_made": True,
            "explanation": "Fixed a simplified/traditional character mismatch using the ASR evidence.",
            "uncertainty_flag": False,
        })

    monkeypatch.setattr(qwen3, "ollama", SimpleNamespace(chat=fake_chat))
    qwen3.get_qwen3_client.cache_clear()

    original, corrected = qwen3.correct_mandarin_transcription("今天的天氣很好")
    assert original == "今天的天氣很好"
    assert corrected == "今天的天气很好"


def test_accepts_uncertain_but_usable_correction(monkeypatch):
    def fake_chat(**kwargs):
        return _mock_ollama_response({
            "corrected_text": "今天的天气很好",
            "correction_made": True,
            "explanation": "The audio and AV hypotheses agree on a corrected character.",
            "uncertainty_flag": True,
        })

    monkeypatch.setattr(qwen3, "ollama", SimpleNamespace(chat=fake_chat))
    qwen3.get_qwen3_client.cache_clear()

    original, corrected = qwen3.correct_mandarin_transcription("今天的天氣很好")
    assert original == "今天的天氣很好"
    assert corrected == "今天的天气很好"


def test_accepts_uncertain_but_larger_correction(monkeypatch):
    def fake_chat(**kwargs):
        return _mock_ollama_response({
            "corrected_text": "在夕阳下和朋友一起沿着小路散步，心情很放松。",
            "correction_made": True,
            "explanation": "The ASR output is garbled but the corrected sentence remains conservative and in Mandarin.",
            "uncertainty_flag": True,
        })

    monkeypatch.setattr(qwen3, "ollama", SimpleNamespace(chat=fake_chat))
    qwen3.get_qwen3_client.cache_clear()

    original = "在夕阳的雨位下和好友漫步相见小道心里用笔放松"
    _, corrected = qwen3.correct_mandarin_transcription(original)
    assert corrected == "在夕阳下和朋友一起沿着小路散步，心情很放松。"


def test_rejects_hallucinated_english_change(monkeypatch):
    def fake_chat(**kwargs):
        return _mock_ollama_response({
            "corrected_text": "This sentence is now in English and should be rejected.",
            "correction_made": True,
            "explanation": "The model rewrote the sentence in English.",
            "uncertainty_flag": False,
        })

    monkeypatch.setattr(qwen3, "ollama", SimpleNamespace(chat=fake_chat))
    qwen3.get_qwen3_client.cache_clear()

    original, corrected = qwen3.correct_mandarin_transcription("今天的天气很好")
    assert original == "今天的天气很好"
    assert corrected == "今天的天气很好"


def test_rejects_invalid_json(monkeypatch):
    def fake_chat(**kwargs):
        return {"message": {"content": "not json"}}

    monkeypatch.setattr(qwen3, "ollama", SimpleNamespace(chat=fake_chat))
    qwen3.get_qwen3_client.cache_clear()

    original, corrected = qwen3.correct_mandarin_transcription("今天天气很好")
    assert original == "今天天气很好"
    assert corrected == "今天天气很好"


def test_rejects_empty_output(monkeypatch):
    def fake_chat(**kwargs):
        return _mock_ollama_response({
            "corrected_text": "",
            "correction_made": True,
            "explanation": "No safe correction found.",
            "uncertainty_flag": False,
        })

    monkeypatch.setattr(qwen3, "ollama", SimpleNamespace(chat=fake_chat))
    qwen3.get_qwen3_client.cache_clear()

    original, corrected = qwen3.correct_mandarin_transcription("今天天气很好")
    assert original == "今天天气很好"
    assert corrected == "今天天气很好"


def test_rejects_non_chinese_output(monkeypatch):
    def fake_chat(**kwargs):
        return _mock_ollama_response({
            "corrected_text": "Hello world",
            "correction_made": True,
            "explanation": "The model switched to English.",
            "uncertainty_flag": False,
        })

    monkeypatch.setattr(qwen3, "ollama", SimpleNamespace(chat=fake_chat))
    qwen3.get_qwen3_client.cache_clear()

    original, corrected = qwen3.correct_mandarin_transcription("今天的天气很好")
    assert original == "今天的天气很好"
    assert corrected == "今天的天气很好"


def test_rejects_changed_numbers(monkeypatch):
    def fake_chat(**kwargs):
        return _mock_ollama_response({
            "corrected_text": "今天有23个人",
            "correction_made": True,
            "explanation": "The number changed and should be rejected.",
            "uncertainty_flag": False,
        })

    monkeypatch.setattr(qwen3, "ollama", SimpleNamespace(chat=fake_chat))
    qwen3.get_qwen3_client.cache_clear()

    original, corrected = qwen3.correct_mandarin_transcription("今天有24个人")
    assert original == "今天有24个人"
    assert corrected == "今天有24个人"


def test_handles_empty_input(monkeypatch):
    monkeypatch.setattr(qwen3, "ollama", SimpleNamespace(chat=lambda **kwargs: {}))
    qwen3.get_qwen3_client.cache_clear()
    original, corrected = qwen3.correct_mandarin_transcription("   ")
    assert original == ""
    assert corrected == ""
