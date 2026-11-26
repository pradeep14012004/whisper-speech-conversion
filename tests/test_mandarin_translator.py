import pytest

import modules.mandarin_translator as mt


class DummyBatchEncoding(dict):
    def to(self, device):
        return self


class DummyTokenizer:
    def __init__(self, decode_output="This is the English translation."):
        self.lang_code_to_id = {"zho_Hans": 100, "eng_Latn": 200}
        self.unk_token_id = 0
        self.src_lang = None
        self.decode_output = decode_output

    def set_src_lang_special_tokens(self, lang):
        self.src_lang = lang
        return self

    def convert_tokens_to_ids(self, token):
        return self.lang_code_to_id.get(token, self.unk_token_id)

    def to(self, device):
        return self

    def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
        return DummyBatchEncoding({"input_ids": [1, 2, 3]})

    def batch_decode(self, outputs, skip_special_tokens=True):
        return [self.decode_output]


class DummyModel:
    def __init__(self):
        self.generate_calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return [[7, 8, 9]]


def test_nllb_language_codes_and_forced_bos(monkeypatch):
    tokenizer = DummyTokenizer()
    assert mt.NLLB_SOURCE_LANG == "zho_Hans"
    assert mt.NLLB_TARGET_LANG == "eng_Latn"
    kwargs = mt._nllb_generation_kwargs(tokenizer)
    assert kwargs["forced_bos_token_id"] == tokenizer.lang_code_to_id["eng_Latn"]


def test_empty_input_returns_empty_string(monkeypatch):
    assert mt.translate_mandarin_to_english("   ") == ""


def test_successful_translation(monkeypatch):
    tokenizer = DummyTokenizer()
    model = DummyModel()
    monkeypatch.setattr(mt, "get_nllb_model", lambda: ("cpu", tokenizer, model))

    result = mt.translate_mandarin_to_english("今天的天气很好。")

    assert result == "This is the English translation."
    assert model.generate_calls[0]["forced_bos_token_id"] == tokenizer.lang_code_to_id["eng_Latn"]


def test_model_loading_failure(monkeypatch):
    def raise_error(*args, **kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(mt.AutoTokenizer, "from_pretrained", raise_error)
    mt.get_nllb_model.cache_clear()
    with pytest.raises(RuntimeError, match="Unable to load NLLB model"):
        mt.get_nllb_model()


def test_translation_failure(monkeypatch):
    tokenizer = DummyTokenizer(decode_output="")
    model = DummyModel()
    monkeypatch.setattr(mt, "get_nllb_model", lambda: ("cpu", tokenizer, model))

    with pytest.raises(ValueError, match="empty translation"):
        mt.translate_mandarin_to_english("今天的天气很好。")


def test_long_input_chunks(monkeypatch):
    tokenizer = DummyTokenizer()
    model = DummyModel()
    monkeypatch.setattr(mt, "get_nllb_model", lambda: ("cpu", tokenizer, model))

    long_text = "今天的天气很好，" * 80
    result = mt.translate_mandarin_to_english(long_text)

    assert "This is the English translation." in result
    assert len(model.generate_calls) >= 2


def test_translation_uses_exact_final_mandarin_text(monkeypatch):
    seen = {}

    class CapturingTokenizer(DummyTokenizer):
        def __call__(self, text, return_tensors=None, padding=False, truncation=False, max_length=None):
            seen["text"] = text
            return DummyBatchEncoding({"input_ids": [1, 2, 3]})

    tokenizer = CapturingTokenizer()
    model = DummyModel()
    monkeypatch.setattr(mt, "get_nllb_model", lambda: ("cpu", tokenizer, model))

    final_text = "在夕阳下和朋友一起沿着小路散步，心情很放松。"
    mt.translate_mandarin_to_english(final_text)

    assert seen["text"] == final_text
