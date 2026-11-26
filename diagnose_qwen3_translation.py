import os

import ollama

from modules.qwen3_translator import _extract_english_sentence, _extract_ollama_message

TEST_CHINESE = "今天天气很好，我想和朋友一起出去散步。"
EXPECTED_ENGLISH = "The weather is very nice today, and I want to go for a walk with my friends."


def main() -> None:
    print("Qwen3 diagnostic check")
    try:
        response = ollama.chat(
            model="qwen3:4b",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Translate the following Simplified Chinese into natural, accurate English.\n"
                        "Preserve all original meaning and details. Do not summarize or add information.\n"
                        "Do not explain your answer. Return only the English translation.\n\n"
                        f"Chinese text:\n{TEST_CHINESE}"
                    ),
                }
            ],
            think=False,
            options={"temperature": 0.0, "num_predict": 128},
            stream=False,
        )
        content, thinking = _extract_ollama_message(response)
        extracted = _extract_english_sentence(content)
        print("Ollama request succeeded: True")
        print("message.content non-empty:", bool(content))
        print("message.thinking exists:", bool(thinking))
        print("final extracted English translation:", extracted or "<no translation text>")
    except Exception as exc:  # pragma: no cover - diagnostic helper
        print("Ollama request succeeded: False")
        print("error:", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
