import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

base = "openai/whisper-small"
adapter = "outputs/whisper_lora_small"

processor = WhisperProcessor.from_pretrained(adapter)
model = WhisperForConditionalGeneration.from_pretrained(
    base,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True
)
model = PeftModel.from_pretrained(model, adapter)
model.eval()

print("Base Whisper + LoRA adapter loaded successfully.")
print("Encoder hidden size:", model.config.d_model)
print("Expected audio sample rate:",
      processor.feature_extractor.sampling_rate)
