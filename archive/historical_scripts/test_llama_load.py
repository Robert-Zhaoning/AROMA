import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_ID)

print("Loading model...")
model = MllamaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model.eval()

print()
print("=== SUCCESS ===")
print("Model:", MODEL_ID)
print("Class:", type(model).__name__)
print("Device:", next(model.parameters()).device)
print("Dtype:", next(model.parameters()).dtype)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Allocated VRAM (GB):", round(torch.cuda.memory_allocated() / 1024**3, 3))
