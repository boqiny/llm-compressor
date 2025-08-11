from transformers import AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
import os
MAX_SEQUENCE_LENGTH = 1024
NUM_CALIBRATION_SAMPLES = 256

os.environ["HF_HOME"] = os.getcwd()

# model_id = "meta-llama/Llama-3.2-1B"
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto", device_map="auto")

# For FP8_DYNAMIC: No calibration data needed (recommended)
# For FP8_STATIC: Would need calibration data
recipe = QuantizationModifier(
    targets="Linear",           # Target all Linear layers
    scheme="FP8_DYNAMIC",      # or "FP8_BLOCK" for block-wise quantization
    ignore=["lm_head"]         # Keep output layer in full precision
)

# FP8_DYNAMIC doesn't require calibration data, so dataset is optional
oneshot(
    model=model,
    recipe=recipe,
    output_dir="Llama-3-8B-FP8",
)
