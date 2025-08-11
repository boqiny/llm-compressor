"""
Example demonstrating how to use the new timing-enabled quantization modifiers.
This shows both LayerQuantizationModifier and TimedQuantizationModifier usage.
"""

import os
from transformers import AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import (
    LayerQuantizationModifier,
    TimedQuantizationModifier,
    quantize_single_layer
)
from llmcompressor.core import State
import torch
import gc

# Set Hugging Face cache directory to current directory
os.environ["HF_HOME"] = os.getcwd()
OUTPUT_FILE_NAME = "first_10_layers_timing_llama3_8b.json"

# Configuration
# MODEL_ID = "meta-llama/Llama-3.2-1B"
MODEL_ID = "meta-llama/Meta-Llama-3-8B"
OUTPUT_DIR = "timing_examples"

def example_1_single_layer():
    """Example 1: Quantize a single layer with timing."""
    print("\n" + "="*60)
    print("Example 1: Single Layer Quantization with Timing")
    print("="*60)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Quantize just layer 0 with timing
    result = quantize_single_layer(
        model=model,
        layer_num=0,
        scheme="FP8_DYNAMIC",
        measure_timing=True,
        output_dir=f"{OUTPUT_DIR}/single_layer"
    )
    
    print(f"\nLayer 0 quantization completed in {result.get('quantization_time', 0):.2f}s")
    print(f"Status: {result['status']}")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()

def example_2_layer_by_layer():
    """Example 2: Layer-by-layer quantization with detailed timing."""
    print("\n" + "="*60)
    print("Example 2: Layer-by-Layer Quantization")
    print("="*60)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Create modifier for layer-by-layer quantization
    modifier = LayerQuantizationModifier(
        scheme="FP8_DYNAMIC",
        measure_timing=True,
        timing_output_dir=f"{OUTPUT_DIR}/layer_by_layer",
        ignore=["lm_head"]
    )
    
    # Initialize and run
    state = State(model=model)
    modifier.on_initialize(state)
    
    # Quantize first 10 layers only
    results = []
    for layer_num in range(10):
        print(f"\nQuantizing layer {layer_num}...")
        result = modifier.quantize_layer(model, layer_num)
        results.append(result)
        print(f"  Time: {result.get('quantization_time', 0):.3f}s")
    
    # Save timing report
    if modifier._timer:
        modifier._timer.save_timing_report(OUTPUT_FILE_NAME)
    
    print(f"\nQuantized {len(results)} layers")
    total_time = sum(r.get('quantization_time', 0) for r in results if r['status'] == 'success')
    print(f"Total time: {total_time:.2f}s")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()

def example_3_timed_oneshot():
    """Example 3: Use TimedQuantizationModifier with oneshot."""
    print("\n" + "="*60)
    print("Example 3: Timed Quantization with oneshot")
    print("="*60)
    
    # Use the TimedQuantizationModifier as a recipe
    recipe = TimedQuantizationModifier(
        targets=["Linear"],
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"],
        enable_timing=True,
        timing_output_dir=f"{OUTPUT_DIR}/timed_oneshot",
        log_module_timing=False  # Set to True for very detailed timing
    )
    
    # Run oneshot with timing
    model = oneshot(
        model=MODEL_ID,
        recipe=recipe,
        output_dir=f"{OUTPUT_DIR}/quantized_model"
    )
    
    print("\nQuantization complete! Check timing reports in:")
    print(f"  {OUTPUT_DIR}/timed_oneshot/")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()

def example_4_custom_layer_targets():
    """Example 4: Quantize specific modules per layer."""
    print("\n" + "="*60)
    print("Example 4: Custom Layer Targets with Timing")
    print("="*60)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Define custom targets for each layer
    # Layer 0: Only attention modules
    # Layer 1: Only MLP modules
    layer_targets = {
        0: [
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
        ],
        1: [
            "model.layers.1.mlp.gate_proj",
            "model.layers.1.mlp.up_proj",
            "model.layers.1.mlp.down_proj",
        ]
    }
    
    modifier = LayerQuantizationModifier(
        layer_targets=layer_targets,
        scheme="FP8_DYNAMIC",
        measure_timing=True,
        timing_output_dir=f"{OUTPUT_DIR}/custom_targets"
    )
    
    state = State(model=model)
    modifier.on_initialize(state)
    
    # Quantize with custom targets
    for layer_num, targets in layer_targets.items():
        print(f"\nQuantizing layer {layer_num} with {len(targets)} targets")
        result = modifier.quantize_layer(model, layer_num, targets)
        print(f"  Time: {result.get('quantization_time', 0):.3f}s")
        print(f"  Modules: {result.get('modules_quantized', 0)}")
    
    # Save report
    if modifier._timer:
        modifier._timer.save_timing_report("custom_targets_timing.json")
    
    del model
    torch.cuda.empty_cache()
    gc.collect()

def main():
    """Run all examples."""
    print("\n" + "#"*60)
    print("# Timing-Enabled Quantization Examples")
    print("#"*60)
    
    # Run examples
    # example_1_single_layer()
    example_2_layer_by_layer()
    # example_3_timed_oneshot()
    # example_4_custom_layer_targets()
    
    print("\n" + "#"*60)
    print("# All examples completed!")
    print("# Check timing reports in: " + OUTPUT_DIR)
    print("#"*60)

if __name__ == "__main__":
    main()