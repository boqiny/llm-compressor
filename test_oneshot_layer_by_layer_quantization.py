"""
Test script for layer-by-layer FP8 quantization with timing measurements.
This script quantizes each layer individually and logs the time taken for each.
Fixed version: Reloads model for each layer to avoid FP8 tensor conflicts.
"""

import time
import json
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
import torch
import gc
import os

# Configuration
MODEL_ID = "meta-llama/Llama-3.2-1B"
OUTPUT_BASE_DIR = "layer_by_layer_fp8_test_fixed"
TIMING_LOG_FILE = "quantization_timing_log.json"

def get_layer_targets(layer_num):
    """Get all Linear module names for a specific layer."""
    return [
        f"model.layers.{layer_num}.self_attn.q_proj",
        f"model.layers.{layer_num}.self_attn.k_proj",
        f"model.layers.{layer_num}.self_attn.v_proj",
        f"model.layers.{layer_num}.self_attn.o_proj",
        f"model.layers.{layer_num}.mlp.gate_proj",
        f"model.layers.{layer_num}.mlp.up_proj",
        f"model.layers.{layer_num}.mlp.down_proj",
    ]

def quantize_single_layer(model_id, layer_num, output_dir):
    """Load model, quantize a single layer, and return timing information."""
    
    print(f"\n{'='*60}")
    print(f"Quantizing Layer {layer_num}")
    print(f"{'='*60}")
    
    # Load fresh model for each layer
    print(f"  Loading fresh model instance...")
    model_load_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype="auto", 
        device_map="auto"
    )
    model_load_time = time.time() - model_load_start
    print(f"  Model loaded in {model_load_time:.2f}s")
    
    # Get targets for this layer
    targets = get_layer_targets(layer_num)
    
    # Create recipe for this specific layer
    recipe = QuantizationModifier(
        targets=targets,
        scheme="FP8_DYNAMIC",
        ignore=["lm_head"]
    )
    
    # Measure quantization time
    quant_start_time = time.time()
    
    try:
        oneshot(
            model=model,
            recipe=recipe,
            output_dir=f"{output_dir}/layer_{layer_num}",
        )
        
        quant_end_time = time.time()
        quant_elapsed_time = quant_end_time - quant_start_time
        
        print(f"✓ Layer {layer_num} quantized successfully in {quant_elapsed_time:.2f} seconds")
        
        result = {
            "layer": layer_num,
            "status": "success",
            "model_load_time": model_load_time,
            "quantization_time": quant_elapsed_time,
            "total_time": model_load_time + quant_elapsed_time,
            "targets": targets,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        quant_end_time = time.time()
        quant_elapsed_time = quant_end_time - quant_start_time
        
        print(f"✗ Layer {layer_num} failed: {str(e)}")
        
        result = {
            "layer": layer_num,
            "status": "failed",
            "model_load_time": model_load_time,
            "quantization_time": quant_elapsed_time,
            "total_time": model_load_time + quant_elapsed_time,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
    
    # Clean up model to free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return result

def main():
    print(f"\n{'#'*60}")
    print(f"Layer-by-Layer FP8 Quantization Test (Fixed)")
    print(f"Model: {MODEL_ID}")
    print(f"{'#'*60}")
    
    # Create output directory
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # First, load model once to detect number of layers
    print("\nDetecting model structure...")
    detect_start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype="auto", 
        device_map="auto"
    )
    
    # Check actual number of layers
    actual_layers = []
    for i in range(100):  # Check up to 100 layers
        try:
            _ = model.get_submodule(f"model.layers.{i}")
            actual_layers.append(i)
        except AttributeError:
            break
    
    num_layers = len(actual_layers)
    detect_time = time.time() - detect_start
    print(f"Found {num_layers} layers in the model (detection took {detect_time:.2f}s)")
    
    # Clean up detection model
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    # Timing results storage
    timing_results = {
        "model": MODEL_ID,
        "total_layers": num_layers,
        "model_detection_time": detect_time,
        "start_time": datetime.now().isoformat(),
        "layers": []
    }
    
    # Quantize each layer individually with fresh model
    total_quantization_start = time.time()
    
    for layer_num in range(num_layers):
        layer_result = quantize_single_layer(MODEL_ID, layer_num, OUTPUT_BASE_DIR)
        timing_results["layers"].append(layer_result)
        
        # Save intermediate results after each layer
        with open(f"{OUTPUT_BASE_DIR}/{TIMING_LOG_FILE}", "w") as f:
            json.dump(timing_results, f, indent=2)
        
        print(f"  Memory cleanup complete. Progress: {layer_num + 1}/{num_layers}")
    
    # Calculate summary statistics
    total_time = time.time() - total_quantization_start
    successful_layers = [l for l in timing_results["layers"] if l["status"] == "success"]
    failed_layers = [l for l in timing_results["layers"] if l["status"] == "failed"]
    
    if successful_layers:
        timing_results["summary"] = {
            "total_time": total_time,
            "successful_layers": len(successful_layers),
            "failed_layers": len(failed_layers),
            "avg_model_load_time": sum(l["model_load_time"] for l in successful_layers) / len(successful_layers),
            "avg_quantization_time": sum(l["quantization_time"] for l in successful_layers) / len(successful_layers),
            "avg_total_time_per_layer": sum(l["total_time"] for l in successful_layers) / len(successful_layers),
            "min_quantization_time": min(l["quantization_time"] for l in successful_layers),
            "max_quantization_time": max(l["quantization_time"] for l in successful_layers),
            "end_time": datetime.now().isoformat()
        }
    else:
        timing_results["summary"] = {
            "total_time": total_time,
            "successful_layers": 0,
            "failed_layers": len(failed_layers),
            "end_time": datetime.now().isoformat()
        }
    
    # Save final results
    with open(f"{OUTPUT_BASE_DIR}/{TIMING_LOG_FILE}", "w") as f:
        json.dump(timing_results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"QUANTIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total layers processed: {num_layers}")
    print(f"Successful: {len(successful_layers)}")
    print(f"Failed: {len(failed_layers)}")
    print(f"Total time: {total_time:.2f} seconds")
    
    if successful_layers:
        print(f"\nTiming Statistics:")
        print(f"  Average model load time: {timing_results['summary']['avg_model_load_time']:.2f}s")
        print(f"  Average quantization time: {timing_results['summary']['avg_quantization_time']:.2f}s")
        print(f"  Average total per layer: {timing_results['summary']['avg_total_time_per_layer']:.2f}s")
        print(f"  Min/Max quantization: {timing_results['summary']['min_quantization_time']:.2f}s / {timing_results['summary']['max_quantization_time']:.2f}s")
    
    print(f"\nDetailed timing log saved to: {OUTPUT_BASE_DIR}/{TIMING_LOG_FILE}")
    
    # Print per-layer summary
    print(f"\n{'='*60}")
    print(f"PER-LAYER TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Layer':<8} {'Status':<10} {'Load(s)':<10} {'Quant(s)':<10} {'Total(s)':<10}")
    print(f"{'-'*48}")
    for layer in timing_results["layers"]:
        status_symbol = "✓" if layer["status"] == "success" else "✗"
        load_time = layer.get("model_load_time", 0)
        quant_time = layer.get("quantization_time", 0)
        total_time = layer.get("total_time", 0)
        print(f"{layer['layer']:<8} {status_symbol} {layer['status']:<8} {load_time:<10.2f} {quant_time:<10.2f} {total_time:<10.2f}")

if __name__ == "__main__":
    main()