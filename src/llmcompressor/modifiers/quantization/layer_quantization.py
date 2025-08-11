"""
Layer-level quantization modifier with built-in timing for fine-grained performance measurement.
"""

import time
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path

import torch
import tqdm
from loguru import logger

from llmcompressor.core import Event, EventType, State
from llmcompressor.modifiers import Modifier
from llmcompressor.modifiers.quantization.calibration import (
    update_weight_global_scale,
    update_weight_zp_scale,
)
from llmcompressor.modifiers.quantization.quantization.mixin import QuantizationMixin
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales

__all__ = ["LayerQuantizationModifier", "LayerQuantizationTimer"]


class LayerQuantizationTimer:
    """Utility class for tracking layer-level quantization timing."""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.timing_data = {
            "start_time": datetime.now().isoformat(),
            "layers": [],
            "summary": {}
        }
        self.output_dir = output_dir or "quantization_timing"
        self.current_layer_start = None
        
    def start_layer(self, layer_num: int, targets: List[str]):
        """Start timing for a specific layer."""
        self.current_layer_start = time.time()
        logger.info(f"Starting quantization for layer {layer_num} with targets: {targets}")
        
    def end_layer(self, layer_num: int, status: str = "success", error: Optional[str] = None):
        """End timing for a specific layer and record results."""
        if self.current_layer_start is None:
            logger.warning(f"end_layer called for layer {layer_num} without start_layer")
            return
            
        elapsed_time = time.time() - self.current_layer_start
        
        layer_data = {
            "layer": layer_num,
            "status": status,
            "quantization_time": elapsed_time,
            "timestamp": datetime.now().isoformat()
        }
        
        if error:
            layer_data["error"] = error
            
        self.timing_data["layers"].append(layer_data)
        self.current_layer_start = None
        
        logger.info(f"Layer {layer_num} quantization {'completed' if status == 'success' else 'failed'} in {elapsed_time:.2f}s")
        
    def save_timing_report(self, file_name: str = "layer_quantization_timing.json"):
        """Save timing report to JSON file."""
        self.timing_data["end_time"] = datetime.now().isoformat()
        
        # Calculate summary statistics
        successful_layers = [l for l in self.timing_data["layers"] if l["status"] == "success"]
        
        if successful_layers:
            quantization_times = [l["quantization_time"] for l in successful_layers]
            self.timing_data["summary"] = {
                "total_layers": len(self.timing_data["layers"]),
                "successful_layers": len(successful_layers),
                "failed_layers": len(self.timing_data["layers"]) - len(successful_layers),
                "total_quantization_time": sum(quantization_times),
                "avg_quantization_time": sum(quantization_times) / len(quantization_times),
                "min_quantization_time": min(quantization_times),
                "max_quantization_time": max(quantization_times)
            }
        
        # Save to file
        output_path = Path(self.output_dir) / file_name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(self.timing_data, f, indent=2)
            
        logger.info(f"Timing report saved to {output_path}")
        return output_path


class LayerQuantizationModifier(Modifier, QuantizationMixin):
    """
    Quantization modifier that processes layers individually with detailed timing.
    
    This modifier extends the base QuantizationModifier to provide:
    - Layer-by-layer quantization with fine-grained control
    - Built-in timing measurements for each layer
    - Detailed logging and reporting of quantization performance
    
    :param layer_targets: Dictionary mapping layer numbers to their target modules
    :param measure_timing: Whether to measure and log timing for each layer
    :param timing_output_dir: Directory to save timing reports
    :param config_groups: Quantization schemes to apply (inherited from QuantizationMixin)
    :param targets: Default targets if not specified per layer
    :param ignore: Modules to ignore during quantization
    :param scheme: Quantization scheme to apply
    """
    
    def __init__(
        self,
        layer_targets: Optional[Dict[int, List[str]]] = None,
        measure_timing: bool = True,
        timing_output_dir: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._layer_targets = layer_targets or {}
        self._measure_timing = measure_timing
        self._timing_output_dir = timing_output_dir
        self._timer = LayerQuantizationTimer(timing_output_dir) if measure_timing else None
        self._processed_layers = set()
        
    def on_initialize(self, state: State, **kwargs) -> bool:
        """Initialize quantization with layer-specific configuration."""
        if not QuantizationMixin.has_config(self):
            raise ValueError(
                "LayerQuantizationModifier requires quantization fields to be specified"
            )
        
        # Initialize base quantization
        QuantizationMixin.initialize_quantization(self, state.model)
        
        # Log layer targets if specified
        if self._layer_targets:
            logger.info(f"Initialized layer-specific quantization for {len(self._layer_targets)} layers")
            
        return True
    
    def quantize_layer(
        self, 
        model: torch.nn.Module, 
        layer_num: int, 
        targets: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Quantize a specific layer with timing measurement.
        
        :param model: The model containing the layer
        :param layer_num: The layer number to quantize
        :param targets: Optional list of target modules for this layer
        :return: Dictionary containing timing and status information
        """
        # Import required functions for calibration status management
        from llmcompressor.modifiers.quantization.calibration import (
            apply_calibration_status,
            freeze_module_quantization,
        )
        from compressed_tensors.quantization import enable_quantization
        
        # Use provided targets or get from layer_targets or use defaults
        if targets is None:
            targets = self._layer_targets.get(layer_num, self._get_default_layer_targets(layer_num))
        
        # Start timing if enabled
        if self._timer:
            self._timer.start_layer(layer_num, targets)
        
        result = {
            "layer": layer_num,
            "targets": targets,
            "start_time": datetime.now().isoformat()
        }
        
        try:
            # Get the actual layer modules
            layer_modules = []
            for target in targets:
                try:
                    module = model.get_submodule(target)
                    layer_modules.append((target, module))
                except AttributeError:
                    logger.warning(f"Target module {target} not found in model")
            
            if not layer_modules:
                raise ValueError(f"No valid target modules found for layer {layer_num}")
            
            # Apply quantization to each module in the layer
            quantization_start = time.time()
            
            # Ensure modules are in calibration mode and quantization is enabled
            for target_name, module in layer_modules:
                apply_calibration_status(module)
                enable_quantization(module)
            
            for target_name, module in layer_modules:
                # Update weight scales for the module
                update_weight_global_scale(module)
                update_fused_layer_weight_global_scales(module)
                update_weight_zp_scale(module)
            
            # Freeze the quantization after calibration
            for target_name, module in layer_modules:
                freeze_module_quantization(module)
                enable_quantization(module)  # Keep quantization enabled after freezing
            
            quantization_time = time.time() - quantization_start
            
            # Mark layer as processed
            self._processed_layers.add(layer_num)
            
            # Record success
            result.update({
                "status": "success",
                "quantization_time": quantization_time,
                "modules_quantized": len(layer_modules),
                "end_time": datetime.now().isoformat()
            })
            
            if self._timer:
                self._timer.end_layer(layer_num, status="success")
            
            logger.info(f"Successfully quantized layer {layer_num} ({len(layer_modules)} modules) in {quantization_time:.2f}s")
            
        except Exception as e:
            # Record failure
            error_msg = str(e)
            result.update({
                "status": "failed",
                "error": error_msg,
                "end_time": datetime.now().isoformat()
            })
            
            if self._timer:
                self._timer.end_layer(layer_num, status="failed", error=error_msg)
            
            logger.error(f"Failed to quantize layer {layer_num}: {error_msg}")
            
        return result
    
    def quantize_all_layers(self, model: torch.nn.Module, num_layers: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Quantize all layers in the model sequentially with timing.
        
        :param model: The model to quantize
        :param num_layers: Number of layers to quantize (auto-detect if None)
        :return: List of results for each layer
        """
        # Auto-detect number of layers if not provided
        if num_layers is None:
            num_layers = self._detect_num_layers(model)
            
        logger.info(f"Starting layer-by-layer quantization for {num_layers} layers")
        
        results = []
        for layer_num in tqdm.tqdm(range(num_layers), desc="Quantizing layers"):
            result = self.quantize_layer(model, layer_num)
            results.append(result)
        
        # Save timing report if enabled
        if self._timer:
            report_path = self._timer.save_timing_report()
            logger.info(f"Quantization timing report saved to {report_path}")
        
        return results
    
    def on_start(self, state: State, event: Event, **kwargs):
        """Begin layer-by-layer calibration and quantization."""
        self.started_ = True
        
        # Start calibration
        QuantizationMixin.start_calibration(self, state.model)
        
        # If layer_targets are specified, quantize only those layers
        if self._layer_targets:
            for layer_num, targets in self._layer_targets.items():
                self.quantize_layer(state.model, layer_num, targets)
        else:
            # Otherwise, quantize all layers
            self.quantize_all_layers(state.model)
    
    def on_end(self, state: State, event: Event, **kwargs):
        """Finish calibration and save timing reports."""
        self.ended_ = True
        
        # End calibration
        QuantizationMixin.end_calibration(self, state.model)
        
        # Generate final timing report
        if self._timer:
            self._timer.save_timing_report("final_layer_quantization_timing.json")
        
        logger.info(f"Layer quantization completed. Processed {len(self._processed_layers)} layers")
    
    def _get_default_layer_targets(self, layer_num: int) -> List[str]:
        """Get default target modules for a transformer layer."""
        return [
            f"model.layers.{layer_num}.self_attn.q_proj",
            f"model.layers.{layer_num}.self_attn.k_proj",
            f"model.layers.{layer_num}.self_attn.v_proj",
            f"model.layers.{layer_num}.self_attn.o_proj",
            f"model.layers.{layer_num}.mlp.gate_proj",
            f"model.layers.{layer_num}.mlp.up_proj",
            f"model.layers.{layer_num}.mlp.down_proj",
        ]
    
    def _detect_num_layers(self, model: torch.nn.Module) -> int:
        """Auto-detect the number of layers in the model."""
        num_layers = 0
        for i in range(200):  # Check up to 200 layers
            try:
                _ = model.get_submodule(f"model.layers.{i}")
                num_layers = i + 1
            except AttributeError:
                break
        
        logger.info(f"Detected {num_layers} layers in the model")
        return num_layers


def quantize_single_layer(
    model: torch.nn.Module,
    layer_num: int,
    scheme: str = "FP8_DYNAMIC",
    targets: Optional[List[str]] = None,
    measure_timing: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to quantize a single layer with timing.
    
    :param model: The model containing the layer
    :param layer_num: The layer number to quantize
    :param scheme: Quantization scheme to use
    :param targets: Optional list of target modules
    :param measure_timing: Whether to measure timing
    :param output_dir: Directory to save timing reports
    :return: Dictionary with quantization results and timing
    """
    modifier = LayerQuantizationModifier(
        scheme=scheme,
        measure_timing=measure_timing,
        timing_output_dir=output_dir,
        ignore=["lm_head"]
    )
    
    # Initialize the modifier
    state = State(model=model)
    modifier.on_initialize(state)
    
    # Quantize the specific layer
    result = modifier.quantize_layer(model, layer_num, targets)
    
    # Save timing if enabled
    if modifier._timer:
        modifier._timer.save_timing_report(f"layer_{layer_num}_timing.json")
    
    return result