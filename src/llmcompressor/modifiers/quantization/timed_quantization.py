"""
Enhanced quantization modifier with integrated timing at the module level.
This provides the most granular timing information for quantization operations.
"""

import time
import json
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from contextlib import contextmanager

import torch
import tqdm
from loguru import logger

from llmcompressor.modifiers.quantization.quantization import QuantizationModifier
from llmcompressor.modifiers.quantization.calibration import (
    update_weight_global_scale,
    update_weight_zp_scale,
)
from llmcompressor.modifiers.utils import update_fused_layer_weight_global_scales
from llmcompressor.core import State, Event

__all__ = ["TimedQuantizationModifier", "QuantizationTimer"]


class QuantizationTimer:
    """Context manager and utility for fine-grained quantization timing."""
    
    def __init__(self):
        self.timings = {}
        self.current_operation = None
        self.start_time = None
        
    @contextmanager
    def time_operation(self, operation_name: str):
        """Context manager to time a specific operation."""
        self.start_time = time.perf_counter()
        self.current_operation = operation_name
        try:
            yield
        finally:
            elapsed = time.perf_counter() - self.start_time
            if operation_name not in self.timings:
                self.timings[operation_name] = []
            self.timings[operation_name].append(elapsed)
            logger.debug(f"{operation_name} took {elapsed:.4f}s")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get timing summary statistics."""
        summary = {}
        for operation, times in self.timings.items():
            summary[operation] = {
                "count": len(times),
                "total": sum(times),
                "avg": sum(times) / len(times) if times else 0,
                "min": min(times) if times else 0,
                "max": max(times) if times else 0
            }
        return summary
    
    def save_report(self, filepath: Path):
        """Save timing report to JSON file."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "operations": self.timings,
            "summary": self.get_summary()
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Timing report saved to {filepath}")


class TimedQuantizationModifier(QuantizationModifier):
    """
    QuantizationModifier with integrated fine-grained timing measurements.
    
    This modifier extends the base QuantizationModifier to provide detailed
    timing information for each step of the quantization process, including:
    - Weight scale updates
    - Zero-point calculations
    - Module-level quantization
    - Layer-level aggregation
    
    :param enable_timing: Whether to enable timing measurements
    :param timing_output_dir: Directory to save timing reports
    :param log_module_timing: Whether to log timing for individual modules
    :param **kwargs: Additional arguments passed to QuantizationModifier
    """
    
    def __init__(
        self,
        enable_timing: bool = True,
        timing_output_dir: Optional[str] = None,
        log_module_timing: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.enable_timing = enable_timing
        self.timing_output_dir = Path(timing_output_dir or "quantization_timing")
        self.log_module_timing = log_module_timing
        self.timer = QuantizationTimer() if enable_timing else None
        self.module_timings = {} if enable_timing else None
        
    def on_start(self, state: State, event: Event, **kwargs):
        """Begin calibration with timing for each operation."""
        self.started_ = True
        
        # Time the calibration start
        if self.timer:
            with self.timer.time_operation("calibration_start"):
                self.start_calibration(state.model)
        else:
            self.start_calibration(state.model)
        
        # Process modules with timing
        modules = list(state.model.modules())
        
        # Time weight global scale updates
        if self.timer:
            with self.timer.time_operation("weight_global_scale_total"):
                for module in tqdm.tqdm(modules, desc="Updating weight global scales"):
                    if self.log_module_timing:
                        module_name = self._get_module_name(state.model, module)
                        with self.timer.time_operation(f"weight_global_scale_{module_name}"):
                            update_weight_global_scale(module)
                    else:
                        update_weight_global_scale(module)
        else:
            for module in tqdm.tqdm(modules):
                update_weight_global_scale(module)
        
        # Time weight calibration
        if self.timer:
            with self.timer.time_operation("weight_calibration_total"):
                for module in tqdm.tqdm(modules, desc="Calibrating weights"):
                    if self.log_module_timing:
                        module_name = self._get_module_name(state.model, module)
                        with self.timer.time_operation(f"weight_calibration_{module_name}"):
                            self._calibrate_module_weights(module)
                    else:
                        self._calibrate_module_weights(module)
        else:
            for module in tqdm.tqdm(modules, desc="Calibrating weights"):
                self._calibrate_module_weights(module)
        
        # Save intermediate timing report if enabled
        if self.timer:
            self._save_timing_report("calibration_timing.json")
    
    def _calibrate_module_weights(self, module: torch.nn.Module):
        """Calibrate weights for a single module with optional timing."""
        if self.timer and self.log_module_timing:
            module_name = str(type(module).__name__)
            
            with self.timer.time_operation(f"fused_scale_update_{module_name}"):
                update_fused_layer_weight_global_scales(module)
            
            with self.timer.time_operation(f"zp_scale_update_{module_name}"):
                update_weight_zp_scale(module)
        else:
            update_fused_layer_weight_global_scales(module)
            update_weight_zp_scale(module)
    
    def quantize_layer_modules(self, model: torch.nn.Module, layer_num: int) -> Dict[str, float]:
        """
        Quantize all modules in a specific layer with timing.
        
        :param model: The model containing the layer
        :param layer_num: The layer number to quantize
        :return: Dictionary of module names to quantization times
        """
        layer_timings = {}
        
        # Get all modules for this layer
        layer_prefix = f"model.layers.{layer_num}"
        
        for name, module in model.named_modules():
            if name.startswith(layer_prefix):
                if self.timer:
                    start_time = time.perf_counter()
                    
                    # Perform quantization operations
                    update_weight_global_scale(module)
                    update_fused_layer_weight_global_scales(module)
                    update_weight_zp_scale(module)
                    
                    elapsed = time.perf_counter() - start_time
                    layer_timings[name] = elapsed
                    
                    if self.log_module_timing:
                        logger.debug(f"Quantized {name} in {elapsed:.4f}s")
                else:
                    # No timing, just quantize
                    update_weight_global_scale(module)
                    update_fused_layer_weight_global_scales(module)
                    update_weight_zp_scale(module)
        
        return layer_timings
    
    def on_end(self, state: State, event: Event, **kwargs):
        """Finish calibration and save comprehensive timing report."""
        if self.timer:
            with self.timer.time_operation("calibration_end"):
                super().on_end(state, event, **kwargs)
        else:
            super().on_end(state, event, **kwargs)
        
        # Save final timing report
        if self.timer:
            self._save_comprehensive_report()
    
    def _save_timing_report(self, filename: str):
        """Save timing report to file."""
        if not self.timer:
            return
        
        filepath = self.timing_output_dir / filename
        self.timer.save_report(filepath)
    
    def _save_comprehensive_report(self):
        """Save comprehensive timing report with all statistics."""
        if not self.timer:
            return
        
        summary = self.timer.get_summary()
        
        # Calculate total time
        total_time = sum(
            stats["total"] 
            for stats in summary.values()
        )
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_quantization_time": total_time,
            "operations_summary": summary,
            "detailed_timings": self.timer.timings,
            "module_timings": self.module_timings if self.log_module_timing else None
        }
        
        # Save report
        filepath = self.timing_output_dir / "comprehensive_timing_report.json"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive timing report saved to {filepath}")
        
        # Print summary to console
        print("\n" + "="*60)
        print("QUANTIZATION TIMING SUMMARY")
        print("="*60)
        print(f"Total time: {total_time:.2f}s")
        print("\nTop operations by total time:")
        
        sorted_ops = sorted(
            summary.items(), 
            key=lambda x: x[1]["total"], 
            reverse=True
        )[:5]
        
        for op_name, stats in sorted_ops:
            print(f"  {op_name}: {stats['total']:.2f}s (avg: {stats['avg']:.4f}s)")
    
    def _get_module_name(self, model: torch.nn.Module, module: torch.nn.Module) -> str:
        """Get the name of a module within the model."""
        for name, mod in model.named_modules():
            if mod is module:
                return name
        return str(type(module).__name__)