"""Optimization utilities for the sulat package."""

import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, Union
import dspy
import logging
import inspect
from .config import get_cache_dir
from .data_utils import load_hf_dataset_and_infer_schema, load_and_infer_schema, create_dynamic_signature, build_trainset_from_examples
from .metrics import make_universal_metric


def _save_optimized_program(program, save_name: str, results: Dict[str, Any] = None):
    """Save the optimized program to the cache."""
    import dspy
    cache_dir = Path(get_cache_dir())
    program_dir = cache_dir / save_name
    program_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the full DSPy program (architecture + state) using the recommended approach
    try:
        # Use save_program=True to save both architecture and state
        program_save_dir = program_dir / "program"
        program.save(str(program_save_dir), save_program=True)
        print(f"Saved optimized program to {program_save_dir}")
    except AttributeError:
        # If save method doesn't exist, try traditional pickle for the program
        import pickle
        program_pickle_path = program_dir / "program.pkl"
        with open(program_pickle_path, 'wb') as f:
            pickle.dump(program, f)
        print(f"Saved optimized program to {program_pickle_path} using pickle")
    
    # Save results and metadata as JSON
    metadata = {
        "type": "optimized_extractor",
        "timestamp": __import__('time').time(),
        "save_name": save_name,
        "results": results or {}
    }
    
    metadata_path = program_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Saved metadata to {metadata_path}")


def _safe_mipro(metric, **kwargs):
    """Safely create a MIPROv2 optimizer with only supported arguments."""
    from dspy.teleprompt import MIPROv2
    sig = inspect.signature(MIPROv2.__init__)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if ignored := [k for k in kwargs if k not in accepted]:
        logger = logging.getLogger(__name__)
        logger.debug(f"MIPROv2 ignored unsupported args: {ignored}")
    return MIPROv2(metric=metric, **accepted)


def analyze_optimization_report(report_data, current_metric_plan):
    """Use an LLM to analyze optimization results and suggest improvements."""
    import dspy
    logger = logging.getLogger(__name__)
    
    try:
        # Create the analyzer signature
        class _OptimizationReportAnalysisSignature(dspy.Signature):
            optimization_report = dspy.InputField(desc="Report from the previous optimization iteration, including statistics and performance metrics.")
            current_metric_plan = dspy.InputField(desc="Current metric configuration in JSON format.")
            suggestions = dspy.OutputField(desc="Suggestions for improving the metric in the next iteration, focusing on what went right and wrong.")

        # Create the analyzer
        analyzer = dspy.Predict(_OptimizationReportAnalysisSignature)
        
        # Convert report data to string format
        report_str = json.dumps(report_data, indent=2) if isinstance(report_data, dict) else str(report_data)
        metric_plan_str = json.dumps(current_metric_plan, indent=2) if isinstance(current_metric_plan, dict) else str(current_metric_plan)
        
        # Get suggestions from LLM
        result = analyzer(
            optimization_report=report_str,
            current_metric_plan=metric_plan_str
        )
        
        return getattr(result, 'suggestions', "No specific suggestions provided.")
    except Exception as e:
        logger.exception("Failed to analyze optimization report")
        return f"Error analyzing report: {str(e)}"