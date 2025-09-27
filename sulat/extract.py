__author__ = "SURUS AI"
__copyright__ = "LLC"
__credits__ = ["SURUS AI"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "SURUS AI"
__email__ = "contact@surus.ai"
__status__ = "Development"

import requests
import os
import json
import glob
import random
import re
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from .config import get_cache_dir
from .text_utils import _to_int, _norm_text, _norm, _to_bool, _parse_number, _should_keep_punct_for_field
from .scoring import _score_date, _score_email, _score_url, _score_phone, _score_id, _flatten_to_kv, _score_dict, _score_field_recursive, _detect_field_type, _parse_date, _normalize_url_fallback
from .data_utils import create_dynamic_signature, load_hf_dataset_and_infer_schema, load_and_infer_schema, build_trainset_from_examples, _get_nested_value, _infer_schema_recursively, infer_metric_schema
from .metrics import make_universal_metric
from .optimization_utils import _save_optimized_program, _safe_mipro, analyze_optimization_report

from dspy.teleprompt import MIPROv2
import logging
import inspect


class MissingAPIKeyError(EnvironmentError):
    """Raised when SURUS_API_KEY is not set."""
    pass

class SurusAPIError(Exception):
    """HTTP error from SURUS API."""
    def __init__(self, status_code: int, details):
        super().__init__(f"SURUS API error {status_code}")
        self.status_code = status_code
        self.details = details


def autotune(data_source: Union[str, Path], save_optimized_name: str, 
             input_key: str = "input_text", 
             output_key: str = "expected_output",
             max_examples: int = 128,
             seed: int = 42,
             algorithm: str = "miprov2",
             num_trials: int = 5,
             max_bootstrapped_demos: int = 8,
             max_labeled_demos: int = 8,
             optimize_metric: bool = False,
             metric_iterations: int = 1,
             optimize_extractor_schedule: str = "first",
             report_path: Optional[str] = None,
             model: str = "litellm_proxy/google/gemma-3n-E4B-it",
             design_with_llm: bool = False,
             design_model: Optional[str] = None) -> Dict[str, Any]:
    """
    Auto-tune the extraction process using DSPy optimization techniques.
    
    Args:
        data_source: Either a directory path with JSON files or a HuggingFace dataset ID
        save_optimized_name: Name to save the optimized configuration in the cache
        input_key: Key for input text in the dataset (default: "input_text")
        output_key: Key for expected output in the dataset (default: "expected_output")
        max_examples: Maximum number of examples to use for optimization
        seed: Random seed for reproducibility
        algorithm: Optimization algorithm to use ('miprov2' or 'gepa')
        num_trials: Number of optimization trials
        max_bootstrapped_demos: Maximum number of bootstrapped demonstrations
        max_labeled_demos: Maximum number of labeled demonstrations
        optimize_metric: Whether to optimize the metric itself
        metric_iterations: Number of iterations to improve the metric
        optimize_extractor_schedule: When to optimize the extractor ('first', 'last', 'each', 'never')
        report_path: Path to write final JSON report
        model: The model name to use for DSPy optimization (default: 'litellm_proxy/google/gemma-3n-E4B-it')
        design_with_llm: Whether to use LLM for designing the metric plan (default: False, uses heuristic)
        design_model: Optional model name for metric-plan design when LLM mode is enabled.
    
    Returns:
        Dictionary containing optimization results
    """
    import dspy
    
    # Configure DSPy with the specified or environment-provided model (single configuration)
    selected_model = model or os.getenv("MODEL_NAME", "litellm_proxy/google/gemma-3n-E4B-it")
    dspy.configure(lm=dspy.LM(
        selected_model,
        api_base=os.getenv("SURUS_API_BASE", "https://api.surus.dev/functions/v1"),
        api_key=os.getenv("SURUS_API_KEY")
    ))
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    
    # Determine if data_source is a directory or a HuggingFace dataset
    is_hf_dataset = False
    if Path(data_source).is_dir():
        is_hf_dataset = False
    else:
        # Check if it could be a valid HF dataset identifier
        # Simple check: if it contains a slash or looks like a valid identifier
        is_hf_dataset = "/" in str(data_source) or str(data_source).count("-") > 0
    
    if is_hf_dataset:
        # Split dataset_id and split if provided (e.g., "dataset_name:split")
        if ":" in str(data_source):
            dataset_id, split = str(data_source).split(":", 1)
        else:
            dataset_id, split = str(data_source), "train"

        logger.info(f"Loading HuggingFace dataset: {dataset_id}[{split}]")
        raw_examples, output_schema = load_hf_dataset_and_infer_schema(
            dataset_id, split, input_key, output_key
        )
        files = None  # Signal directory mode not used
    else:
        logger.info(f"Loading data from directory: {data_source}")
        files, output_schema = load_and_infer_schema(str(data_source), input_key, output_key)
        raw_examples = None

    # Create dynamic signature based on the inferred schema
    DynamicSignature = create_dynamic_signature(input_key, output_schema)
    
    # Build trainset based on the data source
    if raw_examples is not None:
        trainset = build_trainset_from_examples(
            raw_examples, input_key, output_schema, output_key, max_examples, seed
        )
    else:
        # Build trainset from files
        trainset = []
        if files:
            for fp in files:
                try:
                    with open(fp, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    examples = data if isinstance(data, list) else [data]
                    for ex in examples:
                        if input_key in ex and output_key in ex:
                            gold_data = ex[output_key]
                            example_kwargs = {input_key: ex[input_key], **{field: _get_nested_value(gold_data, field) for field in output_schema.keys()}}
                            dspy_ex = dspy.Example(**example_kwargs).with_inputs(input_key)
                            trainset.append(dspy_ex)
                except Exception:
                    logger.exception(f"Failed to load {fp} while building trainset")
                    continue
        random.Random(seed).shuffle(trainset)
        if max_examples is not None and len(trainset) > max_examples:
            trainset = trainset[:max_examples]

    # Split into train/val to avoid evaluation leakage
    total_n = len(trainset)
    if total_n >= 2:
        # At least 20% or 16 examples (whichever larger), capped at half for stability
        val_size = max(1, min(max(int(0.2 * total_n), 16), total_n // 2))
    else:
        val_size = 0
    valset = trainset[:val_size] if val_size > 0 else []
    trainset_core = trainset[val_size:] if val_size > 0 else trainset

    # Create universal metric for evaluation
    logger.info("Creating universal metric...")
    
    # Get inferred schema and weights from train split only
    schema = infer_metric_schema(trainset_core, input_key)
    weights = schema.get("weights", None) if schema else None
    
    selected_metric = make_universal_metric(
        trainset_core,
        input_key,
        list(output_schema.keys()),
        seed=seed,
        weights=weights,
        design_with_llm=design_with_llm,
        design_model=design_model,
    )
    
    def make_gepa_wrapper(metric_fn, label="universal"):
        def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
            try:
                score = float(metric_fn(gold, pred))
            except Exception:
                score = 0.0
            return dspy.Prediction(score=score, feedback=f"{label} metric score={score:.3f}")
        return gepa_metric

    selected_gepa_metric = make_gepa_wrapper(selected_metric)

    # Create metric scoring classes if they don't exist yet
    class _MetricScorerSignature(dspy.Signature):
        gold = dspy.InputField(desc="The ground truth data, serving as the correct answer.")
        pred = dspy.InputField(desc="The extracted data from the model, which needs to be evaluated.")
        score = dspy.OutputField(desc="A float score between 0.0 and 1.0 representing the quality of the prediction.")
        feedback = dspy.OutputField(desc="Constructive feedback explaining the score.")

    class _SimpleMetricScorerSignature(dspy.Signature):
        gold = dspy.InputField(desc="The ground truth data, serving as the correct answer.")
        pred = dspy.InputField(desc="The extracted data from the model, which needs to be evaluated.")
        score = dspy.OutputField(desc="A float score between 0.0 and 1.0 representing the quality of the prediction.")

    class MetricProgram(dspy.Module):
        def __init__(self, metric_fn, output_fields, use_feedback=False):
            super().__init__()
            self.metric_fn = metric_fn
            self.use_feedback = use_feedback
            self.output_fields = output_fields
            self.scorer = dspy.Predict(_MetricScorerSignature if use_feedback else _SimpleMetricScorerSignature)

        def _serialize(self, ex):
            if not ex: return "None"
            fields = [f"- {k}: {getattr(ex, k, None)}" for k in self.output_fields if getattr(ex, k, None) is not None]
            return "\n".join(fields) or "No relevant fields found."

        def forward(self, gold, pred, trace=None, **kwargs):
            py_score = self.metric_fn(gold, pred, trace=trace, **kwargs)
            py_score = getattr(py_score, 'score', py_score)
            lm_result = self.scorer(gold=f"Gold standard:\n{self._serialize(gold)}", pred=f"Prediction:\n{self._serialize(pred)}")
            lm_score = py_score
            try:
                if score_match := re.search(r"([0-9.]+)", str(lm_result.score)):
                    lm_score = float(score_match.group(1))
            except (ValueError, TypeError, AttributeError): pass
            if self.use_feedback:
                feedback = getattr(lm_result, 'feedback', f"Python score: {py_score:.3f}, LM score: {lm_score:.3f}")
                return dspy.Prediction(score=lm_score, feedback=feedback)
            return lm_score

    # Compile the optimized program
    logger.info(f"Optimizing program with algorithm: {algorithm.upper()}")
    
    # Initialize optimization reports
    optimization_reports = []
    
    # Iterate for metric optimization if enabled
    for iteration in range(metric_iterations):
        logger.info(f"Starting iteration {iteration + 1}/{metric_iterations}")

        iteration_report = {
            "iteration": iteration + 1,
            "metric_optimization": {},
            "extractor_optimization": {}
        }

        if optimize_metric:
            logger.info("Metric optimization enabled.")
            metric_program = MetricProgram(selected_gepa_metric if algorithm == 'gepa' else selected_metric, 
                                          list(output_schema.keys()), use_feedback=algorithm == 'gepa')

            # Build metric trainset using a subset of the original trainset
            sample_size = min(32, len(trainset_core))
            metric_trainset_examples = trainset_core[:sample_size]
            
            # Create metric optimization examples
            metric_trainset = []
            base_extractor = dspy.Predict(DynamicSignature)
            for ex in metric_trainset_examples:
                try:
                    pred = base_extractor(**{input_key: getattr(ex, input_key)})
                    true_score = selected_metric(ex, pred)
                    metric_ex = dspy.Example(
                        gold=ex, 
                        pred=pred, 
                        true_score=true_score
                    ).with_inputs('gold', 'pred')
                    metric_trainset.append(metric_ex)
                except Exception as e:
                    logger.debug(f"Error creating metric training example: {e}")
                    continue

            if metric_trainset:
                def meta_metric(ex, pred, trace=None):
                    try:
                        pred_score = float(getattr(pred, 'score', pred))
                        true_score = float(ex.true_score)
                        return abs(pred_score - true_score) < 0.1
                    except Exception:
                        return 0.0

                logger.info(f"Optimizing metric with MIPROv2 on {len(metric_trainset)} examples.")
                metric_optimizer = _safe_mipro(
                    metric=meta_metric, 
                    max_bootstrapped_demos=4, 
                    max_labeled_demos=4, 
                    num_trials=10
                )
                try:
                    optimized_metric_program = metric_optimizer.compile(
                        metric_program, 
                        trainset=metric_trainset
                    )
                    # Use the optimized metric
                    if hasattr(optimized_metric_program, 'metric_fn'):
                        selected_metric = getattr(optimized_metric_program, 'forward', optimized_metric_program)
                        selected_gepa_metric = selected_metric
                    else:
                        selected_metric = optimized_metric_program.forward if hasattr(optimized_metric_program, 'forward') else optimized_metric_program
                    logger.info("Metric optimization complete.")
                    
                    iteration_report["metric_optimization"] = {
                        "status": "completed",
                        "trainset_size": len(metric_trainset),
                    }
                except Exception as e:
                    logger.exception(f"Metric optimization failed: {e}")
                    iteration_report["metric_optimization"] = {
                        "status": "failed",
                        "error": str(e)
                    }
            else:
                logger.warning("Skipping metric optimization: could not build trainset.")
                iteration_report["metric_optimization"] = {
                    "status": "skipped",
                    "reason": "could not build trainset"
                }

        # Decide whether to optimize extractor this iteration
        should_optimize_extractor = (
            (optimize_extractor_schedule == "each") or 
            (optimize_extractor_schedule == "first" and iteration == 0) or 
            (optimize_extractor_schedule == "last" and iteration == metric_iterations - 1)
        )

        if should_optimize_extractor:
            logger.info(f"Extractor optimization enabled with {algorithm.upper()} (schedule={optimize_extractor_schedule}).")
            
            if trainset_core:
                if algorithm == 'gepa':
                    try:
                        from dspy import GEPA
                        optimizer = GEPA(
                            metric=selected_gepa_metric, 
                            track_stats=True, 
                            auto='heavy'
                        )
                        extractor = dspy.Predict(DynamicSignature)
                        effective_valset = valset if valset else trainset_core[:min(len(trainset_core), 16)]
                        optimized_extractor = optimizer.compile(
                            extractor, 
                            trainset=trainset_core, 
                            valset=effective_valset
                        )
                        iteration_report["extractor_optimization"]["status"] = "completed"
                        iteration_report["extractor_optimization"]["trainset_size"] = len(trainset_core)
                        iteration_report["extractor_optimization"]["valset_size"] = len(effective_valset)
                    except ImportError:
                        logger.warning("GEPA not available, falling back to MIPROv2")
                        optimizer = _safe_mipro(
                            metric=selected_metric,
                            max_bootstrapped_demos=max_bootstrapped_demos,
                            max_labeled_demos=max_labeled_demos,
                            num_trials=num_trials
                        )
                        extractor = dspy.Predict(DynamicSignature)
                        optimized_extractor = optimizer.compile(
                            extractor,
                            trainset=trainset_core
                        )
                        iteration_report["extractor_optimization"]["status"] = "completed (fallback MIPROv2)"
                        iteration_report["extractor_optimization"]["trainset_size"] = len(trainset_core)
                else:  # MIPROv2
                    optimizer = _safe_mipro(
                        metric=selected_metric,
                        max_bootstrapped_demos=max_bootstrapped_demos,
                        max_labeled_demos=max_labeled_demos,
                        num_trials=num_trials
                    )
                    extractor = dspy.Predict(DynamicSignature)
                    optimized_extractor = optimizer.compile(
                        extractor,
                        trainset=trainset_core
                    )
                    iteration_report["extractor_optimization"]["status"] = "completed"
                    iteration_report["extractor_optimization"]["trainset_size"] = len(trainset_core)
                
                logger.info(f"Extractor optimization complete for iteration {iteration + 1}.")
            else:
                logger.warning("Skipping extractor optimization: train split is empty.")
                iteration_report["extractor_optimization"] = {
                    "status": "skipped",
                    "reason": "empty train split"
                }
        else:
            # If we're not optimizing this iteration but have an extractor (from previous iteration or initial)
            if 'optimized_extractor' not in locals():
                extractor = dspy.Predict(DynamicSignature)
                optimized_extractor = extractor  # Use non-optimized version

        # Add iteration report to collection
        optimization_reports.append(iteration_report)

        # After each iteration (except the last), analyze the optimization report
        if iteration < metric_iterations - 1 and optimize_metric:
            logger.info("Analyzing optimization results for improvements...")
            
            # Get current metric configuration (simplified)
            current_metric_config = {
                "type": algorithm.upper(),
                "fields": list(output_schema.keys()),
                "optimization_status": iteration_report
            }
            
            # Analyze the report and get suggestions
            analysis = analyze_optimization_report(iteration_report, current_metric_config)
            logger.info(f"Optimization analysis for next iteration: {analysis}")

    # Perform final evaluation on validation split (avoid train leakage)
    logger.info("Running final evaluation on validation split...")
    all_examples = valset if valset else trainset_core
    report_records, final_scores = [], []
    for ex in all_examples:
        try:
            pred = optimized_extractor(**{input_key: getattr(ex, input_key)})
            score = selected_metric(ex, pred)
            score_value = getattr(score, 'score', score)
            final_scores.append(score_value)
            report_records.append({
                "input": getattr(ex, input_key), 
                "prediction": {field: getattr(pred, field, None) for field in output_schema.keys()}, 
                "gold": {field: getattr(ex, field, None) for field in output_schema.keys()}, 
                "score": score_value
            })
        except Exception:
            logger.exception("Error processing example for final report.")
            report_records.append({
                "input": getattr(ex, input_key), 
                "error": "processing_failed"
            })

    avg_score = sum(final_scores) / len(final_scores) if final_scores else 0.0
    logger.info(f"Final average score across {len(all_examples)} examples: {avg_score:.3f}")
    
    stats = {
        "total_examples": len(all_examples), 
        "average_score": avg_score, 
        "errors": len([r for r in report_records if "error" in r]), 
        "iterations": metric_iterations,
        "optimization_reports": optimization_reports,
        "train_examples": len(trainset_core),
        "val_examples": len(valset)
    }

    # Write report if path provided
    if report_path:
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({"stats": stats, "results": report_records}, f, ensure_ascii=False, indent=2)
            logger.info(f"Wrote final report to {report_path}")
        except Exception:
            logger.exception(f"Failed to write report to {report_path}")

    # Prepare results dictionary for saving
    results_dict = {
        "status": "success",
        "optimized_name": save_optimized_name,
        "training_examples": len(trainset_core),
        "output_fields": list(output_schema.keys()),
        "data_source": data_source,
        "input_key": input_key,
        "output_key": output_key,
        "final_average_score": avg_score,
        "stats": stats
    }

    # Save the optimized program to cache
    _save_optimized_program(optimized_extractor, save_optimized_name, results=results_dict)

    logger.info(f"Optimization complete. Saved as '{save_optimized_name}' in cache.")

    return results_dict


def extract(
    text: str,
    json_schema: dict = None,
    model: Optional[str] = "litellm_proxy/google/gemma-3n-E4B-it",
    temperature: float = 0.0,
    timeout: float = 30.0,
    load_optimized_name: Optional[str] = None,
) -> dict:
    """
    Extract structured information from text using SURUS API or optimized models.

    Args:
        text: The input text to process.
        json_schema: The JSON schema to guide the extraction (when not using optimized models).
        model: The name of the model to use for extraction.
        temperature: The temperature for the model.
        load_optimized_name: Optional name of an optimized configuration to load from cache.

    Returns:
        A dictionary containing the extracted information.
    """
    if load_optimized_name is not None:
        # Load optimized program from cache
        cache_dir = Path(get_cache_dir())
        program_dir = cache_dir / load_optimized_name
        
        if not program_dir.exists():
            raise FileNotFoundError(f"Optimized program '{load_optimized_name}' not found in cache. Run autotune first.")
        
        import dspy
        # Load the metadata to get information about the program
        metadata_path = program_dir / "metadata.json"
        input_key = "input_text"  # default
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            # Get the input key from the saved results if available
            input_key = metadata.get('results', {}).get('input_key', 'input_text')
        
        # Try to load the full DSPy program using dspy.load()
        optimized_program = None
        program_subdir = program_dir / "program"
        
        try:
            # Load the full program (architecture + state) using DSPy's recommended approach
            optimized_program = dspy.load(str(program_subdir))
            print(f"Using optimized configuration: {load_optimized_name}")
        except Exception as e:
            print(f"Error loading program with dspy.load(): {e}")
            # Try fallback methods
            program_pickle_path = program_dir / "program.pkl"
            if program_pickle_path.exists():
                try:
                    import pickle
                    with open(program_pickle_path, 'rb') as f:
                        optimized_program = pickle.load(f)
                    print(f"Using optimized configuration: {load_optimized_name} (loaded with pickle)")
                except Exception as pickle_error:
                    print(f"Also failed with pickle load: {pickle_error}")
                    # Fallback to original API approach
                    pass
        
        if optimized_program is not None:
            dspy.configure(lm=dspy.LM(os.getenv("MODEL_NAME", "litellm_proxy/google/gemma-3n-E4B-it"), api_base=os.getenv("SURUS_API_BASE", "https://api.surus.dev/functions/v1"), api_key=os.getenv("SURUS_API_KEY")))
            # For the optimized program, the loaded object might be a predictor or a complete pipeline
            # We need to call it appropriately based on its type
            try:

                # First, try to call it directly as a predictor/function
                if hasattr(optimized_program, 'forward') or callable(optimized_program):
                    result = optimized_program(**{input_key: text})
                else:
                    # If it's not callable or doesn't have forward, we can't use it
                    raise AttributeError("Loaded program doesn't have callable interface")
                
                # Extract the output fields from the result (which should be a DSPy Prediction object)
                if hasattr(result, '_store'):
                    # DSPy prediction with _store attribute
                    output_dict = {k: v for k, v in result._store.items() if not k.startswith('_') and k != input_key}
                elif hasattr(result, 'items') or hasattr(result, '__getitem__'):
                    # If result is dict-like
                    output_dict = {k: v for k, v in result.items() if not k.startswith('_') and k != input_key}
                else:
                    # Try to extract attributes from the result object
                    output_dict = {}
                    for attr_name in dir(result):
                        if not attr_name.startswith('_') and attr_name != input_key and not callable(getattr(result, attr_name)):
                            output_dict[attr_name] = getattr(result, attr_name)
                
                return output_dict
            except Exception as e:
                print(f"Error running optimized program: {e}")
                # Fallback to original API approach
                pass
        else:
            print(f"Could not load optimized program '{load_optimized_name}'")
            # Fallback to original API approach
    
    # Use the original API approach when no optimization is specified or as fallback
    api_key = os.getenv("SURUS_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("SURUS_API_KEY environment variable not set")

    api_url = "https://api.surus.dev/functions/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # If using an optimized model, we might want to construct the system message differently
    schema_desc = json_schema if json_schema is not None else {}
    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant designed to output JSON conforming to this schema: {schema_desc}",
        },
        {"role": "user", "content": text},
    ]

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
    }

    response = requests.post(api_url, headers=headers, json=data, timeout=timeout)

    try:
        response.raise_for_status()
    except requests.HTTPError as err:
        try:
            error_json = response.json()
        except ValueError:
            error_json = {"error": response.text}
        raise SurusAPIError(response.status_code, error_json) from err

    response_json = response.json()
    message_content = response_json["choices"][0]["message"]["content"]
    return json.loads(message_content)