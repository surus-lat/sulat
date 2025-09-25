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


class MissingAPIKeyError(EnvironmentError):
    """Raised when SURUS_API_KEY is not set."""
    pass

class SurusAPIError(Exception):
    """HTTP error from SURUS API."""
    def __init__(self, status_code: int, details):
        super().__init__(f"SURUS API error {status_code}")
        self.status_code = status_code
        self.details = details


# --- Utility Functions ---

def _to_int(text_or_num):
    if text_or_num is None:
        return None
    if isinstance(text_or_num, (int, float)):
        try:
            return int(text_or_num)
        except (ValueError, TypeError):
            return None
    if isinstance(text_or_num, str):
        m = re.search(r"-?\\d+", text_or_num)
        if m:
            try:
                return int(m.group(0))
            except (ValueError, TypeError):
                return None
    return None


def _norm(s):
    if s is None:
        return ""
    s = re.sub(r"[^\\w\\s]", "", str(s))
    return " ".join(s.strip().lower().split())


def _get_nested_value(data: Dict[str, Any], key: str) -> Any:
    """Access a nested value in a dictionary using dot notation."""
    keys = key.split('.')
    value = data
    for k in keys:
        if isinstance(value, dict):
            value = value.get(k)
        else:
            return None
    return value


def _infer_schema_recursively(data: dict, parent_key: str = "") -> dict:
    schema = {}
    for key, value in data.items():
        full_key = f"{parent_key}.{key}" if parent_key else key
        if isinstance(value, dict) and value:
            schema.update(_infer_schema_recursively(value, full_key))
        else:
            desc_key = key.replace("_", " ")
            if parent_key:
                desc_parent = parent_key.replace(".", " ").replace("_", " ")
                description = f"The {desc_key} of the {desc_parent}."
            else:
                description = f"The extracted {desc_key}."
            schema[full_key] = description
    return schema


def create_dynamic_signature(input_key: str, output_schema: Dict[str, str]):
    """Dynamically creates a dspy.Signature class from a schema."""
    import dspy
    fields = {input_key: dspy.InputField(desc="Input text for extraction.")}
    docstring = "Extract the following fields from the input text:\n"
    for field_name, description in output_schema.items():
        fields[field_name] = dspy.OutputField(desc=description)
        docstring += f"- {field_name}: {description}\n"

    DynamicSignature = type("DynamicSignature", (dspy.Signature,), fields)
    DynamicSignature.__doc__ = docstring
    return DynamicSignature


def load_hf_dataset_and_infer_schema(dataset_id: str, split: str, input_key: str, output_key: str, max_rows: Optional[int] = None):
    """Load a HuggingFace dataset and infer output schema (output_key column must be a dict or JSON object)."""
    import dspy
    from datasets import load_dataset
    ds = load_dataset(dataset_id, split=split)
    examples = []
    for i, row in enumerate(ds):
        if max_rows and i >= max_rows:
            break
        if input_key not in row or output_key not in row:
            continue
        out_val = row[output_key]
        # If output column is a JSON string, try parse
        if isinstance(out_val, str):
            out_val_str = out_val.strip()
            if out_val_str.startswith("{") and out_val_str.endswith("}"):
                try:
                    out_val = json.loads(out_val_str)
                except Exception:
                    pass
        if not isinstance(out_val, dict):
            # Skip rows where we can't coerce dict
            continue
        examples.append({input_key: row[input_key], output_key: out_val})
    if not examples:
        raise ValueError(f"No usable rows in dataset '{dataset_id}' split '{split}' with columns '{input_key}' and '{output_key}'.")
    # Infer schema from first example's output dict keys
    first = examples[0][output_key]
    output_schema = _infer_schema_recursively(first)
    print(f"HF dataset {dataset_id}[{split}]: collected {len(examples)} examples. Fields: {list(output_schema.keys())}")
    return examples, output_schema


def load_and_infer_schema(data_dir: str, input_key: str, output_key: str):
    """Finds data files and infers the output schema from the first valid example."""
    import dspy
    patterns = [os.path.join(data_dir, "*.json"), os.path.join(data_dir, "**", "*.json")]
    files = sorted({fp for pat in patterns for fp in glob.glob(pat, recursive=True)})
    if not files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            examples = data if isinstance(data, list) else [data]
            for ex in examples:
                if input_key in ex and output_key in ex and isinstance(ex[output_key], dict):
                    output_schema = _infer_schema_recursively(ex[output_key])
                    print(f"Inferred output schema with fields: {list(output_schema.keys())}")
                    return files, output_schema
        except Exception:
            print(f"Skipping file {fp} during schema inference")
            continue

    raise ValueError(f"Could not infer schema. No valid examples found in {data_dir} with keys '{input_key}' and '{output_key}'.")


def build_trainset_from_examples(examples: List[Dict], input_key: str, output_schema: Dict[str, str], output_key: str, max_examples: Optional[int], seed: int):
    """Build trainset from in-memory examples (list of dicts)."""
    import dspy
    output_fields = list(output_schema.keys())
    trainset = []
    for ex in examples:
        if input_key in ex and output_key in ex and isinstance(ex[output_key], dict):
            gold_data = ex[output_key]
            example_kwargs = {input_key: ex[input_key], **{field: _get_nested_value(gold_data, field) for field in output_fields}}
            dspy_ex = dspy.Example(**example_kwargs).with_inputs(input_key)
            trainset.append(dspy_ex)
    random.Random(seed).shuffle(trainset)
    if max_examples is not None and len(trainset) > max_examples:
        trainset = trainset[:max_examples]
    return trainset


def make_universal_metric(trainset, input_key: str, output_fields: List[str], seed: int = 42, k: int = 5):
    """
    Build a metric by prompting the LLM to design a 'lead metric' plan based on up to 5 examples.
    Returns a callable metric(gold, pred, trace=None) -> float.
    """
    import dspy
    import ast
    rnd = random.Random(seed)

    def _ex_attr(ex, key, default=None):
        if hasattr(ex, '_store'):
            return ex._store.get(key, default)
        return getattr(ex, key, default)

    def _serialize_examples(ts, ik, ofs, max_k=5):
        sample = ts[:max_k] if len(ts) <= max_k else rnd.sample(ts, max_k)
        lines = []
        for i, ex in enumerate(sample, 1):
            lines.append(f"Example {i}:")
            lines.append(f"Input: {_ex_attr(ex, ik, '')}")
            lines.append("Gold:")
            for f in ofs:
                val = _ex_attr(ex, f, None)
                lines.append(f"- {f}: {val}")
            lines.append("")  # spacer
        return "\n".join(lines).strip()

    def _strip_code_fences(s: str) -> str:
        s = str(s).strip()
        # Remove common Markdown code fences
        if s.startswith("```"):
            s = re.sub(r"^```(?:json|JSON)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        return s.strip()

    def _safe_json_extract(s):
        # Try hard to parse a JSON-like plan
        s = _strip_code_fences(s)
        # Fast path: direct JSON
        try:
            return json.loads(s)
        except Exception:
            pass
        # Extract first {...} block
        try:
            m = re.search(r"\\{.*\\}", s, flags=re.DOTALL)
            if m:
                block = m.group(0)
                try:
                    return json.loads(block)
                except Exception:
                    # Try Python-literal style (single quotes, trailing commas)
                    return ast.literal_eval(block)
        except Exception:
            pass
        # Try full content as Python literal
        try:
            return ast.literal_eval(s)
        except Exception:
            return None

    def _is_blank_value(v):
        if v is None:
            return True
        if isinstance(v, (list, dict, tuple, set)):
            return len(v) == 0
        if isinstance(v, bool):
            return False
        # strings or others
        return _norm(v) == ""

    def _to_number(x):
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        # Find first number (supports decimals and leading -)
        m = re.search(r"-?\d+(?:\.\d+)?", str(x))
        if not m:
            return None
        try:
            return float(m.group(0))
        except Exception:
            return None

    def _to_bool(x):
        # Accept booleans and common string synonyms
        if isinstance(x, bool):
            return x
        s = _norm(x)
        if s in {"true", "yes", "y", "1"}:
            return True
        if s in {"false", "no", "n", "0"}:
            return False
        return None

    # Prepare prompt to design the plan.
    examples_str = _serialize_examples(trainset, input_key, output_fields, max_k=k)
    instructions = (
        "Design a single lead metric plan (strict JSON). "
        "Goals: fairly compare predictions to gold across fields, scale scores 0.0–1.0. "
        "Output ONLY JSON. Required keys: "
        "- weights: {field: float in [0.0, 2.0]} "
        "- field_types: {field: 'numeric'|'text'} "
        "- numeric: {field: {abs_close:int|float, abs_ok:int|float, relative_after:int|float}} "
        "- text: {method:'jaccard'|'contains', short_tokens:int, minlen_substring:int, token_min_len:int} "
        "- hallucination_penalty: true|false"
    )
    
    # Create metric design signature
    class _MetricDesignSignature(dspy.Signature):
        examples = dspy.InputField(desc="Up to 5 labeled examples including input and gold outputs.")
        field_names = dspy.InputField(desc="Comma-separated list of output field names.")
        metric_instructions = dspy.InputField(desc="Design a lead metric plan as strict JSON with weights and per-field rules.")
        plan = dspy.OutputField(desc="Strict JSON defining the lead metric. No extra text.")

    # Use a dummy LM for metric design to avoid needing API keys during metric creation
    # We'll create a simple metric based on field analysis
    schema = None
    try:
        schema = infer_metric_schema(trainset, input_key)
    except Exception:
        schema = None

    # Build a robust default
    default_text_cfg = {"method": "jaccard", "short_tokens": 3, "minlen_substring": 10, "token_min_len": 2}
    default_numeric_cfg = {"default": {"abs_close": 2, "abs_ok": 6, "relative_after": 12}}
    if schema and isinstance(schema.get("numeric"), dict):
        default_numeric_cfg.update({k: v for k, v in schema["numeric"].items() if k != "default"})

    # If no plan or parsing failed, create one from schema
    # Create a basic metric plan
    try:
        weights = {f: 1.0 for f in output_fields}
        numeric_cfg = {k: v for k, v in default_numeric_cfg.items() if k != "default"}
        inferred_numeric_fields = set(numeric_cfg.keys())
        field_types = {f: ("numeric" if f in inferred_numeric_fields else "text") for f in output_fields}
        plan_json = {
            "weights": {f: float(weights.get(f, 1.0)) for f in output_fields},
            "field_types": field_types,
            "numeric": numeric_cfg,
            "text": default_text_cfg.copy(),
            "hallucination_penalty": True,
        }
    except Exception:
        plan_json = {
            "weights": {f: 1.0 for f in output_fields},
            "field_types": {f: "text" for f in output_fields},
            "numeric": {},
            "text": default_text_cfg.copy(),
            "hallucination_penalty": True,
        }

    # Validate and complete the plan
    allowed_methods = {"jaccard", "contains"}
    # Weights
    raw_weights = plan_json.get("weights", {}) or {}
    weights = {}
    for f in output_fields:
        try:
            w = float(raw_weights.get(f, 1.0))
        except Exception:
            w = 1.0
        # Clamp to [0, 2]
        weights[f] = max(0.0, min(2.0, w))

    # Field types
    raw_field_types = plan_json.get("field_types", {}) or {}
    # If schema suggests numeric fields, prefer that hint
    inferred_numeric_fields = set()
    if schema and isinstance(schema.get("numeric"), dict):
        inferred_numeric_fields = {k for k in schema["numeric"].keys() if k != "default"}

    field_types = {}
    for f in output_fields:
        ft = raw_field_types.get(f, None)
        if ft not in {"numeric", "text"}:
            ft = "numeric" if f in inferred_numeric_fields else "text"
        field_types[f] = ft

    # Text config
    text_cfg = default_text_cfg.copy()
    try:
        text_cfg.update({k: v for k, v in (plan_json.get("text") or {}).items() if k in {"method", "short_tokens", "minlen_substring", "token_min_len"}})
    except Exception:
        pass
    if text_cfg.get("method") not in allowed_methods:
        text_cfg["method"] = "jaccard"
    # Normalize ints
    for k in ("short_tokens", "minlen_substring", "token_min_len"):
        try:
            text_cfg[k] = int(text_cfg.get(k, default_text_cfg[k]))
        except Exception:
            text_cfg[k] = default_text_cfg[k]

    # Numeric config
    numeric_cfg = {}
    try:
        cfg = plan_json.get("numeric") or {}
        if not isinstance(cfg, dict):
            cfg = {}
        # Keep only per-field overrides
        for k, v in cfg.items():
            if not isinstance(v, dict):
                continue
            # Coerce values to floats
            local = {}
            for kk in ("abs_close", "abs_ok", "relative_after"):
                try:
                    local[kk] = float(v.get(kk, default_numeric_cfg["default"][kk]))
                except Exception:
                    local[kk] = float(default_numeric_cfg["default"][kk])
            numeric_cfg[k] = local
    except Exception:
        numeric_cfg = {}
    numeric_defaults = {
        "abs_close": float(default_numeric_cfg["default"]["abs_close"]),
        "abs_ok": float(default_numeric_cfg["default"]["abs_ok"]),
        "relative_after": float(default_numeric_cfg["default"]["relative_after"]),
    }

    hallucination_penalty = bool(plan_json.get("hallucination_penalty", True))

    # Tokenizer for text comparisons
    tiny_stop = {"the", "a", "an", "of", "and", "to", "in", "on", "for", "is", "are", "was", "were"}
    def _tokens(s: str):
        s = _norm(s)
        toks = [t for t in s.split() if len(t) >= max(1, text_cfg.get("token_min_len", 2)) and t not in tiny_stop]
        return toks

    def _score_text(g, p):
        # Normalize types to strings
        g = "" if g is None else str(g)
        p = "" if p is None else str(p)

        g_norm = _norm(g)
        p_norm = _norm(p)

        # Both blank
        if g_norm == "" and p_norm == "":
            return 1.0
        # Gold blank
        if g_norm == "":
            return 0.0 if (hallucination_penalty and p_norm != "") else 1.0
        # Pred blank
        if p_norm == "":
            return 0.0

        # Very short gold: require substring match
        if len(g_norm.split()) <= text_cfg.get("short_tokens", 3) or len(g_norm) <= text_cfg.get("minlen_substring", 10):
            return 1.0 if (g_norm in p_norm or p_norm in g_norm) else 0.0

        method = text_cfg.get("method", "jaccard")
        if method == "contains":
            return 1.0 if g_norm in p_norm else 0.0

        # Default to Jaccard over tokens
        gt = set(_tokens(g))
        pt = set(_tokens(p))
        if not gt and not pt:
            return 1.0
        if not gt or not pt:
            return 0.0
        inter = len(gt & pt)
        union = len(gt | pt)
        return inter / union if union else 1.0

    def _score_list(g_list, p_list):
        # Compare sets of normalized string items
        g_norm = { _norm(x) for x in (g_list or []) if not _is_blank_value(x) }
        p_norm = { _norm(x) for x in (p_list or []) if not _is_blank_value(x) }
        if not g_norm and not p_norm:
            return 1.0
        if not g_norm:
            return 0.0 if (hallucination_penalty and p_norm) else 1.0
        if not p_norm:
            return 0.0
        inter = len(g_norm & p_norm)
        union = len(g_norm | p_norm)
        return inter / union if union else 1.0

    def _flatten_dict(dct):
        # Join stringified leaf values
        try:
            items = []
            for k, v in (dct or {}).items():
                if isinstance(v, (list, tuple, set)):
                    items.extend([str(x) for x in v])
                elif isinstance(v, dict):
                    items.append(json.dumps(v, ensure_ascii=False))
                else:
                    items.append(str(v))
            return " | ".join(items)
        except Exception:
            return str(dct)

    def _score_num(field, g, p):
        # Booleans: treat as exact match
        gb, pb = _to_bool(g), _to_bool(p)
        if gb is not None or pb is not None:
            if gb is None or pb is None:
                # One side booleanifiable, the other not
                return 0.0
            return 1.0 if gb == pb else 0.0

        g, p = _to_number(g), _to_number(p)
        if g is None and p is None:
            return 1.0
        if g is None:
            return 0.0 if hallucination_penalty and (p is not None) else 1.0
        if p is None:
            return 0.0

        cfg = numeric_defaults.copy()
        cfg.update(numeric_cfg.get(field, {}))
        diff = abs(p - g)
        if diff == 0:
            return 1.0
        if diff <= cfg["abs_close"]:
            return 0.8
        if diff <= cfg["abs_ok"]:
            return 0.5
        if abs(g) > cfg["relative_after"]:
            # Linearly decay by relative error
            return max(0.0, 1.0 - (diff / abs(g)))
        return 0.0

    def _score_field(field, g_val, p_val, ftype):
        # Normalize common structures
        if ftype == "numeric":
            return _score_num(field, g_val, p_val)

        # If both lists -> list scoring
        if isinstance(g_val, list) or isinstance(p_val, list):
            g_list = g_val if isinstance(g_val, list) else ([] if _is_blank_value(g_val) else [g_val])
            p_list = p_val if isinstance(p_val, list) else ([] if _is_blank_value(p_val) else [p_val])
            return _score_list(g_list, p_list)

        # If dicts, flatten and treat as text
        if isinstance(g_val, dict) or isinstance(p_val, dict):
            g_txt = _flatten_dict(g_val if isinstance(g_val, dict) else {"value": g_val})
            p_txt = _flatten_dict(p_val if isinstance(p_val, dict) else {"value": p_val})
            return _score_text(g_txt, p_txt)

        # Try boolean-as-text exactness first
        gb, pb = _to_bool(g_val), _to_bool(p_val)
        if gb is not None or pb is not None:
            if gb is None or pb is None:
                return 0.0
            return 1.0 if gb == pb else 0.0

        # Default text scoring
        return _score_text(g_val, p_val)

    def metric(gold, pred, trace=None):
        total_w, acc = 0.0, 0.0

        def _get(obj, key):
            if hasattr(obj, "_store"):
                return obj._store.get(key)
            return getattr(obj, key, None)

        for field in output_fields:
            w = float(weights.get(field, 1.0))
            if w <= 0:
                continue
            g_val, p_val = _get(gold, field), _get(pred, field)
            ftype = field_types.get(field, "text")
            sc = _score_field(field, g_val, p_val, ftype)
            acc += w * sc
            total_w += w
        return (acc / total_w) if total_w > 0 else 0.0

    return metric


def infer_metric_schema(trainset, input_key):
    stats = {}
    for ex in trainset:
        # Fix: Access data from _store for dspy.Example objects
        if hasattr(ex, '_store'):
            attrs = {k: v for k, v in ex._store.items() if not k.startswith("_") and k != input_key}
        else:
            attrs = {k: v for k, v in vars(ex).items() if not k.startswith("_") and k != input_key}
        for k, v in attrs.items():
            s = stats.setdefault(k, {"present": 0, "total": 0, "nums": 0, "values": []})
            s["total"] += 1
            if v is not None and str(v).strip() != "":
                s["present"] += 1
                if (vi := _to_int(v)) is not None:
                    s["nums"] += 1
                    s["values"].append(vi)
    fields = sorted(stats.keys())
    weights = {k: round(0.75 + 0.25 * (s["present"] / s["total"]), 2) for k, s in stats.items() if s["total"]}
    numeric_cfg = {"default": {"abs_close": 2, "abs_ok": 6, "relative_after": 12}}
    for k, s in stats.items():
        if (s["present"] > 0 and (s["nums"] / s["present"]) >= 0.7) and (nums := [n for n in s["values"] if isinstance(n, int)]):
            nums.sort()
            med = nums[len(nums) // 2]
            numeric_cfg[k] = {"abs_close": 1 if med <= 10 else 2, "abs_ok": max(2, (1 if med <= 10 else 2) * 2), "relative_after": max(12, med)}
    return {"fields": fields, "inputs": [input_key], "weights": weights, "numeric": numeric_cfg}


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
             model: str = "google/gemma-3n-E4B-it") -> Dict[str, Any]:
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
        model: The model name to use for DSPy optimization (default: 'google/gemma-3n-E4B-it')
    
    Returns:
        Dictionary containing optimization results
    """
    import dspy
    from dspy.teleprompt import MIPROv2
    import logging
    import inspect
    
    # Configure DSPy with the specified model
    lm = dspy.LM(model)
    #dspy.configure(lm=lm)
    dspy.configure(lm=dspy.LM(os.getenv("MODEL_NAME", "litellm_proxy/google/gemma-3n-E4B-it"), api_base=os.getenv("SURUS_API_BASE", "https://api.surus.dev/functions/v1"), api_key=os.getenv("SURUS_API_KEY")))
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)
    
    def _safe_mipro(metric, **kwargs):
        sig = inspect.signature(MIPROv2.__init__)
        accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
        if ignored := [k for k in kwargs if k not in accepted]:
            logger.debug(f"MIPROv2 ignored unsupported args: {ignored}")
        return MIPROv2(metric=metric, **accepted)

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

    # Create universal metric for evaluation
    logger.info("Creating universal metric...")
    selected_metric = make_universal_metric(trainset, input_key, list(output_schema.keys()), seed=seed)
    
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
            return "\\n".join(fields) or "No relevant fields found."

        def forward(self, gold, pred, trace=None, **kwargs):
            py_score = self.metric_fn(gold, pred, trace=trace, **kwargs)
            py_score = getattr(py_score, 'score', py_score)
            lm_result = self.scorer(gold=f"Gold standard:\\n{self._serialize(gold)}", pred=f"Prediction:\\n{self._serialize(pred)}")
            lm_score = py_score
            try:
                if score_match := re.search(r"([0-9.]+)", str(lm_result.score)):
                    lm_score = float(score_match.group(1))
            except (ValueError, TypeError, AttributeError): pass
            if self.use_feedback:
                feedback = getattr(lm_result, 'feedback', f"Python score: {py_score:.3f}, LM score: {lm_score:.3f}")
                return dspy.Prediction(score=lm_score, feedback=feedback)
            return lm_score

    def analyze_optimization_report(report_data, current_metric_plan):
        """Use an LLM to analyze optimization results and suggest improvements."""
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
            sample_size = min(32, len(trainset))
            metric_trainset_examples = trainset[:sample_size]
            
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
                        selected_metric = optimized_metric_program.metric_fn
                        selected_gepa_metric = optimized_metric_program.metric_fn
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
            
            if trainset:
                if algorithm == 'gepa':
                    try:
                        from dspy import GEPA
                        optimizer = GEPA(
                            metric=selected_gepa_metric, 
                            track_stats=True, 
                            auto='heavy'
                        )
                        extractor = dspy.Predict(DynamicSignature)
                        valset_size = min(len(trainset), 16)
                        optimized_extractor = optimizer.compile(
                            extractor, 
                            trainset=trainset, 
                            valset=trainset[:valset_size]
                        )
                        iteration_report["extractor_optimization"]["status"] = "completed"
                        iteration_report["extractor_optimization"]["trainset_size"] = len(trainset)
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
                            trainset=trainset
                        )
                        iteration_report["extractor_optimization"]["status"] = "completed (fallback MIPROv2)"
                        iteration_report["extractor_optimization"]["trainset_size"] = len(trainset)
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
                        trainset=trainset
                    )
                    iteration_report["extractor_optimization"]["status"] = "completed"
                    iteration_report["extractor_optimization"]["trainset_size"] = len(trainset)
                
                logger.info(f"Extractor optimization complete for iteration {iteration + 1}.")
            else:
                logger.warning("Skipping extractor optimization: could not build trainset.")
                iteration_report["extractor_optimization"] = {
                    "status": "skipped",
                    "reason": "could not build trainset"
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

    # Perform final evaluation
    logger.info("Running final evaluation...")
    all_examples = trainset  # Using the training examples for final evaluation
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
        "optimization_reports": optimization_reports
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
        "training_examples": len(trainset),
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
    model: Optional[str] = "google/gemma-3n-E4B-it",
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
        "model": model if model else "surus-mixtral-8x7b-inst-v0.1",
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