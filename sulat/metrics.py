"""Metric utilities for the sulat package.

Note:
- By default, the metric plan is heuristic (no LLM call). This is intentional to avoid requiring API keys.
- You can opt in to LLM-designed plans via make_universal_metric(..., design_with_llm=True, ...).
"""

import random
import re
import json
import ast
from typing import Optional, List, Dict, Any
import dspy
import logging
import os
from .text_utils import _norm, _should_keep_punct_for_field, _to_bool, _parse_number
from .scoring import _score_date, _score_email, _score_url, _score_phone, _score_id, _score_dict, _score_field_recursive, _detect_field_type


def _ex_attr(ex, key, default=None):
    if hasattr(ex, '_store'):
        return ex._store.get(key, default)
    return getattr(ex, key, default)


def _serialize_examples(ts, ik, ofs, max_k=5):
    rnd = random.Random(42)  # Using fixed seed for consistency
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
        m = re.search(r"\{.*\}", s, flags=re.DOTALL)
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


def _tokens(s: str, tiny_stop=None, text_cfg=None):
    if tiny_stop is None:
        tiny_stop = {"the", "a", "an", "of", "and", "to", "in", "on", "for", "is", "are", "was", "were"}
    if text_cfg is None:
        text_cfg = {"token_min_len": 2}
    
    from .text_utils import _norm
    s = _norm(s)
    toks = [t for t in s.split() if len(t) >= max(1, text_cfg.get("token_min_len", 2)) and t not in tiny_stop]
    return toks


def _tokens_with_field(s: str, field_name: str = None, tiny_stop=None, text_cfg=None):
    """Tokenize text with field-specific normalization."""
    if tiny_stop is None:
        tiny_stop = {"the", "a", "an", "of", "and", "to", "in", "on", "for", "is", "are", "was", "were"}
    if text_cfg is None:
        text_cfg = {"token_min_len": 2}
    
    # Import needed functions locally to avoid closure issues
    from .text_utils import _norm_text, _should_keep_punct_for_field, _norm
    
    # Determine normalization function based on field name
    if field_name and _should_keep_punct_for_field(field_name):
        norm_func = lambda x: _norm_text(x, keep_punct=True)
    else:
        norm_func = _norm
    
    s = norm_func(s)
    toks = [t for t in s.split() if len(t) >= max(1, text_cfg.get("token_min_len", 2)) and t not in tiny_stop]
    return toks


def _score_text(g, p, field_name: str = None):
    # This is the full implementation of the nested function in make_universal_metric
    # Normalize types to strings
    g = "" if g is None else str(g)
    p = "" if p is None else str(p)

    # Import needed functions locally to avoid closure issues
    from .text_utils import _norm_text, _should_keep_punct_for_field, _norm
    
    # Determine normalization function based on field name
    if field_name and _should_keep_punct_for_field(field_name):
        norm_func = lambda x: _norm_text(x, keep_punct=True)
    else:
        norm_func = _norm

    g_norm = norm_func(g)
    p_norm = norm_func(p)

    # Both blank
    if g_norm == "" and p_norm == "":
        return 1.0
    # Gold blank
    if g_norm == "":
        # This would need hallucination_penalty to be passed in, defaulting to True here
        hallucination_penalty = True  # Defaulting to True as per original
        return 0.0 if (hallucination_penalty and p_norm != "") else 1.0
    # Pred blank
    if p_norm == "":
        return 0.0

    # Very short gold: require substring match
    if len(g_norm.split()) <= 3 or len(g_norm) <= 10:  # Using defaults from make_universal_metric
        return 1.0 if (g_norm in p_norm or p_norm in g_norm) else 0.0

    # Default to Jaccard over tokens
    # For this implementation, we'll use hardcoded defaults
    default_text_cfg = {"method": "jaccard", "short_tokens": 3, "minlen_substring": 10, "token_min_len": 2}
    
    method = default_text_cfg.get("method", "jaccard")
    if method == "contains":
        return 1.0 if g_norm in p_norm else 0.0

    # Default to Jaccard over tokens
    tiny_stop = {"the", "a", "an", "of", "and", "to", "in", "on", "for", "is", "are", "was", "were"}
    text_cfg = default_text_cfg
    
    gt = set(_tokens_with_field(g, field_name, tiny_stop, text_cfg))
    pt = set(_tokens_with_field(p, field_name, tiny_stop, text_cfg))
    if not gt and not pt:
        return 1.0
    if not gt or not pt:
        return 0.0
    inter = len(gt & pt)
    union = len(gt | pt)
    return inter / union if union else 1.0


def _score_list_f1(g_list, p_list):
    # Import functions locally to avoid closure issues
    from .text_utils import _norm
    from .metrics import _is_blank_value
    
    # Compare sets of normalized string items using F1 score
    g_norm = { _norm(x) for x in (g_list or []) if not _is_blank_value(x) }
    p_norm = { _norm(x) for x in (p_list or []) if not _is_blank_value(x) }
    if not g_norm and not p_norm:
        return 1.0
    if not g_norm:
        # This would need hallucination_penalty to be passed in, defaulting to True here
        hallucination_penalty = True
        return 0.0 if (hallucination_penalty and p_norm) else 1.0
    if not p_norm:
        return 0.0
    
    intersection = len(g_norm & p_norm)
    if intersection == 0:
        return 0.0
    
    precision = intersection / len(p_norm) if p_norm else 0.0
    recall = intersection / len(g_norm) if g_norm else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def _score_list(g_list, p_list, method="f1"):
    # Import functions locally to avoid closure issues
    from .text_utils import _norm
    from .metrics import _is_blank_value, _score_list_f1
    
    # Compare sets of normalized string items
    g_norm = { _norm(x) for x in (g_list or []) if not _is_blank_value(x) }
    p_norm = { _norm(x) for x in (p_list or []) if not _is_blank_value(x) }
    if not g_norm and not p_norm:
        return 1.0
    if not g_norm:
        # This would need hallucination_penalty to be passed in, defaulting to True here
        hallucination_penalty = True
        return 0.0 if (hallucination_penalty and p_norm) else 1.0
    if not p_norm:
        return 0.0
        
    if method == "jaccard":
        inter = len(g_norm & p_norm)
        union = len(g_norm | p_norm)
        return inter / union if union else 1.0
    else:  # Default to F1
        return _score_list_f1(g_list, p_list)


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


def _infer_numeric_subtype(field_name):
    """
    Infer the subtype of a numeric field based on its name.
    
    Args:
        field_name: The name of the field
        
    Returns:
        String representing the inferred numeric subtype
    """
    field_lower = field_name.lower()
    if 'percent' in field_lower or field_lower.endswith('%') or 'pct' in field_lower:
        return "numeric_percent"
    elif 'currency' in field_lower or 'price' in field_lower or 'cost' in field_lower or 'amount' in field_lower:
        return "numeric_currency"
    elif 'count' in field_lower or 'number' in field_lower or 'num' in field_lower or field_lower.endswith('_id') or field_lower == 'id':
        return "numeric_count"
    else:
        return "numeric_general"


def _score_num(field, g, p):
    # Import functions locally to avoid closure issues
    from .text_utils import _to_bool, _parse_number
    from .metrics import _infer_numeric_subtype
    
    # Booleans: treat as exact match
    gb, pb = _to_bool(g), _to_bool(p)
    if gb is not None or pb is not None:
        # Fix: return 0.0 if either is None, else compare equality
        if gb is None or pb is None:
            return 0.0
        return 1.0 if gb == pb else 0.0

    g, p = _parse_number(g), _parse_number(p)
    if g is None and p is None:
        return 1.0
    if g is None:
        # This would need hallucination_penalty to be passed in, defaulting to True here
        hallucination_penalty = True
        return 0.0 if hallucination_penalty and (p is not None) else 1.0
    if p is None:
        return 0.0

    # Using default numeric config values from make_universal_metric
    numeric_defaults = {
        "abs_close": 2.0,
        "abs_ok": 6.0,
        "relative_after": 12.0
    }
    cfg = numeric_defaults.copy()
    
    # Adjust scoring based on field-specific requirements for percentages
    field_subtype = _infer_numeric_subtype(field)
    if field_subtype == "numeric_percent" and str(g).endswith('%'):
        # If g is a percentage, we may want to handle it differently
        # For now, we use the parsed value but future updates can handle this differently
        pass
    
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
    # Import functions locally to avoid closure issues
    from .scoring import _score_date, _score_email, _score_url, _score_phone, _score_id, _score_dict
    from .text_utils import _to_bool
    from .metrics import _is_blank_value, _score_num, _score_list, _score_text
    
    # Check for specific field types before default logic
    field_type = ftype
    if field_type == "date":
        return _score_date(g_val, p_val)
    elif field_type == "email":
        return _score_email(g_val, p_val)
    elif field_type == "url":
        return _score_url(g_val, p_val)
    elif field_type == "phone":
        return _score_phone(g_val, p_val)
    elif field_type == "id":
        return _score_id(g_val, p_val)
    elif field_type == "dict":
        return _score_dict(g_val, p_val, field_name=field)
    elif field_type == "numeric":
        return _score_num(field, g_val, p_val)
    
    # If both lists -> list scoring
    if isinstance(g_val, list) or isinstance(p_val, list):
        g_list = g_val if isinstance(g_val, list) else ([] if _is_blank_value(g_val) else [g_val])
        p_list = p_val if isinstance(p_val, list) else ([] if _is_blank_value(p_val) else [p_val])
        # Use F1 as default, but allow per-field configuration via text config
        # For this basic implementation, defaulting to F1
        list_method = "f1"  # Default to F1
        return _score_list(g_list, p_list, method=list_method)

    # If dicts, use specialized dict scoring instead of flattening
    if isinstance(g_val, dict) or isinstance(p_val, dict):
        return _score_dict(g_val, p_val, field_name=field)

    # Try boolean-as-text exactness first
    gb, pb = _to_bool(g_val), _to_bool(p_val)
    if gb is not None or pb is not None:
        # Fix: return 0.0 if either is None, else compare equality
        if gb is None or pb is None:
            return 0.0
        return 1.0 if gb == pb else 0.0

    # Default text scoring
    return _score_text(g_val, p_val, field)


def make_universal_metric(
    trainset,
    input_key: str,
    output_fields: List[str],
    seed: int = 42,
    k: int = 5,
    weights: Optional[Dict[str, float]] = None,
    design_with_llm: bool = False,
    design_lm=None,
    design_model: Optional[str] = "litellm_proxy/google/gemma-3n-E4B-it",
):
    """
    Build a universal metric.
    - Default: heuristic plan (no LLM). Clear logs indicate this.
    - Opt-in: LLM-designed plan when design_with_llm=True. You may pass an LM via `design_lm` or a model name via `design_model`.
      If neither is provided, tries the current dspy context LM; falls back to heuristic if none is available.

    Returns: callable metric(gold, pred, trace=None) -> float.
    """
    import dspy
    import ast
    rnd = random.Random(seed)
    logger = logging.getLogger(__name__)

    # Prepare prompt to design the plan.
    examples_str = _serialize_examples(trainset, input_key, output_fields, max_k=k)
    instructions = (
        "Design a single lead metric plan (strict JSON). "
        "Goals: fairly compare predictions to gold across fields, scale scores 0.0–1.0. "
        "Output ONLY JSON. Required keys: "
        "- weights: {field: float in [0.0, 2.0]} "
        "- field_types: {field: 'numeric'|'text'} "
        "- numeric: {field: {abs_close:int|float, abs_ok:int|float, relative_after:int|float}} "
        "- text: {method:'jaccard'|'contains', short_tokens:int, minlen_substring:int, token_min_len:int, list_method:'f1'|'jaccard'} "
        "- hallucination_penalty: true|false"
    )
    
    # Create metric design signature
    class _MetricDesignSignature(dspy.Signature):
        examples = dspy.InputField(desc="Up to 5 labeled examples including input and gold outputs.")
        field_names = dspy.InputField(desc="Comma-separated list of output field names.")
        metric_instructions = dspy.InputField(desc="Design a lead metric plan as strict JSON with weights and per-field rules.")
        plan = dspy.OutputField(desc="Strict JSON defining the lead metric. No extra text.")

    # Build a robust default
    default_text_cfg = {"method": "jaccard", "short_tokens": 3, "minlen_substring": 10, "token_min_len": 2, "list_method": "f1"}
    default_numeric_cfg = {"default": {"abs_close": 2, "abs_ok": 6, "relative_after": 12}}
    try:
        from data_utils import infer_metric_schema
        schema = infer_metric_schema(trainset, input_key)
        if schema and isinstance(schema.get("numeric"), dict):
            default_numeric_cfg.update({k: v for k, v in schema["numeric"].items() if k != "default"})
    except Exception:
        schema = None

    # Try to get LLM-designed plan using _MetricDesignSignature
    plan_json = None
    try:
        # A minimal LM that returns a heuristic plan; used for heuristic mode and as fallback.
        class DummyLM(dspy.LM):
            def __init__(self):
                super().__init__(model='dummy')
            def __call__(self, prompt=None, **kwargs):
                # Accept both completion- and chat-style arguments
                _ = prompt or kwargs.get("messages")
                heuristic_plan = self._create_heuristic_plan()
                # JSONAdapter expects a JSON with the output fields; here: {"plan": "..."}
                response_text = json.dumps({"plan": json.dumps(heuristic_plan)})
                return [{"text": response_text}]
            def _create_heuristic_plan(self):
                try:
                    weights_dict = weights or {f: 1.0 for f in output_fields}
                    numeric_cfg = {k: v for k, v in default_numeric_cfg.items() if k != "default"}
                    inferred_numeric_fields = set(numeric_cfg.keys())
                    field_types = {f: ("numeric" if f in inferred_numeric_fields else "text") for f in output_fields}
                    return {
                        "weights": {f: float(weights_dict.get(f, 1.0)) for f in output_fields},
                        "field_types": field_types,
                        "numeric": numeric_cfg,
                        "text": default_text_cfg.copy(),
                        "hallucination_penalty": True,
                    }
                except Exception:
                    return {
                        "weights": {f: 1.0 for f in output_fields},
                        "field_types": {f: "text" for f in output_fields},
                        "numeric": {},
                        "text": default_text_cfg.copy(),
                        "hallucination_penalty": True,
                    }

        def _design_plan_with_lm():
            nonlocal plan_json
            metric_designer = dspy.Predict(_MetricDesignSignature)
            result = metric_designer(
                examples=examples_str,
                field_names=", ".join(output_fields),
                metric_instructions=instructions
            )
            plan_str = result.plan
            parsed = _safe_json_extract(plan_str)
            if parsed and isinstance(parsed, dict):
                plan_json = parsed

        plan_json = None
        if design_with_llm:
            logger.info("Designing metric plan via LLM (opt-in enabled).")
            lm_to_use = design_lm
            if lm_to_use is None and design_model:
                lm_to_use = dspy.LM(
                    design_model,
                    api_base=os.getenv("SURUS_API_BASE", "https://api.surus.dev/functions/v1"),
                    api_key=os.getenv("SURUS_API_KEY"),
                )
            # Use current dspy settings LM if still None
            if lm_to_use is None:
                try:
                    lm_to_use = dspy.settings.lm  # type: ignore
                except Exception:
                    lm_to_use = None
            if lm_to_use is not None:
                try:
                    with dspy.context(lm=lm_to_use):
                        _design_plan_with_lm()
                except Exception:
                    logger.exception("LLM metric-plan design failed; falling back to heuristic.")
                    plan_json = None
            else:
                logger.warning("No LM available for metric-plan design; falling back to heuristic plan.")
        else:
            logger.info("Using heuristic metric plan (no LLM).")

        # If not using LLM or if LLM attempt produced no plan, use Dummy heuristic.
        if not plan_json:
            dummy_lm = DummyLM()
            with dspy.context(lm=dummy_lm):
                _design_plan_with_lm()
            if not plan_json:
                # Final safety fallback to the dummy heuristic builder
                plan_json = DummyLM()._create_heuristic_plan()
    except Exception:
        # If anything fails, fall back to heuristic
        logger = logging.getLogger(__name__)
        logger.exception("Metric-plan creation encountered an error; using heuristic plan.")
        plan_json = None
    # If no plan from LLM or parsing failed, use heuristic approach
    if not plan_json:
         try:
             weights_dict = weights or {f: 1.0 for f in output_fields}
             numeric_cfg = {k: v for k, v in default_numeric_cfg.items() if k != "default"}
             inferred_numeric_fields = set(numeric_cfg.keys())
             field_types = {f: ("numeric" if f in inferred_numeric_fields else "text") for f in output_fields}
             plan_json = {
                 "weights": {f: float(weights_dict.get(f, 1.0)) for f in output_fields},
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
    else:
        # Clarify in logs whether plan was LLM-designed or heuristic
        logger = logging.getLogger(__name__)
        logger.info("Metric plan ready (%s).", "LLM-designed" if design_with_llm else "heuristic")

    # Validate and complete the plan
    allowed_methods = {"jaccard", "contains"}
    # Weights
    raw_weights = plan_json.get("weights", {}) or {}
    final_weights = {}
    for f in output_fields:
        # Use the provided weights parameter as primary source, then fall back to plan_json, then default to 1.0
        provided_weight = weights.get(f, None) if weights else None
        if provided_weight is not None:
            # Use provided weight if available
            w = provided_weight
        else:
            # Otherwise, use weight from the plan
            w = raw_weights.get(f, 1.0)
        
        try:
            w = float(w)
        except Exception:
            w = 1.0
        # Clamp to [0, 2]
        final_weights[f] = max(0.0, min(2.0, w))
        
    # Use the final weights instead of the original weights variable
    weights = final_weights

    # Field types
    raw_field_types = plan_json.get("field_types", {}) or {}
    # If schema suggests numeric fields, prefer that hint
    inferred_numeric_fields = set()
    if schema and isinstance(schema.get("numeric"), dict):
        inferred_numeric_fields = {k for k in schema["numeric"].keys() if k != "default"}

    field_types = {}
    for f in output_fields:
        ft = raw_field_types.get(f, None)
        if ft not in {"numeric", "text", "date", "email", "url", "phone", "id", "dict"}:
            # Try to detect field type based on field name
            detected_type = _detect_field_type(f)
            ft = detected_type
        field_types[f] = ft

    # Text config
    text_cfg = default_text_cfg.copy()
    try:
        text_cfg.update({k: v for k, v in (plan_json.get("text") or {}).items() if k in {"method", "short_tokens", "minlen_substring", "token_min_len", "list_method"}})
    except Exception:
        pass
    if text_cfg.get("method") not in allowed_methods:
        text_cfg["method"] = "jaccard"
    # Set default list_method to "f1" if not specified
    if "list_method" not in text_cfg:
        text_cfg["list_method"] = "f1"
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
    # Helper to determine field-specific normalization
    def _get_norm_function(field_name):
        if _should_keep_punct_for_field(field_name):
            return lambda x: _norm_text(x, keep_punct=True)
        return _norm_text  # Default to normal _norm_text with punctuation removal

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