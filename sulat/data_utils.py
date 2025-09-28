"""Data utilities for the sulat package."""

import json
import glob
import random
import re
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from .text_utils import _norm, _to_int

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
    """Dynamically creates a dspy.Signature class from a schema, sanitizing field names."""
    import dspy
    import re

    path_to_attr = {}
    fields = {input_key: dspy.InputField(desc="Input text for extraction.")}
    docstring = "Extract the following fields from the input text:\n"
    for field_name, description in output_schema.items():
        sanitized_name = re.sub(r'[^a-zA-Z0-9_]', '_', field_name)
        if sanitized_name and sanitized_name[0].isdigit():
            sanitized_name = f"_{sanitized_name}"
        
        # handle collisions
        original_sanitized_name = sanitized_name
        i = 1
        while sanitized_name in fields:
            sanitized_name = f"{original_sanitized_name}_{i}"
            i += 1

        path_to_attr[field_name] = sanitized_name
        fields[sanitized_name] = dspy.OutputField(desc=description)
        docstring += f"- {sanitized_name} (from {field_name}): {description}\n"

    DynamicSignature = type("DynamicSignature", (dspy.Signature,), fields)
    DynamicSignature.__doc__ = docstring
    return DynamicSignature, path_to_attr


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


def build_trainset_from_examples(examples: List[Dict], input_key: str, path_to_attr: Dict[str, str], output_key: str, max_examples: Optional[int], seed: int):
    """Build trainset from in-memory examples (list of dicts)."""
    import dspy
    trainset = []
    for ex in examples:
        if input_key in ex and output_key in ex and isinstance(ex[output_key], dict):
            gold_data = ex[output_key]
            example_kwargs = {input_key: ex[input_key]}
            for field, attr in path_to_attr.items():
                example_kwargs[attr] = _get_nested_value(gold_data, field)
            dspy_ex = dspy.Example(**example_kwargs).with_inputs(input_key)
            trainset.append(dspy_ex)
    random.Random(seed).shuffle(trainset)
    if max_examples is not None and len(trainset) > max_examples:
        trainset = trainset[:max_examples]
    return trainset


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