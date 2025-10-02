__author__ = "SURUS AI"
__copyright__ = "LLC"
__credits__ = ["SURUS AI"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "SURUS AI"
__email__ = "contact@surus.dev"
__status__ = "Development"

import requests
import os
import json
from pathlib import Path
from typing import Optional
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
        cache_dir = Path(get_cache_dir()).resolve()
        # Sanitize load_optimized_name to avoid path traversal or absolute paths
        safe_name = os.path.basename(load_optimized_name)
        program_dir = (cache_dir / safe_name).resolve()

        # Ensure the resolved program_dir is a descendant of cache_dir
        try:
            program_dir.relative_to(cache_dir)
        except Exception:
            raise FileNotFoundError(f"Optimized program '{load_optimized_name}' resolves outside the cache directory")

        if not program_dir.exists():
            raise FileNotFoundError(f"Optimized program '{safe_name}' not found in cache. Run autotune first.")
        
        import dspy
        # Load the metadata to get information about the program
        metadata_path = program_dir / "metadata.json"
        input_key = "input_text"  # default
        path_to_attr = None
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            results = metadata.get('results', {})
            # Get the input key from the saved results if available
            input_key = results.get('input_key', 'input_text')
            path_to_attr = results.get('path_to_attr')
        
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
                # Prefer using dspy.load() even for the pickle path if dspy supports it
                try:
                    optimized_program = dspy.load(str(program_pickle_path))
                    print(f"Using optimized configuration: {safe_name} (loaded with dspy.load from pickle path)")
                except Exception as dspy_pickle_load_error:
                    # Only allow raw pickle deserialization if explicitly opted-in via env var
                    allow_pickle = os.getenv("SULAT_ALLOW_PICKLE_CACHE", "false").lower() in ("1", "true", "yes")
                    if allow_pickle:
                        try:
                            import pickle
                            with open(program_pickle_path, 'rb') as f:
                                optimized_program = pickle.load(f)
                            print(f"Using optimized configuration: {safe_name} (loaded with pickle - opt-in)")
                        except Exception as pickle_error:
                            print(f"Also failed with pickle load: {pickle_error}")
                            # Fallback to original API approach
                            pass
                    else:
                        print("Skipping unsafe pickle.load for cached program (SULAT_ALLOW_PICKLE_CACHE not set).")
        
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
                raw_output_dict = {}
                if hasattr(result, '_store'):
                    # DSPy prediction with _store attribute
                    raw_output_dict = {k: v for k, v in result._store.items() if not k.startswith('_') and k != input_key}
                elif hasattr(result, 'items') and callable(getattr(result, 'items', None)):
                    # If result implements .items(), treat it as dict-like
                    try:
                        raw_output_dict = {k: v for k, v in result.items() if not k.startswith('_') and k != input_key}
                    except Exception:
                        # Fall back to attribute introspection if items() fails unexpectedly
                        raw_output_dict = {}
                        for attr_name in dir(result):
                            if not attr_name.startswith('_') and attr_name != input_key and not callable(getattr(result, attr_name)):
                                raw_output_dict[attr_name] = getattr(result, attr_name)
                else:
                    # Try to extract attributes from the result object (handles sequences and other objects)
                    for attr_name in dir(result):
                        if not attr_name.startswith('_') and attr_name != input_key and not callable(getattr(result, attr_name)):
                            raw_output_dict[attr_name] = getattr(result, attr_name)

                if 'path_to_attr' in locals() and path_to_attr:
                    output_dict = {}
                    attr_to_path = {v: k for k, v in path_to_attr.items()}
                    for attr, value in raw_output_dict.items():
                        original_path = attr_to_path.get(attr)
                        if original_path:
                            keys = original_path.split('.')
                            d = output_dict
                            for i, key in enumerate(keys):
                                if i == len(keys) - 1:
                                    d[key] = value
                                else:
                                    d = d.setdefault(key, {})
                        else:
                            output_dict[attr] = value # Keep fields not in map
                    return output_dict
                
                return raw_output_dict
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

    api_url = os.getenv("SURUS_API_BASE", "https://api.surus.dev/functions/v1") + "/chat/completions"
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