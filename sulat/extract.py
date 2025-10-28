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
        # When using the new task endpoints, optimized programs are no longer needed
        # since the prompting is handled server-side
        print(f"Note: optimized program '{load_optimized_name}' is no longer supported with the new task endpoints")
        print("Using the standard extract API endpoint instead...")
    
    # Use the new API approach with the dedicated extract endpoint
    api_key = os.getenv("SURUS_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("SURUS_API_KEY environment variable not set")

    api_url = os.getenv("SURUS_API_BASE", "https://api.surus.dev/functions/v1") + "/extract"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "text": text,
        "json_schema": json_schema,
        "model": model,
        "temperature": temperature,
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
    return response_json