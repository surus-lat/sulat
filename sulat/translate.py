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
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def translate(text: str,
              target_lang: str,
              model: Optional[str] = None,
              sampling_params: Optional[dict] = None) -> str:
    """
    Translate text using SURUS chat completions API.
    """
    api_key = os.getenv("SURUS_API_KEY")
    if not api_key:
        raise ValueError("SURUS_API_KEY environment variable not set")

    if sampling_params is not None and not isinstance(sampling_params, dict):
        raise TypeError("sampling_params must be a dict when provided")

    api_url = "https://api.surus.dev/functions/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model or "tencent/Hunyuan-MT-7B-fp8",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Translate the following segment into {target_lang}, without additional explanation.\n\n"
                    f"{text}"
                ),
            }
        ],
    }
    if sampling_params:
        payload["sampling_params"] = sampling_params

    response = requests.post(api_url, headers=headers, json=payload)
    try:
        response.raise_for_status()
    except requests.HTTPError as err:
        try:
            error_json = response.json()
        except ValueError:
            error_json = {"error": response.text}
        raise Exception(f"SURUS API error {response.status_code}: {error_json}") from err

    result = response.json()
    return result.get("choices", [{}])[0].get("message", {}).get("content", "")