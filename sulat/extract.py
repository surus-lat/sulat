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

import json

# Load environment variables from .env file
load_dotenv()


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
    json_schema: dict,
    model: Optional[str] = "google/gemma-3n-E4B-it",
    temperature: float = 0.0,
    timeout: float = 30.0,
) -> dict:
    """
    Extract structured information from text using SURUS API.

    Args:
        text: The input text to process.
        json_schema: The JSON schema to guide the extraction.
        model: The name of the model to use for extraction.
        temperature: The temperature for the model.

    Returns:
        A dictionary containing the extracted information.
    """
    api_key = os.getenv("SURUS_API_KEY")
    if not api_key:
        raise MissingAPIKeyError("SURUS_API_KEY environment variable not set")

    api_url = "https://api.surus.dev/functions/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = [
        {
            "role": "system",
            "content": f"You are a helpful assistant designed to output JSON conforming to this schema: {json_schema}",
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