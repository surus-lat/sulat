#!/usr/bin/env python

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


def extract(
    text: str,
    json_schema: dict,
    model: Optional[str] = "google/gemma-3n-E4B-it",
    temperature: float = 0.0,
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
        raise ValueError("SURUS_API_KEY environment variable not set")

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

    response = requests.post(api_url, headers=headers, json=data)

    if response.status_code != 200:
        error_text = response.text
        try:
            error_json = response.json()
            raise Exception(f"API Error {response.status_code}: {error_json}")
        except ValueError:
            raise Exception(f"API Error {response.status_code}: {error_text}")

    response_json = response.json()
    message_content = response_json["choices"][0]["message"]["content"]
    return json.loads(message_content)