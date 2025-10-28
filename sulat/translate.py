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
              **kwargs) -> str:
    """
    Translate text using SURUS API via the new /translate endpoint.
    """
    api_key = os.getenv("SURUS_API_KEY")
    if not api_key:
        raise ValueError("SURUS_API_KEY environment variable not set")

    api_url = "https://api.surus.dev/functions/v1/translate"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "text": text,
        "target_lang": target_lang
    }
    
    # Add any additional parameters passed as kwargs
    payload.update(kwargs)

    try:
        # Use a connect/read timeout tuple so unresponsive servers fail fast.
        response = requests.post(api_url, headers=headers, json=payload, timeout=(3.05, 27))
        response.raise_for_status()
    except requests.exceptions.Timeout as err:
        # Timed out connecting to or reading from the SURUS API
        raise TimeoutError("Timeout while contacting SURUS API") from err
    except requests.HTTPError as err:
        try:
            error_json = response.json()
        except ValueError:
            error_json = {"error": response.text}
        raise Exception(f"SURUS API error {response.status_code}: {error_json}") from err
    except requests.RequestException as err:
        # Catch other requests-related errors (connection errors, invalid URL, etc.)
        raise Exception("Error while contacting SURUS API") from err

    result = response.json()
    return result.get("text", str(result))