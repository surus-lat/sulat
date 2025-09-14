


```python
import requests

SURUS_API_KEY = "tu_clave_api"
API_URL = "https://api.surus.dev/functions/v1/audio/transcriptions"
headers = {"Authorization": "Bearer " + SURUS_API_KEY}

with open('audio.wav', 'rb') as f:
    files = {'file': f}
    data = {'model': 'nvidia/canary-1b-v2'}
    response = requests.post(API_URL, headers=headers, data=data, files=files)
    print(response.json())

```



```bash
curl -X POST https://api.surus.dev/functions/v1/audio/transcriptions \
  -H "Authorization: Bearer tu_clave_api" \
  -F "model=nvidia/canary-1b-v2" \
  -F "file=@file.wav" \
  -F "source_lang=es" \
  -F "target_lang=es"
```