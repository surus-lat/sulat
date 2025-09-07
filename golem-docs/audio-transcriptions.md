# Audio Transcriptions

Convierte archivos de audio en texto. Es útil para transcribir grabaciones, entrevistas, etc.

## Solicitud

### Modelo: nvidia/canary-1b-v2 (Canary)

Canary es el modelo de transcripción de Nvidia, con capacidad adicional de traducción entre 25 idiomas. El idioma por defecto del endpoint es "es" (español). La API aún no soporta los parámetros response_format, temperature ni timestamp_granularities.

=== "Python"
    ```python
    import requests

    SURUS_API_KEY = "tu_clave_api"
    API_URL = "https://api.surus.dev/functions/v1/audio/transcriptions"
    headers = {"Authorization": "Bearer " + SURUS_API_KEY}

    with open('audio.wav', 'rb') as f:
        files = {'file': f}
        data = {
            'model': 'nvidia/canary-1b-v2',
            'source_lang': 'es',   # opcional
            'target_lang': 'es'    # opcional
        }
        response = requests.post(API_URL, headers=headers, data=data, files=files)
        print(response.json())
    ```

=== "JavaScript"
    ```javascript
    const SURUS_API_KEY = "tu_clave_api";
    const API_URL = 'https://api.surus.dev/functions/v1/audio/transcriptions';
    const formData = new FormData();
    formData.append('model', 'nvidia/canary-1b-v2');
    formData.append('file', audioFileInput.files[0]); // uso de 'file' para Canary
    formData.append('source_lang', 'es'); // opcional
    formData.append('target_lang', 'es'); // opcional

    fetch(API_URL, {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + SURUS_API_KEY
      },
      body: formData
    })
    .then(res => res.json())
    .then(data => console.log(data));
    ```

=== "cURL"
    ```bash
    curl -X POST https://api.surus.dev/functions/v1/audio/transcriptions \
      -H "Authorization: Bearer tu_clave_api" \
      -F "model=nvidia/canary-1b-v2" \
      -F "file=@file.wav" \
      -F "source_lang=es" \
      -F "target_lang=es"
    ```

---

### Modelo: Whisper (surus-lat/whisper-large-v3-turbo-latam)

Whisper es un modelo de transcripción y traducción de código abierto desarrollado por OpenAI. El modelo `surus-lat/whisper-large-v3-turbo-latam` está optimizado para español latinoamericano.

=== "Python"
    ```python
    import requests

    SURUS_API_KEY = "tu_clave_api"
    API_URL = "https://api.surus.dev/functions/v1/audio/transcriptions"
    headers = {"Authorization": "Bearer " + SURUS_API_KEY}

    with open('audio.wav', 'rb') as f:
        files = {'audio': f}
        data = {'model': 'surus-lat/whisper-large-v3-turbo-latam'}
        response = requests.post(API_URL, headers=headers, data=data, files=files)
        print(response.json())
    ```

=== "JavaScript"
    ```javascript
    const SURUS_API_KEY = "tu_clave_api";
    const API_URL = 'https://api.surus.dev/functions/v1/audio/transcriptions';
    const formData = new FormData();
    formData.append('model', 'surus-lat/whisper-large-v3-turbo-latam');
    formData.append('audio', audioFileInput.files[0]); // audioFileInput is an <input type='file'>
    fetch(API_URL, {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + SURUS_API_KEY
      },
      body: formData
    })
    .then(res => res.json())
    .then(data => console.log(data));
    ```

=== "cURL"
    ```bash
    curl -X POST https://api.surus.dev/functions/v1/audio/transcriptions \
      -H "Authorization: Bearer tu_clave_api" \
      -F "model=surus-lat/whisper-large-v3-turbo-latam" \
      -F "audio=@file.wav"
    ```

    
### Respuesta

```json
{
  "text": "Hola, ¿cómo estás?"
}
```

---

## Parámetros extra

Podés enviar los siguientes parámetros adicionales para controlar el comportamiento de la generación y el formato de la respuesta.

Nota: Algunos parámetros aplican solo a ciertos modelos. En particular, response_format, temperature y timestamp_granularities son utilizados por modelos Whisper; nvidia/canary-1b-v2 no soporta esos parámetros y usa en cambio campos de idioma opcionales (source_lang, target_lang).

### Parámetros de formato y salida

- `response_format` (`string`, default: `"json"`): El formato de la salida. Opciones: `json`, `text`, `srt`, `verbose_json`, o `vtt`. Para algunos modelos, solo se soporta `json`.
- `stream` (`bool`, default: `False`): Si se establece en `true`, la respuesta será transmitida al cliente usando server-sent events. Nota: No todos los modelos soportan streaming.
- `temperature` (`number`, default: `0`): La temperatura de muestreo, entre 0 y 1. Valores más altos como 0.8 harán la salida más aleatoria, mientras que valores más bajos como 0.2 la harán más enfocada y determinística.
- `timestamp_granularities` (`array`, default: `["segment"]`): La granularidad de timestamps a incluir. Debe usarse con `response_format` establecido en `verbose_json`. Opciones: `word`, `segment`.

=== "Python"
    ```python
    import requests

    SURUS_API_KEY = "tu_clave_api"
    API_URL = "https://api.surus.dev/functions/v1/audio/transcriptions"
    headers = {"Authorization": "Bearer " + SURUS_API_KEY}

    with open('audio.wav', 'rb') as f:
        files = {'audio': f}
        data = {
            'model': 'surus-lat/whisper-large-v3-turbo-latam',
            'response_format': 'verbose_json',
            'temperature': 0.2,
            'timestamp_granularities': ['word', 'segment']
        }
        response = requests.post(API_URL, headers=headers, data=data, files=files)
        print(response.json())
    ```

=== "JavaScript"
    ```javascript
    const SURUS_API_KEY = "tu_clave_api";
    const API_URL = 'https://api.surus.dev/functions/v1/audio/transcriptions';
    const formData = new FormData();
    formData.append('model', 'surus-lat/whisper-large-v3-turbo-latam');
    formData.append('audio', audioFileInput.files[0]);
    formData.append('response_format', 'verbose_json');
    formData.append('temperature', '0.2');
    formData.append('timestamp_granularities', 'word');
    formData.append('timestamp_granularities', 'segment');
    
    fetch(API_URL, {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + SURUS_API_KEY
      },
      body: formData
    })
    .then(res => res.json())
    .then(data => console.log(data));
    ```

=== "cURL"
    ```bash
    curl -X POST https://api.surus.dev/functions/v1/audio/transcriptions \
      -H "Authorization: Bearer tu_clave_api" \
      -F "model=surus-lat/whisper-large-v3-turbo-latam" \
      -F "audio=@file.wav" \
      -F "response_format=verbose_json" \
      -F "temperature=0.2" \
      -F "timestamp_granularities=word" \
      -F "timestamp_granularities=segment"
    ```
