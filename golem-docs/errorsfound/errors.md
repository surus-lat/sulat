(surus) dobleefe@FrancisdeMacBook-Air surus % uv run testing.py
Traceback (most recent call last):
  File "/Users/dobleefe/surus/testing.py", line 7, in <module>
    surus.transcribe("audio_test.wav")
    ~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^
  File "/Users/dobleefe/surus/surus/transcribe.py", line 76, in transcribe
    response.raise_for_status()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/Users/dobleefe/surus/.venv/lib/python3.13/site-packages/requests/models.py", line 1026, in raise_for_status
    raise HTTPError(http_error_msg, response=self)
requests.exceptions.HTTPError: 400 Client Error: Bad Request for url: https://api.surus.dev/functions/v1/audio/transcriptions
(surus) dobleefe@FrancisdeMacBook-Air surus % 