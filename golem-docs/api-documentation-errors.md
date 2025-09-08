# SURUS API Documentation Errors

## Critical Error: Incorrect Field Names

**Issue:** Whisper model documentation shows wrong form field name.

**Documentation states:** `files = {'audio': f}` (line 76)
**API requires:** `files = {'file': f}`

**Impact:** 400 Bad Request error for all Whisper transcriptions.

## Error Details

| Model | Doc Field | Actual Field | Status |
|-------|-----------|--------------|---------|
| `surus-lat/whisper-large-v3-turbo-latam` | `'audio'` | `'file'` | ❌ Broken |
| `nvidia/canary-1b-v2` | `'file'` | `'file'` | ✅ Correct |

## Specific Lines to Fix

**File:** `golem-docs/audio-transcriptions.md`

**Line 76:** 
```python
# Wrong
files = {'audio': f}

# Correct  
files = {'file': f}
```

**Line 88 (JavaScript):** Similar error exists in JS example.

## Verification

**Error response:**
```json
{
  "error": {
    "message": "The 'file' parameter is required and must be an audio file",
    "type": "invalid_request_error",
    "param": "file"
  }
}
```

**Fix confirmed:** Both models now work correctly using `'file'` field.