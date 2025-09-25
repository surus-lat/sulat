"""Scoring utilities for the sulat package."""

import re
from typing import Optional, Dict, Any
from .text_utils import _norm, _norm_text, _should_keep_punct_for_field, _to_bool, _parse_number


def _score_date(g, p, thresholds=None):
    """
    Score date similarity based on absolute delta thresholds.
    
    Args:
        g: Gold (expected) date value
        p: Predicted date value
        thresholds: Dict with threshold settings (days, scores). Default thresholds:
            - 0 days -> 1.0
            - <=1 day -> 0.8
            - <=7 days -> 0.5
            - else relative decay
    
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if thresholds is None:
        thresholds = {
            "exact_match_days": 0,  # 0 days -> 1.0
            "close_match_days": 1,  # <=1 day -> 0.8
            "week_match_days": 7,   # <=7 days -> 0.5
            "relative_decay_start": 30  # Start relative decay after 30 days
        }
    
    # Parse both dates
    g_date = _parse_date(g)
    p_date = _parse_date(p)
    
    # Both None
    if g_date is None and p_date is None:
        return 1.0
    
    # One is None
    if g_date is None or p_date is None:
        # If hallucination penalty is enabled, this should return 0.0 when gold is None but pred is given
        return 0.0
    
    # Calculate the absolute difference in days
    date_diff = abs((g_date - p_date).days)
    
    # Apply thresholds
    if date_diff <= thresholds["exact_match_days"]:
        return 1.0
    elif date_diff <= thresholds["close_match_days"]:
        return 0.8
    elif date_diff <= thresholds["week_match_days"]:
        return 0.5
    elif date_diff <= thresholds["relative_decay_start"]:
        # Linear decay from 0.5 to 0.0 between week_match_days and relative_decay_start
        days_range = thresholds["relative_decay_start"] - thresholds["week_match_days"]
        days_past_week = date_diff - thresholds["week_match_days"]
        score = 0.5 - (0.5 * (days_past_week / days_range))
        return max(0.0, score)
    else:
        # For dates far apart, use relative decay based on how much time has passed relative to the expected date
        # This prevents extremely old dates from always getting 0.0 when they might be contextually related
        return max(0.0, 0.5 * (thresholds["relative_decay_start"] / date_diff))


def _score_email(g, p):
    """
    Compare email addresses with case-insensitive matching and exact match requirement.
    
    Args:
        g: Gold (expected) email address
        p: Predicted email address
    
    Returns:
        1.0 if emails match (case-insensitive, after normalization), 0.0 otherwise
    """
    if g is None and p is None:
        return 1.0
    if g is None or p is None:
        return 0.0
    
    # Convert to lowercase and strip whitespace
    g_norm = str(g).strip().lower()
    p_norm = str(p).strip().lower()
    
    # Basic email validation: must contain exactly one @
    g_parts = g_norm.split('@')
    p_parts = p_norm.split('@')
    
    if len(g_parts) != 2 or len(p_parts) != 2:
        # If either isn't a valid email format, check if they're exactly the same
        return 1.0 if g_norm == p_norm else 0.0
    
    # Compare local part and domain separately (both case-insensitive)
    g_local, g_domain = g_parts
    p_local, p_domain = p_parts
    
    # Domain should be normalized to lowercase (domain names are case-insensitive)
    g_domain = g_domain.lower()
    p_domain = p_domain.lower()
    
    # Local part is case-sensitive by RFC standard, but many providers handle it case-insensitively
    # For our purposes, we'll use case-insensitive comparison for both parts to be lenient
    return 1.0 if (g_local.lower() == p_local.lower() and g_domain == p_domain) else 0.0


def _score_url(g, p):
    """
    Compare URLs by normalizing scheme, stripping trailing slash, and canonicalizing host lowercasing.
    
    Args:
        g: Gold (expected) URL
        p: Predicted URL
    
    Returns:
        1.0 if URLs match after normalization, 0.0 otherwise
    """
    if g is None and p is None:
        return 1.0
    if g is None or p is None:
        return 0.0
    
    try:
        from urllib.parse import urlparse, urlunparse
    except ImportError:
        # Fallback normalization without urllib
        g_norm = _normalize_url_fallback(str(g).strip())
        p_norm = _normalize_url_fallback(str(p).strip())
        return 1.0 if g_norm == p_norm else 0.0
    
    try:
        g_parsed = urlparse(str(g).strip())
        p_parsed = urlparse(str(p).strip())
    except Exception:
        # If parsing fails, fall back to string comparison
        g_norm = _normalize_url_fallback(str(g).strip())
        p_norm = _normalize_url_fallback(str(p).strip())
        return 1.0 if g_norm == p_norm else 0.0
    
    # Normalize scheme (default to https if not provided)
    g_scheme = g_parsed.scheme.lower() if g_parsed.scheme else 'https'
    p_scheme = p_parsed.scheme.lower() if p_parsed.scheme else 'https'
    
    # Normalize netloc (domain) to lowercase
    g_netloc = g_parsed.netloc.lower()
    p_netloc = p_parsed.netloc.lower()
    
    # Remove 'www.' prefix for comparison (common normalization)
    g_netloc = g_netloc.lstrip('www.')
    p_netloc = p_netloc.lstrip('www.')
    
    # Normalize path - remove trailing slash if not root
    g_path = g_parsed.path.rstrip('/') if g_parsed.path != '/' else g_parsed.path
    p_path = p_parsed.path.rstrip('/') if p_parsed.path != '/' else g_parsed.path
    
    # Reconstruct URL with normalized components
    g_normalized = urlunparse((g_scheme, g_netloc, g_path, g_parsed.params, g_parsed.query, g_parsed.fragment))
    p_normalized = urlunparse((p_scheme, p_netloc, p_path, p_parsed.params, p_parsed.query, p_parsed.fragment))
    
    return 1.0 if g_normalized == p_normalized else 0.0


def _normalize_url_fallback(url_str):
    """
    Fallback URL normalization when urllib is not available.
    """
    s = url_str.strip()
    
    # Add default scheme if missing
    if not s.startswith(('http://', 'https://')):
        s = 'https://' + s
    
    # Convert to lowercase (for domain)
    parts = s.split('://', 1)
    if len(parts) == 2:
        scheme, rest = parts
        # Split at first slash after scheme to separate domain/path
        if '/' in rest:
            domain, path = rest.split('/', 1)
            domain = domain.lower().lstrip('www.')
            # Remove trailing slash from path if not root
            if path and path != '/':
                path = path.rstrip('/')
            s = f"{scheme}://{domain}/{path}" if path else f"{scheme}://{domain}"
        else:
            domain = rest.lower().lstrip('www.')
            s = f"{scheme}://{domain}"
    
    return s


def _score_phone(g, p):
    """
    Compare phone numbers by keeping digits and '+', dropping spaces/() -, comparing normalized forms.
    
    Args:
        g: Gold (expected) phone number
        p: Predicted phone number
    
    Returns:
        1.0 if phone numbers match after normalization, 0.0 otherwise
    """
    if g is None and p is None:
        return 1.0
    if g is None or p is None:
        return 0.0
    
    def normalize_phone(phone_str):
        # Keep only digits and '+', remove everything else
        import re
        digits = re.sub(r'[^\d+]', '', str(phone_str).strip())
        
        # Remove leading '1' if it looks like country code and there are 11 digits total
        if len(digits) == 11 and digits.startswith('1'):
            digits = digits[1:]
        
        return digits
    
    g_norm = normalize_phone(g)
    p_norm = normalize_phone(p)
    
    return 1.0 if g_norm == p_norm else 0.0


def _score_id(g, p):
    """
    Compare IDs with exact match after trimming whitespace.
    
    Args:
        g: Gold (expected) ID
        p: Predicted ID
    
    Returns:
        1.0 if IDs match after trimming, 0.0 otherwise
    """
    if g is None and p is None:
        return 1.0
    if g is None or p is None:
        return 0.0
    
    # Convert to string, strip whitespace
    g_norm = str(g).strip()
    p_norm = str(p).strip()
    
    return 1.0 if g_norm == p_norm else 0.0


def _flatten_to_kv(d, prefix=""):
    """
    Flatten a nested dictionary to a set of "key_path=value" pairs.
    
    Args:
        d: The dictionary to flatten
        prefix: The prefix to use for nested keys (default: "")
    
    Returns:
        A dictionary mapping key paths to values
    """
    if d is None:
        return {}
    
    result = {}
    if isinstance(d, dict):
        for k, v in d.items():
            new_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                # Recursively flatten nested dictionaries
                result.update(_flatten_to_kv(v, new_key))
            else:
                # For non-dict values, store the key-value pair
                result[new_key] = v
    elif isinstance(d, (list, tuple)):
        # Handle lists by creating indexed keys
        for i, v in enumerate(d):
            new_key = f"{prefix}.{i}" if prefix else str(i)
            if isinstance(v, dict):
                result.update(_flatten_to_kv(v, new_key))
            else:
                result[new_key] = v
    else:
        # If it's not a dict or list, just return it as a single value with the prefix as key
        if prefix:
            result[prefix] = d
    
    return result


def _score_dict(g, p, field_name=None):
    """
    Score dictionary similarity by comparing key-value pairs.
    
    Args:
        g: Gold (expected) dictionary
        p: Predicted dictionary
        field_name: Name of the field (for recursive calls to nested dicts)
    
    Returns:
        Similarity score between 0.0 and 1.0
    """
    if g is None and p is None:
        return 1.0
    if g is None or p is None:
        return 0.0
    
    # Convert to dictionaries if they're not already
    if isinstance(g, str):
        try:
            import json
            g = json.loads(g)
        except:
            g = {"error": g}
    if isinstance(p, str):
        try:
            import json
            p = json.loads(p)
        except:
            p = {"error": p}
    
    if not isinstance(g, dict):
        g = {"value": g}
    if not isinstance(p, dict):
        p = {"value": p}
    
    # Flatten both dictionaries to key-value pairs
    g_flat = _flatten_to_kv(g)
    p_flat = _flatten_to_kv(p)
    
    # Get all unique keys from both dictionaries
    all_keys = set(g_flat.keys()) | set(p_flat.keys())
    
    if not all_keys:
        return 1.0
    
    # Calculate scores for keys present in gold
    gold_keys = set(g_flat.keys())
    scores = []
    valid_comparisons = 0
    
    for key in gold_keys:
        g_val = g_flat.get(key)
        p_val = p_flat.get(key)
        
        # Determine the type of value to decide how to score it
        if p_val is None and g_val is not None:
            # Gold has a value, prediction doesn't (hallucination penalty might apply)
            score = 0.0
        elif g_val is None and p_val is not None:
            # Prediction has a value, gold doesn't
            score = 0.0  # This could be affected by hallucination penalty
        elif g_val is None and p_val is None:
            score = 1.0
        else:
            # Both have values, score them based on their types
            # We'll use a recursive approach with the main scoring logic
            score = _score_field_recursive(g_val, p_val, field_name=f"{field_name}.{key}" if field_name else key)
        
        scores.append(score)
        valid_comparisons += 1
    
    # Calculate macro-average score across keys present in gold
    if valid_comparisons > 0:
        avg_score = sum(scores) / len(scores) if len(scores) > 0 else 1.0
    else:
        avg_score = 0.0  # No keys in gold, so can't score properly
    
    # If there are keys in prediction not in gold, that might be penalized depending on hallucination settings
    extra_keys_in_pred = set(p_flat.keys()) - set(g_flat.keys())
    if extra_keys_in_pred:
        # For now, we focus on keys in gold, but could implement penalty for extra keys
        pass
    
    return avg_score


def _score_field_recursive(g_val, p_val, field_name=""):
    """
    Recursively score fields accounting for nested structures.
    This is a helper for scoring dict values with the appropriate logic.
    """
    # Detect field type based on field name
    field_type = _detect_field_type(field_name)
    
    # Apply type-specific scoring
    if field_type == "date":
        return _score_date(g_val, p_val)
    elif field_type == "email":
        return _score_email(g_val, p_val)
    elif field_type == "url":
        return _score_url(g_val, p_val)
    elif field_type == "phone":
        return _score_phone(g_val, p_val)
    elif field_type == "id":
        return _score_id(g_val, p_val)
    elif field_type == "dict":
        return _score_dict(g_val, p_val, field_name)
    elif isinstance(g_val, dict) or isinstance(p_val, dict):
        return _score_dict(g_val, p_val, field_name)
    elif field_type == "numeric":
        # For numeric comparison, we need to use the number parsing from the original code
        gb, pb = _to_bool(g_val), _to_bool(p_val)
        if gb is not None or pb is not None:
            if gb is None or pb is not None:
                return 0.0
            return 1.0 if gb == pb else 0.0

        g_num = _parse_number(g_val)
        p_num = _parse_number(p_val)
        
        if g_num is None and p_num is None:
            return 1.0
        if g_num is None or p_num is None:
            return 0.0

        diff = abs(p_num - g_num)
        if diff == 0:
            return 1.0
        if diff <= 2:  # abs_close
            return 0.8
        if diff <= 6:  # abs_ok
            return 0.5
        if abs(g_num) > 12:  # relative_after
            return max(0.0, 1.0 - (diff / abs(g_num)))
        return 0.0
    else:
        # Default to text scoring
        return _score_text(str(g_val) if g_val is not None else "", 
                          str(p_val) if p_val is not None else "", 
                          field_name)


def _detect_field_type(field_name):
    """
    Detect field type based on its name for specialized comparison.
    
    Args:
        field_name: Name of the field to analyze
        
    Returns:
        String representing the detected field type
    """
    if not field_name:
        return "text"
    
    field_lower = field_name.lower()
    
    # Date types
    date_patterns = ['date', 'time', 'year', 'month', 'day', 'timestamp', 'created', 'updated', 'modified', 
                     'birth', 'anniversary', 'deadline', 'due', 'scheduled', 'start', 'end', 'from', 'to']
    if any(pattern in field_lower for pattern in date_patterns):
        return "date"
    
    # Email types
    email_patterns = ['email', 'mail', 'e_mail', 'e-mail']
    if any(pattern in field_lower for pattern in email_patterns):
        return "email"
    
    # URL types
    url_patterns = ['url', 'uri', 'link', 'website', 'web', 'page', 'href', 'src']
    if any(pattern in field_lower for pattern in url_patterns):
        return "url"
    
    # Phone types
    phone_patterns = ['phone', 'tel', 'telephone', 'mobile', 'cell', 'contact']
    if any(pattern in field_lower for pattern in phone_patterns):
        return "phone"
    
    # ID types
    id_patterns = ['id', 'identifier', 'uuid', 'guid', 'code', 'number', 'num', 'ref', 'reference']
    if any(pattern in field_lower for pattern in id_patterns):
        return "id"
    
    # Dict types
    dict_patterns = ['data', 'info', 'details', 'properties', 'attributes', 'metadata', 'config', 'settings']
    if any(pattern in field_lower for pattern in dict_patterns):
        return "dict"
    
    # Numeric types (if inferred from field names)
    numeric_patterns = ['count', 'amount', 'price', 'cost', 'age', 'score', 'rating', 'size', 'length', 
                        'width', 'height', 'weight', 'percentage', 'percent', 'pct', 'value']
    if any(pattern in field_lower for pattern in numeric_patterns):
        return "numeric"
    
    # Default to text
    return "text"


def _parse_date(s):
    """
    Parse a string to a date with python-dateutil if available; fallback to simple formats without hard dependency.
    Support timezone-agnostic comparisons if no timezone data.
    
    Args:
        s: Input string to parse as a date
        
    Returns:
        datetime object if parseable, otherwise None
    """
    if s is None:
        return None
    
    # First try with python-dateutil if available
    try:
        from dateutil import parser as date_parser
        from dateutil.parser import ParserError
        if isinstance(s, str):
            return date_parser.parse(s, fuzzy=True, ignoretz=True)
        elif hasattr(s, 'date'):  # datetime object
            return s
    except ImportError:
        # If python-dateutil is not available, use fallback methods
        pass
    except (ValueError, ParserError):
        # If dateutil fails, use fallback methods
        pass
    
    # Fallback to simple date formats without hard dependency
    import datetime
    
    if isinstance(s, str):
        # Try common date formats
        date_formats = [
            "%Y-%m-%d",      # 2023-01-15
            "%m/%d/%Y",      # 01/15/2023
            "%d/%m/%Y",      # 15/01/2023
            "%m-%d-%Y",      # 01-15-2023
            "%d-%m-%Y",      # 15-01-2023
            "%Y/%m/%d",      # 2023/01/15
            "%d.%m.%Y",      # 15.01.2023
            "%m.%d.%Y",      # 01.15.2023
            "%B %d, %Y",     # January 15, 2023
            "%b %d, %Y",     # Jan 15, 2023
            "%d %B %Y",      # 15 January 2023
            "%d %b %Y",      # 15 Jan 2023
            "%Y-%m-%d %H:%M:%S",  # 2023-01-15 14:30:00
            "%Y-%m-%dT%H:%M:%S",  # ISO format 2023-01-15T14:30:00
            "%Y-%m-%dT%H:%M:%SZ", # ISO format with Z 2023-01-15T14:30:00Z
            "%Y-%m-%d %H:%M:%S.%f",  # With microseconds
        ]
        
        for fmt in date_formats:
            try:
                return datetime.datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        
        # Try to extract date parts with regex as last resort
        import re
        # Look for YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD patterns
        date_match = re.search(r'(\d{4})[^\d](\d{1,2})[^\d](\d{1,2})', s)
        if date_match:
            try:
                year, month, day = map(int, date_match.groups())
                return datetime.date(year, month, day)
            except ValueError:
                pass
        
        # Look for DD/MM/YYYY or DD-MM-YYYY patterns where DD comes first
        date_match_eu = re.search(r'(\d{1,2})[^\d](\d{1,2})[^\d](\d{4})', s)
        if date_match_eu:
            try:
                day, month, year = map(int, date_match_eu.groups())
                # Only if day > 12 (which would make it invalid as a month), assume EU format
                if day > 12:
                    return datetime.date(year, month, day)
                else:
                    # Could be ambiguous, let's try to be more careful
                    # Check if first number is > 12, then it's likely a day
                    first_num = int(date_match_eu.group(1))
                    if first_num > 12:
                        return datetime.date(year, month, day)  # EU format
                    # Otherwise, we can't be sure, would need more context
            except ValueError:
                pass

    elif isinstance(s, datetime.date):
        return s
    elif isinstance(s, datetime.datetime):
        return s.date()
    
    return None


def _score_text(g, p, field_name: str = None):
    """Score text similarity."""
    # This is a simplified version of the nested function in make_universal_metric
    # It's needed here since _score_field_recursive calls _score_text
    from .text_utils import _norm_text, _should_keep_punct_for_field
    
    # Normalize types to strings
    g = "" if g is None else str(g)
    p = "" if p is None else str(p)

    # Determine normalization function based on field name
    if field_name and _should_keep_punct_for_field(field_name):
        norm_func = lambda x: _norm_text(x, keep_punct=True)
    else:
        from .text_utils import _norm
        norm_func = _norm

    g_norm = norm_func(g)
    p_norm = norm_func(p)

    # Both blank
    if g_norm == "" and p_norm == "":
        return 1.0
    # Gold blank
    if g_norm == "":
        return 0.0
    # Pred blank
    if p_norm == "":
        return 0.0

    # Default to simple equality for this basic implementation
    return 1.0 if g_norm == p_norm else 0.0