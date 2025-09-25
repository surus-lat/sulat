"""Text processing utilities for the sulat package."""

import re
from typing import Optional, Union


def _to_int(text_or_num):
    if text_or_num is None:
        return None
    if isinstance(text_or_num, (int, float)):
        try:
            return int(text_or_num)
        except (ValueError, TypeError):
            return None
    if isinstance(text_or_num, str):
        m = re.search(r"-?\d+", text_or_num)
        if m:
            try:
                return int(m.group(0))
            except (ValueError, TypeError):
                return None
    return None


def _norm_text(s, keep_punct=False):
    """Normalize text by removing punctuation optionally.
    
    Args:
        s: Input string to normalize
        keep_punct: Whether to preserve punctuation (True) or strip it (False)
    
    Returns:
        Normalized string
    """
    if s is None:
        return ""
    
    if keep_punct:
        # Only normalize whitespace and case, keep punctuation
        s = str(s)
    else:
        # Strip all punctuation except @ (for emails) and . (for URLs/decimals)
        s = re.sub(r"[^\w\s@%.]", "", str(s))
    
    return " ".join(s.strip().lower().split())


def _norm(s):
    """Legacy _norm function that removes all punctuation."""
    return _norm_text(s, keep_punct=False)


def _to_bool(x):
    # Accept booleans and common string synonyms
    # This is a copy of the function inside make_universal_metric to make it globally accessible
    if isinstance(x, bool):
        return x
    s = _norm(x)
    if s in {"true", "yes", "y", "1"}:
        return True
    if s in {"false", "no", "n", "0"}:
        return False
    return None


def _parse_number(s):
    """
    Parse a string to a number with support for various formats:
    - Thousands separators: 1,234.56
    - Percentages: "12%" -> 0.12 or treated as 12 depending on context
    - Magnitudes: 3k -> 3000, 1.2M -> 1200000
    - Scientific notation: 1e3
    - Ranges: "10–12" -> mean of min and max
    This is a copy of the function inside make_universal_metric to make it globally accessible.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return float(s)
    
    s_str = str(s).strip()
    
    # Handle ranges (e.g., "10-12", "10–12", "10 to 12")
    range_patterns = [
        r"(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)"
    ]
    for pattern in range_patterns:
        match = re.search(pattern, s_str)
        if match:
            min_val = float(match.group(1))
            max_val = float(match.group(2))
            return (min_val + max_val) / 2  # Return mean of range
    
    # Handle magnitudes (k, M, B suffixes)
    magnitude_patterns = {
        r'(\d+(?:\.\d+)?)\s*k': lambda x: x * 1000,
        r'(\d+(?:\.\d+)?)\s*M': lambda x: x * 1000000,
        r'(\d+(?:\.\d+)?)\s*B': lambda x: x * 1000000000
    }
    for pattern, multiplier in magnitude_patterns.items():
        match = re.search(pattern, s_str, re.IGNORECASE)
        if match:
            num = float(match.group(1))
            return multiplier(num)
    
    # Handle percentages
    percent_match = re.search(r'(\d+(?:\.\d+)?)\s*%', s_str)
    if percent_match:
        num = float(percent_match.group(1))
        return num  # Return the percentage value as-is (e.g., 12 for "12%"), adjust as needed
    
    # Remove commas for thousands separators and try to parse
    s_clean = re.sub(r'[,$]', '', s_str)
    
    # Handle scientific notation
    try:
        return float(s_clean)
    except ValueError:
        # If it's not a valid number, try to extract the first number
        num_match = re.search(r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?', s_clean)
        if num_match:
            try:
                return float(num_match.group(0))
            except ValueError:
                return None
        return None


def _should_keep_punct_for_field(field_name):
    """
    Determine if punctuation should be preserved for a field based on its name.
    
    Args:
        field_name: The name of the field
        
    Returns:
        Boolean indicating whether to preserve punctuation
    """
    field_name_lower = field_name.lower()
    # Check for common field patterns that require punctuation
    punct_preserving_patterns = [
        'email', 'url', 'website', 'phone', 'id', 'identifier', 
        'uuid', 'guid', 'ip', 'address', 'account', 'username',
        'domain', 'uri', 'path', 'ref', 'reference', 'code'
    ]
    return any(pattern in field_name_lower for pattern in punct_preserving_patterns)