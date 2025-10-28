#!/usr/bin/env python
"""
Test script for the autotune function
"""

import json
import os
from pathlib import Path
import tempfile

# Create some sample test data
sample_data = [
    {
        "input_text": "John Doe is 30 years old and works as an engineer.",
        "expected_output": {
            "name": "John Doe",
            "age": 30,
            "occupation": "engineer"
        }
    },
    {
        "input_text": "Jane Smith is 25 years old and works as a doctor.",
        "expected_output": {
            "name": "Jane Smith",
            "age": 25,
            "occupation": "doctor"
        }
    }
]

def test_autotune():
    """Test the autotune function with sample data."""
    from sulat.extract import autotune
    
    # Create a temporary directory for test data
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a sample data file
        data_file = Path(temp_dir) / "sample_data.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2)
        
        # Test the autotune function
        result = autotune(
            data_source=temp_dir,
            save_optimized_name="test_optimized_extractor",
            input_key="input_text",
            output_key="expected_output",
            max_examples=10,
            seed=42,
            algorithm="miprov2",
            num_trials=2,  # Reduce for faster testing
            max_bootstrapped_demos=2,  # Reduce for faster testing
            max_labeled_demos=2,  # Reduce for faster testing
            optimize_metric=False  # Disable for faster testing
        )
        
        print("Autotune result:", result)
        
        # Verify the result has expected keys
        assert "status" in result
        assert result["status"] == "success"
        assert "optimized_name" in result
        assert "training_examples" in result
        assert "output_fields" in result
        assert "data_source" in result
        
        print("Test passed!")

if __name__ == "__main__":
    test_autotune()