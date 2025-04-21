#!/usr/bin/env python
"""
Wrapper script to run the preprocessing module.
This script helps avoid import errors by ensuring the proper Python path.
"""
import os
import sys

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    # Import the module
    import src.preprocessing.preprocess
    
    # The module will handle argument parsing and execution
    # This wrapper simply ensures the correct Python path 