"""
Root pytest configuration for the backend test suite.

Inserts the directory containing ``app/`` at the front of ``sys.path`` 
so that all test files can import modules as ``from app.xxx import yyy`` 
consistently, matching the production import paths used throughout the codebase.
"""
import sys
import pathlib

# Ensure the backend package root is first on sys.path
_BACKEND_ROOT = str(pathlib.Path(__file__).parent)
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)
