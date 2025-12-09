# Backend/conftest.py
import sys
from pathlib import Path

# Project root = AeroSense (parent of Backend)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Ensure root (AeroSense) is on sys.path so "Data" package can be imported
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
