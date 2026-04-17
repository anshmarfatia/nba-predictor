"""Put the project root on sys.path so `from src.features...` imports work
regardless of the directory pytest is invoked from.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
