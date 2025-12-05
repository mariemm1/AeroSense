# Data/ExtractData_IAModel/config_iamodel.py
from __future__ import annotations

from pathlib import Path
import os

# Resolve project root: AEROSENSE/
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

# Where your training CSVs live (you can change this or use env var)
# Put your S5P_*.csv + MODIS_LST_*.csv there.
DATA_CSV_DIR = Path(
    os.getenv(
        "IA_DATA_DIR",
        PROJECT_ROOT / "Data" / "ML_Datasets"  # <- create & move CSVs here
    )
)

# Where models + scalers + metadata will be saved
MODELS_DIR = Path(
    os.getenv(
        "IA_MODELS_DIR",
        PROJECT_ROOT / "Data" / "ExtractData_IAModel" / "artifacts"
    )
)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
