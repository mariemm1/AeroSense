#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (AeroSense) relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtualenv if it exists
if [ -d ".venv" ]; then
  # Linux / WSL layout
  source ".venv/bin/activate"
fi

# Load Backend/.env into environment
if [ -f "Backend/.env" ]; then
  export $(grep -v '^#' Backend/.env | xargs)
fi

# Run S5P pipeline
python Data/ExtractData/s5p_pipeline.py \
  --regions "tunisia,ariana,tozeur,manouba" \
  --top 3 \
  --mongo-uri "$MONGO_URI" \
  --mongo-db "$MONGO_DB" \
  --mongo-s5p-col "$MONGO_S5P_COL"
