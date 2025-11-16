#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (AeroSense) relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtualenv if it exists
if [ -d ".venv" ]; then
  source ".venv/bin/activate"
fi

# Load Backend/.env into environment
if [ -f "Backend/.env" ]; then
  export $(grep -v '^#' Backend/.env | xargs)
fi

# Run S3 LST pipeline
python Data/ExtractData/s3_pipeline.py \
  --regions "tunisia,ariana,tozeur" \
  --top 3 \
  --mongo-uri "$MONGO_URI" \
  --mongo-db "$MONGO_DB" \
  --mongo-s3-col "$MONGO_S3_COL"
