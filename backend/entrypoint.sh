#!/bin/bash
set -e

# Seed the volume with pre-trained models if empty
if [ ! -f "$MODEL_DIR/xgb_categorizer.pkl" ]; then
    echo "Seeding model volume from build artifacts..."
    cp /app/ml/models/* "$MODEL_DIR/"
    echo "Models copied to $MODEL_DIR"
else
    echo "Models already exist at $MODEL_DIR"
fi

exec "$@"
