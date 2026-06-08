#!/bin/bash
set -e

echo "=== Qwen3-TTS-API ==="
echo "Model: ${MODEL_ID}"
echo "Dtype: ${DTYPE}"
echo "Attn:  ${ATTN_IMPLEMENTATION}"
echo "Port:  ${PORT}"
echo "======================"

exec python -m uvicorn api.main:app \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --log-level info \
    --timeout-keep-alive 65
