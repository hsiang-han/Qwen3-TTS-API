FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        sox libsox-dev ffmpeg curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    qwen-tts \
    fastapi \
    uvicorn[standard] \
    python-multipart \
    soundfile

# Install flash-attn from pre-built wheel (no source compilation needed)
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl || \
    echo "WARNING: flash-attn install failed, will use sdpa fallback"

COPY api/ ./api/
COPY docker/scripts/entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    DTYPE=bfloat16 \
    DEVICE=cuda:0 \
    ATTN_IMPLEMENTATION=flash_attention_2 \
    PORT=8080 \
    HF_HOME=/root/.cache/huggingface \
    HF_ENDPOINT=https://huggingface.co \
    PYTHONUNBUFFERED=1

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -sf http://localhost:${PORT}/health || exit 1

ENTRYPOINT ["./entrypoint.sh"]
