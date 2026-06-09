# Qwen3-TTS-API

[中文文档](README_zh.md)

OpenAI-compatible Text-to-Speech API powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) with [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) CUDA graph acceleration.

7-10x faster than stock inference. Real-time generation on RTX 4060/5060 Ti class GPUs. No flash-attn, no vLLM, no Triton — just CUDA graphs.

## Features

- OpenAI-compatible `/v1/audio/speech` endpoint (JSON body)
- **CUDA graph acceleration** — 7-10x faster than baseline
- Streaming output (`"stream": true` returns chunked WAV)
- Voice cloning from 3-second reference audio
- 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- 9 built-in voices (CustomVoice model) with instruction-based emotion control
- No flash-attn dependency required
- Supports RTX 50-series (Blackwell) GPUs

## Quick Start

```bash
docker run -d --gpus all \
  -p 8080:8080 \
  -v /mnt/user/appdata/qwen3-tts-api/models:/root/.cache/huggingface \
  -e MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --shm-size=4g \
  --name qwen3-tts-api \
  ghcr.io/hsiang-han/qwen3-tts-api:latest
```

Or with docker compose:

```bash
docker compose -f docker/gpu/docker-compose.yml up -d
```

First start downloads model (~3-7GB) and captures CUDA graphs on first request. China users: set `HF_ENDPOINT=https://hf-mirror.com`.

## Usage Examples

```bash
# Generate speech with built-in voice
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, this is a test.", "voice": "Vivian", "language": "English"}' \
  --output output.wav

# With emotion instruction (1.7B CustomVoice only)
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "我真的太开心了！", "voice": "Vivian", "instruct": "用特别开心的语气说"}' \
  --output happy.wav

# Streaming output
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Streaming test.", "voice": "Vivian", "stream": true}' \
  --output stream.wav

# List available voices
curl http://localhost:8080/v1/voices
```

## Built-in Voices (CustomVoice model)

| Voice | Description | Native Language |
|-------|-------------|-----------------|
| Vivian | Bright, slightly edgy young female | Chinese |
| Serena | Warm, gentle young female | Chinese |
| Uncle_Fu | Seasoned male, low mellow timbre | Chinese |
| Dylan | Youthful Beijing male, clear natural | Chinese (Beijing) |
| Eric | Lively Chengdu male, slightly husky | Chinese (Sichuan) |
| Ryan | Dynamic male, strong rhythmic drive | English |
| Aiden | Sunny American male, clear midrange | English |
| Ono_Anna | Playful Japanese female, light nimble | Japanese |
| Sohee | Warm Korean female, rich emotion | Korean |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/audio/speech` | POST | Text-to-speech (JSON body, OpenAI-compatible) |
| `/v1/audio/speech/clone` | POST | Voice cloning (Form + file upload, Base model only) |
| `/v1/voices` | GET | List available voices and languages |
| `/v1/models` | GET | List models |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger documentation |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_ID | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | HuggingFace model ID or local path |
| DTYPE | bfloat16 | Model precision (float16, bfloat16, float32) |
| DEVICE | cuda:0 | CUDA device |
| ATTN_IMPLEMENTATION | sdpa | Attention backend (sdpa, eager) |
| PORT | 8080 | API server port |
| HF_HOME | /root/.cache/huggingface | HuggingFace cache directory |
| HF_ENDPOINT | https://huggingface.co | HuggingFace mirror (China: https://hf-mirror.com) |

## Available Models

| Model ID | Type | VRAM | Features |
|----------|------|------|----------|
| Qwen/Qwen3-TTS-12Hz-0.6B-Base | Base | ~3GB | Voice clone |
| Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | CustomVoice | ~3GB | 9 built-in voices |
| Qwen/Qwen3-TTS-12Hz-1.7B-Base | Base | ~6GB | Voice clone |
| Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | CustomVoice | ~6GB | 9 built-in voices + instruction control |
| Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | VoiceDesign | ~6GB | Design voice from text description |

## Hardware Requirements

- NVIDIA GPU with 4GB+ VRAM (0.6B) or 8GB+ VRAM (1.7B)
- NVIDIA driver 550+ (Ampere/Ada) or 570+ (Blackwell RTX 50-series)
- Docker with NVIDIA Container Toolkit

## Credits

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team — the model
- [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) by [@andimarafioti](https://github.com/andimarafioti) — CUDA graph acceleration (7-10x speedup)
