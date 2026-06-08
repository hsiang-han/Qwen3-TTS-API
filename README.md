# Qwen3-TTS-API

OpenAI-compatible Text-to-Speech API powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba Qwen Team).

State-of-the-art multilingual TTS with 10 languages, voice cloning, voice design, and instruction-based control. Supports RTX 50-series (Blackwell) GPUs.

## Features

- OpenAI-compatible `/v1/audio/speech` endpoint
- Voice cloning from 3-second reference audio
- 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- Multiple model sizes: 0.6B (~3GB VRAM) and 1.7B (~6GB VRAM)
- Streaming-ready architecture (97ms first-packet latency)

## Quick Start

```bash
docker compose -f docker/gpu/docker-compose.yml up
```

## API Endpoints

### Health Check
```
GET /health
```

### List Models & Voices
```
GET /v1/models
GET /v1/voices
```

### Text-to-Speech (CustomVoice / VoiceDesign)
```
POST /v1/audio/speech
```
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input | string | required | Text to synthesize |
| voice | string | Vivian | Speaker name |
| language | string | Auto | Language (Auto, Chinese, English, Japanese, etc.) |
| instruct | string | null | Instruction for tone/emotion control |

### Voice Clone (Base model only)
```
POST /v1/audio/speech/clone
```
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| input | string | required | Text to synthesize |
| ref_audio | file | required | Reference audio file (WAV) |
| ref_text | string | null | Transcript of reference audio (improves quality) |
| language | string | Auto | Target language |
| x_vector_only | bool | false | Use speaker embedding only (no ICL) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_ID | Qwen/Qwen3-TTS-12Hz-1.7B-Base | HuggingFace model ID or local path |
| DTYPE | bfloat16 | Model precision (float16, bfloat16, float32) |
| DEVICE | cuda:0 | Device to load model on |
| ATTN_IMPLEMENTATION | flash_attention_2 | Attention backend (flash_attention_2, sdpa, eager) |
| PORT | 8080 | API server port |
| HF_HOME | /root/.cache/huggingface | HuggingFace cache directory |
| HF_ENDPOINT | https://huggingface.co | HuggingFace mirror (use https://hf-mirror.com for China) |

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
