# Qwen3-TTS-API

[中文文档](README_zh.md)

OpenAI-compatible Text-to-Speech API powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba Qwen Team).

State-of-the-art multilingual TTS with 10 languages, voice cloning, voice design, and instruction-based control. Supports RTX 50-series (Blackwell) GPUs.

## Features

- OpenAI-compatible `/v1/audio/speech` endpoint
- Voice cloning from 3-second reference audio
- 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- Multiple model sizes: 0.6B (~3GB VRAM) and 1.7B (~6GB VRAM)
- Optimized inference: torch.compile + fast codebook (3-6x faster than stock qwen-tts)
- Compile cache persisted to disk — fast restarts after first run

## Quick Start

```bash
docker run -d --gpus all \
  -p 8080:8080 \
  -v /mnt/user/appdata/qwen3-tts-api/models:/root/.cache/huggingface \
  -v /mnt/user/appdata/qwen3-tts-api/torch_cache:/root/.cache/torch_inductor \
  -e MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --shm-size=4g \
  --name qwen3-tts-api \
  ghcr.io/hsiang-han/qwen3-tts-api:latest
```

First start compiles optimized CUDA kernels (~60s) and downloads model (~3-7GB). Subsequent starts are fast (cache persisted to `torch_cache` volume).

Or with docker compose:

```bash
docker compose -f docker/gpu/docker-compose.yml up -d
```

First start downloads model files (~3-7GB) from HuggingFace. China users: set `HF_ENDPOINT=https://hf-mirror.com` for faster downloads.

## Usage Examples

```bash
# Generate speech with built-in voice
curl -X POST http://localhost:8080/v1/audio/speech \
  -F "input=Hello, this is a test." \
  -F "voice=Vivian" \
  -F "language=English" \
  --output output.wav

# With emotion instruction
curl -X POST http://localhost:8080/v1/audio/speech \
  -F "input=我真的太开心了！" \
  -F "voice=Vivian" \
  -F "language=Chinese" \
  -F "instruct=用特别开心的语气说" \
  --output happy.wav

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
| MODEL_ID | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | HuggingFace model ID or local path |
| DTYPE | bfloat16 | Model precision (float16, bfloat16, float32) |
| DEVICE | cuda:0 | Device to load model on |
| ATTN_IMPLEMENTATION | sdpa | Attention backend (sdpa, eager) |
| COMPILE_MODE | auto | Torch compile mode (auto, max-autotune, reduce-overhead, default). Auto selects based on GPU SM count |
| PORT | 8080 | API server port |
| HF_HOME | /root/.cache/huggingface | HuggingFace cache directory |
| HF_ENDPOINT | https://huggingface.co | HuggingFace mirror (China: https://hf-mirror.com) |
| TORCHINDUCTOR_CACHE_DIR | /root/.cache/torch_inductor | Torch compile cache (persist for fast restarts) |

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

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by Alibaba Qwen Team — the model and official inference code
- [Qwen3-TTS-streaming](https://github.com/dffdeeq/Qwen3-TTS-streaming) by [@dffdeeq](https://github.com/dffdeeq) — torch.compile optimizations and fast codebook that provide 3-6x inference speedup
