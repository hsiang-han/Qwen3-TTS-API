# Qwen3-TTS-API

OpenAI-compatible Text-to-Speech API powered by [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Alibaba Qwen Team).

State-of-the-art multilingual TTS with 10 languages, voice cloning, voice design, and instruction-based control. Supports RTX 50-series (Blackwell) GPUs.

## Features

- OpenAI-compatible `/v1/audio/speech` endpoint
- Voice cloning from 3-second reference audio
- 10 languages: Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian
- Multiple model sizes: 0.6B (~3GB VRAM) and 1.7B (~6GB VRAM)
- Streaming-ready architecture (97ms first-packet latency)
- Dual download source: HuggingFace (default) or ModelScope

## Quick Start

```bash
docker compose -f docker/gpu/docker-compose.yml up
```

First start downloads model files (~3-7GB). Default download source is HuggingFace.

For users in China, set `HF_ENDPOINT=https://hf-mirror.com` for faster downloads, or pre-download via ModelScope:

```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir /path/to/models/Qwen3-TTS-12Hz-1.7B-Base
# Then set MODEL_ID=/path/to/models/Qwen3-TTS-12Hz-1.7B-Base (local path)
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

---

# Qwen3-TTS-API（中文）

基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（阿里通义千问团队）的 OpenAI 兼容语音合成 API。

支持 10 种语言、3 秒声音克隆、语音设计、指令控制情感语气。支持 RTX 50 系列（Blackwell）显卡。

## 功能特性

- OpenAI 兼容的 `/v1/audio/speech` 接口
- 3 秒参考音频即可克隆声音
- 10 种语言：中文、英文、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语
- 多种模型规格：0.6B（约 3GB 显存）和 1.7B（约 6GB 显存）
- 首包延迟仅 97ms

## 快速开始

```bash
docker compose -f docker/gpu/docker-compose.yml up
```

首次启动会从 HuggingFace 下载模型（约 3-7GB）。

**国内用户加速方案：**

方案 1：使用 HuggingFace 镜像
```bash
# 设置环境变量
HF_ENDPOINT=https://hf-mirror.com
```

方案 2：使用 ModelScope 预下载，然后 MODEL_ID 传本地路径
```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-Base --local_dir /mnt/user/appdata/qwen3-tts-api/models/Qwen3-TTS-12Hz-1.7B-Base
# 然后设置 MODEL_ID=/root/.cache/huggingface/Qwen3-TTS-12Hz-1.7B-Base（容器内路径）
```

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODEL_ID | Qwen/Qwen3-TTS-12Hz-1.7B-Base | HuggingFace 模型 ID 或本地路径 |
| DTYPE | bfloat16 | 模型精度 |
| ATTN_IMPLEMENTATION | flash_attention_2 | 注意力后端 |
| PORT | 8080 | API 端口 |
| HF_ENDPOINT | https://huggingface.co | HuggingFace 镜像地址（国内用 https://hf-mirror.com） |

## 硬件要求

- NVIDIA 显卡，4GB+ 显存（0.6B）或 8GB+ 显存（1.7B）
- 驱动版本 550+（Ampere/Ada）或 570+（Blackwell RTX 50 系列）
- 安装 NVIDIA Container Toolkit 的 Docker 环境
