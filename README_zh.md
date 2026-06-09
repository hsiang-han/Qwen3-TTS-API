# Qwen3-TTS-API

[English](README.md)

基于 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 的 OpenAI 兼容语音合成 API，使用 [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) CUDA 图加速。

比原版推理快 7-10 倍。RTX 4060/5060 Ti 级别显卡即可实时生成。无需 flash-attn、vLLM 或 Triton——纯 CUDA 图加速。

## 功能特性

- OpenAI 兼容的 `/v1/audio/speech` 接口（JSON body）
- **CUDA 图加速** — 比原版快 7-10 倍
- 流式输出（`"stream": true` 返回分块 WAV）
- 3 秒参考音频即可克隆声音
- 10 种语言：中文、英文、日语、韩语、德语、法语、俄语、葡萄牙语、西班牙语、意大利语
- 9 种内置音色（CustomVoice 模型）+ 指令情感控制
- 不需要 flash-attn
- 支持 RTX 50 系列（Blackwell）显卡

## 快速开始

```bash
docker run -d --gpus all \
  -p 8080:8080 \
  -v /mnt/user/appdata/qwen3-tts-api/models:/root/.cache/huggingface \
  -e MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --shm-size=4g \
  --name qwen3-tts-api \
  ghcr.io/hsiang-han/qwen3-tts-api:latest
```

或使用 docker compose：

```bash
docker compose -f docker/gpu/docker-compose.yml up -d
```

首次启动下载模型（约 3-7GB），第一次请求时捕获 CUDA 图。国内用户设置 `HF_ENDPOINT=https://hf-mirror.com` 加速下载。

## 使用示例

```bash
# 使用内置音色合成语音
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "你好，这是一个测试。", "voice": "Vivian", "language": "Chinese"}' \
  --output output.wav

# 带情感指令（仅 1.7B CustomVoice）
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "我真的太开心了！", "voice": "Vivian", "instruct": "用特别开心的语气说"}' \
  --output happy.wav

# 流式输出
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "流式测试。", "voice": "Vivian", "stream": true}' \
  --output stream.wav

# 查看可用音色
curl http://localhost:8080/v1/voices
```

## 内置音色（CustomVoice 模型）

| 音色 | 描述 | 母语 |
|------|------|------|
| Vivian | 明亮、略带个性的年轻女声 | 中文 |
| Serena | 温暖、温柔的年轻女声 | 中文 |
| Uncle_Fu | 成熟男声，低沉醇厚 | 中文 |
| Dylan | 年轻北京男声，清晰自然 | 中文（北京话） |
| Eric | 活泼成都男声，略带沙哑 | 中文（四川话） |
| Ryan | 有活力的男声，节奏感强 | 英文 |
| Aiden | 阳光美式男声，中频清亮 | 英文 |
| Ono_Anna | 活泼日本女声，轻快灵动 | 日文 |
| Sohee | 温暖韩国女声，情感丰富 | 韩文 |

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1/audio/speech` | POST | 语音合成（JSON body，OpenAI 兼容） |
| `/v1/audio/speech/clone` | POST | 声音克隆（Form + 文件上传，仅 Base 模型） |
| `/v1/voices` | GET | 列出可用音色和语言 |
| `/v1/models` | GET | 列出模型 |
| `/health` | GET | 健康检查 |
| `/docs` | GET | Swagger 文档 |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODEL_ID | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | HuggingFace 模型 ID 或本地路径 |
| DTYPE | bfloat16 | 模型精度（float16、bfloat16、float32） |
| DEVICE | cuda:0 | CUDA 设备 |
| ATTN_IMPLEMENTATION | sdpa | 注意力后端（sdpa、eager） |
| PORT | 8080 | API 端口 |
| HF_HOME | /root/.cache/huggingface | HuggingFace 缓存目录 |
| HF_ENDPOINT | https://huggingface.co | HuggingFace 镜像地址（国内用 https://hf-mirror.com） |

## 可用模型

| 模型 ID | 类型 | 显存 | 功能 |
|---------|------|------|------|
| Qwen/Qwen3-TTS-12Hz-0.6B-Base | Base | ~3GB | 声音克隆 |
| Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice | CustomVoice | ~3GB | 9 种内置音色 |
| Qwen/Qwen3-TTS-12Hz-1.7B-Base | Base | ~6GB | 声音克隆 |
| Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | CustomVoice | ~6GB | 9 种内置音色 + 指令控制 |
| Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign | VoiceDesign | ~6GB | 用文字描述设计音色 |

## 硬件要求

- NVIDIA 显卡，4GB+ 显存（0.6B）或 8GB+ 显存（1.7B）
- 驱动版本 550+（Ampere/Ada）或 570+（Blackwell RTX 50 系列）
- 安装 NVIDIA Container Toolkit 的 Docker 环境

## 致谢

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — 阿里通义千问团队，模型
- [faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts) — [@andimarafioti](https://github.com/andimarafioti)，CUDA 图加速（7-10 倍提速）
