# Qwen3-TTS-API

[English](README.md)

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

首次启动会从 HuggingFace 下载模型（约 3-7GB）。

**国内用户加速方案：**

方案 1：使用 HuggingFace 镜像
```bash
# docker run 时加 -e HF_ENDPOINT=https://hf-mirror.com
```

方案 2：使用 ModelScope 预下载，然后 MODEL_ID 传本地路径
```bash
pip install modelscope
modelscope download --model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --local_dir /mnt/user/appdata/qwen3-tts-api/models/Qwen3-TTS-12Hz-1.7B-CustomVoice
# 然后设置 MODEL_ID 为容器内的本地路径
```

## 使用示例

```bash
# 使用内置音色合成语音
curl -X POST http://localhost:8080/v1/audio/speech \
  -F "input=你好，这是一个测试。" \
  -F "voice=Vivian" \
  -F "language=Chinese" \
  --output output.wav

# 带情感指令
curl -X POST http://localhost:8080/v1/audio/speech \
  -F "input=我真的太开心了！" \
  -F "voice=Vivian" \
  -F "language=Chinese" \
  -F "instruct=用特别开心的语气说" \
  --output happy.wav

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

### 健康检查
```
GET /health
```

### 列出模型和音色
```
GET /v1/models
GET /v1/voices
```

### 语音合成（CustomVoice / VoiceDesign）
```
POST /v1/audio/speech
```
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| input | string | 必填 | 要合成的文本 |
| voice | string | Vivian | 音色名称 |
| language | string | Auto | 语言（Auto、Chinese、English、Japanese 等） |
| instruct | string | null | 语气/情感控制指令 |

### 声音克隆（仅 Base 模型）
```
POST /v1/audio/speech/clone
```
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| input | string | 必填 | 要合成的文本 |
| ref_audio | file | 必填 | 参考音频文件（WAV） |
| ref_text | string | null | 参考音频的文字内容（提升克隆质量） |
| language | string | Auto | 目标语言 |
| x_vector_only | bool | false | 仅使用说话人特征向量（不使用 ICL） |

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| MODEL_ID | Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice | HuggingFace 模型 ID 或本地路径 |
| DTYPE | bfloat16 | 模型精度（float16、bfloat16、float32） |
| DEVICE | cuda:0 | 加载设备 |
| ATTN_IMPLEMENTATION | flash_attention_2 | 注意力后端（flash_attention_2、sdpa、eager） |
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
