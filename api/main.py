import io
import os
import struct
import threading
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice")
DTYPE = os.getenv("DTYPE", "bfloat16")
DEVICE = os.getenv("DEVICE", "cuda:0")
ATTN_IMPL = os.getenv("ATTN_IMPLEMENTATION", "sdpa")

SAMPLE_RATE = 24000
_model = None
_model_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, SAMPLE_RATE
    from faster_qwen3_tts import FasterQwen3TTS

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    _model = FasterQwen3TTS.from_pretrained(
        MODEL_ID,
        device=DEVICE,
        dtype=dtype_map.get(DTYPE, torch.bfloat16),
        attn_implementation=ATTN_IMPL,
    )
    SAMPLE_RATE = _model.sample_rate

    yield
    _model = None


app = FastAPI(title="Qwen3-TTS-API", version="0.2.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok" if _model else "loading",
        "model": MODEL_ID,
        "dtype": DTYPE,
        "attention": ATTN_IMPL,
        "sample_rate": SAMPLE_RATE,
    }


@app.get("/v1/models")
async def list_models():
    model_type = _detect_model_type()
    speakers = _get_speakers()
    languages = _get_languages()
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_ID,
                "object": "model",
                "owned_by": "Qwen",
                "type": model_type,
                "available_voices": speakers,
                "available_languages": languages,
            }
        ],
    }


@app.get("/v1/voices")
async def list_voices():
    speakers = _get_speakers()
    languages = _get_languages()
    return {"voices": speakers, "languages": languages}


class SpeechRequest(BaseModel):
    model: Optional[str] = None
    input: str
    voice: str = "Vivian"
    language: str = "Auto"
    response_format: str = "wav"
    speed: float = 1.0
    stream: bool = False
    instruct: Optional[str] = None


@app.post("/v1/audio/speech")
async def text_to_speech(req: SpeechRequest):
    if not _model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    model_type = _detect_model_type()

    if model_type == "custom_voice":
        if req.stream:
            return StreamingResponse(
                _stream_custom_voice(req.input, req.voice, req.language, req.instruct),
                media_type="audio/wav",
            )
        audio, sr = _generate_custom_voice(req.input, req.voice, req.language, req.instruct)
    elif model_type == "voice_design":
        audio, sr = _generate_voice_design(req.input, req.language, req.instruct or "")
    elif model_type == "base":
        raise HTTPException(
            status_code=400,
            detail="Base model requires reference audio. Use /v1/audio/speech/clone instead.",
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

    if req.response_format == "pcm":
        return Response(content=_to_pcm16(audio), media_type="audio/pcm")
    return Response(content=_to_wav(audio, sr), media_type="audio/wav")


@app.post("/v1/audio/speech/clone")
async def clone_speech(
    input: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(default=None),
    language: str = Form(default="Auto"),
    x_vector_only: bool = Form(default=False),
    stream: bool = Form(default=False),
):
    if not _model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not input.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    model_type = _detect_model_type()
    if model_type != "base":
        raise HTTPException(
            status_code=400,
            detail=f"Voice clone requires a Base model. Current model type: {model_type}",
        )

    if not x_vector_only and not ref_text:
        raise HTTPException(
            status_code=400,
            detail="ref_text is required when x_vector_only is false (ICL mode).",
        )

    import tempfile
    audio_bytes = await ref_audio.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        if stream:
            return StreamingResponse(
                _stream_voice_clone(input, language, tmp_path, ref_text, x_vector_only),
                media_type="audio/wav",
            )
        audio, sr = _generate_voice_clone(input, language, tmp_path, ref_text, x_vector_only)
    finally:
        if not stream:
            os.unlink(tmp_path)

    return Response(content=_to_wav(audio, sr), media_type="audio/wav")


# --- Generation functions ---

def _generate_custom_voice(text: str, speaker: str, language: str, instruct: Optional[str]):
    lang = language if language != "Auto" else "Auto"
    with _model_lock:
        wavs, sr = _model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=lang,
            instruct=instruct if instruct else None,
        )
    return wavs[0], sr


def _stream_custom_voice(text: str, speaker: str, language: str, instruct: Optional[str]):
    lang = language if language != "Auto" else "Auto"
    header_sent = False
    with _model_lock:
        for chunk, sr, timing in _model.generate_custom_voice_streaming(
            text=text,
            speaker=speaker,
            language=lang,
            instruct=instruct if instruct else None,
            chunk_size=8,
        ):
            if not header_sent:
                yield _wav_header(sr)
                header_sent = True
            yield _to_pcm16(chunk)


def _generate_voice_design(text: str, language: str, instruct: str):
    lang = language if language != "Auto" else "Auto"
    with _model_lock:
        wavs, sr = _model.generate_voice_design(
            text=text,
            language=lang,
            instruct=instruct,
        )
    return wavs[0], sr


def _generate_voice_clone(text: str, language: str, ref_audio: str, ref_text: Optional[str], xvec_only: bool):
    lang = language if language != "Auto" else "Auto"
    with _model_lock:
        wavs, sr = _model.generate_voice_clone(
            text=text,
            language=lang,
            ref_audio=ref_audio,
            ref_text=ref_text,
            xvec_only=xvec_only,
        )
    return wavs[0], sr


def _stream_voice_clone(text: str, language: str, ref_audio: str, ref_text: Optional[str], xvec_only: bool):
    lang = language if language != "Auto" else "Auto"
    header_sent = False
    with _model_lock:
        for chunk, sr, timing in _model.generate_voice_clone_streaming(
            text=text,
            language=lang,
            ref_audio=ref_audio,
            ref_text=ref_text or "",
            xvec_only=xvec_only,
            chunk_size=8,
        ):
            if not header_sent:
                yield _wav_header(sr)
                header_sent = True
            yield _to_pcm16(chunk)


# --- Helpers ---

def _detect_model_type() -> str:
    if not _model:
        return "unknown"
    base = _model.model
    tts_type = getattr(base.model, "tts_model_type", None)
    if tts_type:
        return tts_type
    model_id_lower = MODEL_ID.lower()
    if "customvoice" in model_id_lower:
        return "custom_voice"
    elif "voicedesign" in model_id_lower:
        return "voice_design"
    elif "base" in model_id_lower:
        return "base"
    return "unknown"


def _get_speakers() -> list:
    if not _model:
        return []
    base = _model.model
    fn = getattr(base.model, "get_supported_speakers", None)
    if callable(fn):
        result = fn()
        return [str(s) for s in result] if result else []
    return []


def _get_languages() -> list:
    if not _model:
        return []
    base = _model.model
    fn = getattr(base.model, "get_supported_languages", None)
    if callable(fn):
        result = fn()
        return [str(l) for l in result] if result else []
    return []


def _to_pcm16(audio: np.ndarray) -> bytes:
    return np.clip(audio * 32768, -32768, 32767).astype(np.int16).tobytes()


def _wav_header(sample_rate: int, data_len: int = 0xFFFFFFFF) -> bytes:
    n_channels = 1
    bits = 16
    byte_rate = sample_rate * n_channels * bits // 8
    block_align = n_channels * bits // 8
    riff_size = 0xFFFFFFFF if data_len == 0xFFFFFFFF else 36 + data_len
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", riff_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, n_channels, sample_rate, byte_rate, block_align, bits))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_len))
    return buf.getvalue()


def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    raw = _to_pcm16(audio)
    return _wav_header(sample_rate, len(raw)) + raw
