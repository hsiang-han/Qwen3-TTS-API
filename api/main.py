import io
import os
import tempfile
import wave
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
DTYPE = os.getenv("DTYPE", "bfloat16")
DEVICE = os.getenv("DEVICE", "cuda:0")
ATTN_IMPL = os.getenv("ATTN_IMPLEMENTATION", "flash_attention_2")

_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    from qwen_tts import Qwen3TTSModel

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    attn_impl = ATTN_IMPL
    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print("WARNING: flash-attn not available, falling back to sdpa")
            attn_impl = "sdpa"

    _model = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map=DEVICE,
        dtype=dtype_map.get(DTYPE, torch.bfloat16),
        attn_implementation=attn_impl,
    )
    yield
    _model = None


app = FastAPI(title="Qwen3-TTS-API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "ok" if _model else "loading",
        "model": MODEL_ID,
        "dtype": DTYPE,
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
    instruct: Optional[str] = None


@app.post("/v1/audio/speech")
async def text_to_speech(req: SpeechRequest):
    if not _model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not req.input.strip():
        raise HTTPException(status_code=400, detail="Input text is empty")

    model_type = _detect_model_type()

    if model_type == "custom_voice":
        wavs, sr = _model.generate_custom_voice(
            text=req.input,
            language=req.language if req.language != "Auto" else None,
            speaker=req.voice,
            instruct=req.instruct if req.instruct else None,
        )
    elif model_type == "voice_design":
        wavs, sr = _model.generate_voice_design(
            text=req.input,
            language=req.language if req.language != "Auto" else None,
            instruct=req.instruct if req.instruct else "",
        )
    elif model_type == "base":
        raise HTTPException(
            status_code=400,
            detail="Base model requires reference audio. Use /v1/audio/speech/clone instead.",
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

    wav_bytes = _to_wav(wavs[0], sr)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/v1/audio/speech/clone")
async def clone_speech(
    input: str = Form(...),
    ref_audio: UploadFile = File(...),
    ref_text: str = Form(default=None),
    language: str = Form(default="Auto"),
    x_vector_only: bool = Form(default=False),
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
            detail="ref_text is required when x_vector_only is false (ICL mode). Provide the transcript of the reference audio, or set x_vector_only=true.",
        )

    audio_bytes = await ref_audio.read()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        wavs, sr = _model.generate_voice_clone(
            text=input,
            language=language if language != "Auto" else None,
            ref_audio=tmp_path,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )
    finally:
        os.unlink(tmp_path)

    wav_bytes = _to_wav(wavs[0], sr)
    return Response(content=wav_bytes, media_type="audio/wav")


def _detect_model_type() -> str:
    if not _model:
        return "unknown"
    tts_type = getattr(_model.model, "tts_model_type", None)
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
    fn = getattr(_model.model, "get_supported_speakers", None)
    if callable(fn):
        result = fn()
        return result if result else []
    return []


def _get_languages() -> list:
    if not _model:
        return []
    fn = getattr(_model.model, "get_supported_languages", None)
    if callable(fn):
        result = fn()
        return result if result else []
    return []


def _to_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()
