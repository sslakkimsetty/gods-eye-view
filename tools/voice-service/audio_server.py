"""
audio_server.py — STT/TTS for the God's Eye View local voice agent.

Runs on Magi-02:8890, reachable from a client through the 11462 tunnel. Holds
the two small audio models resident so a voice turn costs one HTTP round trip
each way; the LLM half lives separately on 8881.

  GET  /health  -> model + device status
  POST /stt     -> WAV bytes in, {"text": ...} out
  POST /tts     -> {"text": ...} in, WAV bytes out

Both models load at startup rather than on first request: lazy loading would
put a multi-second stall inside the first user utterance, which reads as a bug.
"""

from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager

import mlx.core as mx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

STT_MODEL_ID = os.environ.get("GEV_STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v3")
TTS_MODEL_ID = os.environ.get("GEV_TTS_MODEL", "mlx-community/Kokoro-82M-bf16")
DEFAULT_VOICE = os.environ.get("GEV_TTS_VOICE", "af_heart")

# A spoken command is seconds long. Anything past this is a client bug or an
# unbounded upload, not speech worth transcribing.
MAX_AUDIO_BYTES = 10 * 1024 * 1024
MAX_TTS_CHARS = 2000

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("gev-audio")

_models: dict[str, object] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    from kokoro_mlx import KokoroTTS
    from parakeet_mlx import from_pretrained

    started = time.time()
    log.info("loading STT %s", STT_MODEL_ID)
    _models["stt"] = from_pretrained(STT_MODEL_ID)
    log.info("loading TTS %s", TTS_MODEL_ID)
    _models["tts"] = KokoroTTS.from_pretrained(TTS_MODEL_ID)
    # First synthesis pays ~26s of phonemizer/voice warmup. Spending it here
    # keeps it out of the user's first utterance, where it reads as a hang.
    warm_started = time.time()
    _models["tts"].generate("Ready.", voice=DEFAULT_VOICE)
    log.info("TTS warmed in %.1fs", time.time() - warm_started)
    log.info("models ready in %.1fs", time.time() - started)
    yield
    _models.clear()


app = FastAPI(title="GEV audio", lifespan=lifespan)


class TtsRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TTS_CHARS)
    voice: str = DEFAULT_VOICE
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok" if {"stt", "tts"} <= _models.keys() else "loading",
            "stt_model": STT_MODEL_ID,
            "tts_model": TTS_MODEL_ID,
            "voice": DEFAULT_VOICE,
        }
    )


def decode_wav_to_mono(audio_bytes: bytes, target_rate: int) -> np.ndarray:
    """
    Decode WAV bytes to mono float32 at `target_rate`.

    parakeet-mlx's own load_audio() shells out to ffmpeg unconditionally, which
    is not installed on this machine and is not worth adding for a format the
    client already controls. The browser sends PCM WAV, so soundfile decodes it
    directly and this stays a pure-Python path.
    """
    samples, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=True)
    mono = samples.mean(axis=1)
    if sample_rate != target_rate:
        # Rare: the client is expected to record at the model's rate. librosa
        # ships with parakeet-mlx, so this costs no extra dependency.
        import librosa

        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=target_rate)
    return np.ascontiguousarray(mono, dtype=np.float32)


@app.post("/stt")
async def stt(request: Request) -> JSONResponse:
    """Transcribe WAV bytes posted as the raw request body."""
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=400, detail="empty request body")
    if len(audio) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="audio too large")
    model = _models.get("stt")
    if model is None:
        raise HTTPException(status_code=503, detail="STT model still loading")

    started = time.time()
    try:
        from parakeet_mlx.audio import get_logmel

        config = model.preprocessor_config
        waveform = decode_wav_to_mono(audio, config.sample_rate)
        if waveform.size < config.hop_length:
            return JSONResponse({"text": "", "ms": 0, "note": "audio shorter than one frame"})
        # No chunking: chunk_duration windows drop speech at the seams, and a
        # spoken command is short enough to transcribe in a single pass.
        mel = get_logmel(mx.array(waveform), config)
        result = model.generate(mel)[0]
    except HTTPException:
        raise
    except Exception as error:  # noqa: BLE001 - surface a clean 502 to the proxy
        log.exception("transcription failed")
        raise HTTPException(status_code=502, detail=f"transcription failed: {error}") from error

    text = (getattr(result, "text", "") or "").strip()
    elapsed_ms = int((time.time() - started) * 1000)
    log.info("stt %d bytes -> %r (%dms)", len(audio), text[:80], elapsed_ms)
    return JSONResponse({"text": text, "ms": elapsed_ms})


@app.post("/tts")
def tts(payload: TtsRequest) -> Response:
    """Synthesize speech and return a WAV the browser can play directly."""
    model = _models.get("tts")
    if model is None:
        raise HTTPException(status_code=503, detail="TTS model still loading")

    started = time.time()
    try:
        result = model.generate(payload.text, voice=payload.voice, speed=payload.speed)
    except Exception as error:  # noqa: BLE001
        log.exception("synthesis failed")
        raise HTTPException(status_code=502, detail=f"synthesis failed: {error}") from error

    samples = getattr(result, "audio", result)
    sample_rate = int(getattr(result, "sample_rate", 24000))
    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
    data = buffer.getvalue()
    elapsed_ms = int((time.time() - started) * 1000)
    log.info("tts %r -> %d bytes (%dms)", payload.text[:60], len(data), elapsed_ms)
    return Response(
        content=data,
        media_type="audio/wav",
        headers={"X-Synthesis-Ms": str(elapsed_ms), "Cache-Control": "no-store"},
    )
