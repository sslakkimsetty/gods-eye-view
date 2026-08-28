# voice-service — local STT/TTS for the God's Eye View voice agent

FastAPI service holding two small MLX audio models resident so a voice turn
costs one HTTP round trip each way. The LLM half is a separate
`mlx_lm.server`; this process only does speech.

| Endpoint | In | Out |
|----------|----|-----|
| `GET /health` | — | model + status JSON |
| `POST /stt` | WAV bytes | `{"text": ..., "ms": ...}` |
| `POST /tts` | `{"text": ..., "voice"?, "speed"?}` | `audio/wav` |

## Install

Python 3.10+ on Apple Silicon. `kokoro-mlx` declares 3.10–3.12 but runs fine on
3.14.

```sh
uv venv .venv
uv pip install --python .venv/bin/python \
  parakeet-mlx kokoro-mlx fastapi uvicorn python-multipart soundfile numpy
```

## Run

```sh
export HF_HOME=/path/to/your/hf/cache        # keeps weights off the boot volume
.venv/bin/uvicorn audio_server:app --host 127.0.0.1 --port 8890
```

Startup loads parakeet (~600 MB) and Kokoro (~330 MB) and then synthesizes one
throwaway phrase. That warmup is deliberate: the first Kokoro call costs ~26s of
phonemizer initialization, which inside a user's first utterance reads as a hang.
After warmup, TTS is ~70–200ms and STT ~40ms for a spoken command.

Point the app at it:

```sh
GEV_VOICE_STT_URL=http://127.0.0.1:8890
GEV_VOICE_TTS_URL=http://127.0.0.1:8890
```

## Why it decodes WAV itself

`parakeet_mlx.audio.load_audio()` shells out to `ffmpeg` unconditionally. Rather
than add that dependency, `/stt` decodes WAV with `soundfile` and calls
`get_logmel()` + `model.generate()` directly — the same path `transcribe()` takes
after loading. The browser sends 16 kHz mono PCM WAV, so no resample is needed in
the common case.

## Environment

| Variable | Default |
|----------|---------|
| `GEV_STT_MODEL` | `mlx-community/parakeet-tdt-0.6b-v3` |
| `GEV_TTS_MODEL` | `mlx-community/Kokoro-82M-bf16` |
| `GEV_TTS_VOICE` | `af_heart` |
