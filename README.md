# pi-voice-server

Single CPU Python Kokoro 82M service on Server2 loopback 8181. Replaces the
previous kokoro-js/q4 backend without changing existing consumers. The service
and socket names remain `pi-voice-server.{service,socket}`; the owning deployer
remains `deploy-pi-voice-server` (tracked in `infra`). No second voice service.

## HTTP compatibility

- `GET /health`: status, modelAvailable, modelLoaded/modelResident, modelLoading,
  dtype (`fp32`), voice, voiceCount, maxTextChars, maxAudioBytes,
  idleTimeoutMs, activeSyntheses; additionally advertises alignment capability.
  An unloaded model is available, not unhealthy.
- `GET /voices`: `{ "voices": [...] }`, the same ordered 28 English voice names,
  with `af_heart` the default. Both US and British voices use the corresponding
  upstream English phonemiser. Names/voices remain; pronunciation need not be
  sample-identical to the former quantised backend.
- `POST /tts`: JSON `{text, voice?, speed?}` → mono PCM16, 24 kHz WAV.
- `POST /tts/stream`: identical JSON → `application/octet-stream`.
  Frames are **one byte type + uint32 LE payload byte length + payload**:
  DATA=1 carries a complete WAV chunk; END=0 has no payload; ERROR=128 carries a
  UTF-8 error message and is terminal (not followed by a false successful END).

Speed range is 0.5–2, default 1. Text is trimmed and limited to 30,000 characters;
request bodies are limited to 128 KiB and output to 64 MiB by default. Errors are
JSON before streaming headers, ERROR frames afterwards. Requests are serialised
across both synthesis routes. Disconnects abandon queued/future chunks; native
inference already executing finishes before another job uses the model.

Text is split at sentence/line boundaries with a ramped chunk budget; official
KPipeline token-aware splitting handles long sentences. Over-context individual
tokens are split before inference, never silently truncated.

## Optional synthesis alignment

Only `/tts/stream`, **only** when the request includes:

```
X-Kokoro-Alignment: kokoro-pred-dur-v1
```

The server echoes that header and emits a type **2 JSON** metadata frame before
each type 1 WAV. Unknown alignment versions are rejected (400). Without this
header the old stream has exactly its old frame types; no client migration is
required. `/tts` remains WAV-only and rejects alignment negotiation.

Metadata includes `submittedText` (exact trimmed synthesis input, not chat
Markdown), chunk `text`, `sampleRate:24000`, chunk start/end sample offsets, and
`words:[{text,whitespace,startSample,endSample}]`. Offsets are absolute within the
concatenated request audio, **before** WAV headers, including every prior chunk.
Punctuation may be a separate token. `unalignedTokens` and `alignmentComplete`
expose missing token mappings rather than inventing timings.

Timings come directly from official `KPipeline.infer` → `output.pred_dur` →
`KPipeline.join_timestamps`; timestamp seconds are converted to 24 kHz sample
indices. No ASR or proportional division. The upstream algorithm allocates
phoneme/space duration and a BOS offset; it is synthesis alignment, not a
measurement of pronunciation or proof anybody heard the sound. A player must
retain only tokens whose endSample passed its actual output clock, and preserve
failure/truncation/zero-play uncertainty.

Example from real af_heart inference of `Hello world.` at speed 1:
`Hello: [8400,15000]`, `world: [15000,31200]`, `.: [31200,35400]`, in a
37,800-sample waveform. These are returned model-duration alignments, not evenly
spaced guesses. Different model/runtime versions may produce different values.

This repository exposes alignment; it does **not** change Relay engine selection
or its played-prefix recording. That browser/extension integration is separate.

## Runtime and deployment

Python 3.12; CPU-only Torch and dependencies are pinned in `uv.lock`. `prepare_model.py`
stages the model config/weights and all 28 voice packs at a pinned Hugging Face
revision into `model/`. Runtime inference uses those files; the service sets
`HF_HUB_OFFLINE=1`. Staging tests synthesis before the immutable release switch.
No model download is permitted/required by the sandboxed live process.

```
uv sync --frozen
.venv/bin/python prepare_model.py
.venv/bin/python -m unittest -v
KOKORO_PORT=18181 HF_HUB_OFFLINE=1 .venv/bin/python server.py
.venv/bin/python verify_contract.py http://127.0.0.1:18181
.venv/bin/python verify_lifecycle.py
```

The runtime honours systemd's inherited `tts` FD (loopback only); outside
systemd it binds KOKORO_HOST/KOKORO_PORT, default 127.0.0.1:8181. It lazily loads
one model, defaults to four Torch CPU threads (`KOKORO_THREADS`), and exits
successfully after ten minutes without synthesis. Health checks do not load
weights or extend the idle lifetime. `KOKORO_IDLE_TIMEOUT_MS` permits 60,000 to
86,400,000; `KOKORO_VOICE` and `KOKORO_MAX_AUDIO_BYTES` remain supported.
`KOKORO_MODEL_ID`/`KOKORO_DTYPE` are intentionally retired: Python uses the pinned
82M fp32 model, not a different ONNX model or dtype.

Deployment: `deploy-pi-voice-server`. It fetches committed GitHub main, builds an
immutable v3 release, validates dependencies/tests and real synthesis, and uses
the existing transactional release helpers. It restarts **only** the voice
service/socket. ACB, KidStories, KidStories2 and Lappy tunnels do not change.

**Rollback:** `deploy-pi-voice-server --rollback-js` restores the preserved v2
immutable kokoro-js release **and its matching service unit**, under the same
socket. It refuses a busy/unhealthy current service, like normal deployment.
The first cutover saves `rollback-js` and `rollback-js.service` under the release
root; these are intentionally not pruned during this trial. No JS process runs
alongside Python. Reapply normal deployment to return to Python. Discarding the
on-disk rollback artefacts is a later owner decision, not part of the trial.

## Measurements

See `benchmarks/2026-11-15/README.md` and raw JSON. Python is faster warm on the
measured texts but slower cold and uses more memory/disk. These are Server2
measurements, not a guarantee about perceived conversational latency.

## Licence

MIT server wrapper; Kokoro, its model, English phonemisation and Torch retain
their upstream licences. Original server adapted from @s1m0n38/pi-voice.
