# pi-voice-server

Minimal **Kokoro-82M** TTS HTTP server. Loads one ONNX model on startup, keeps
it warm in memory, and serves synthesized speech over a tiny REST surface.

This is the speech backend for [**agentchatbox**](https://github.com/anunkai1/agentchatbox)
(its `/api/tts` proxies here) and [**kidstories**](https://gitea.mavali.top/admin/kidstories)
(book narration). agentchatbox's `pi-voice-reply` extension decides *what* to
say; this server handles *how* to say it.

Adapted from [`@s1m0n38/pi-voice`](https://github.com/s1m0n38/pi-voice) (MIT).
Rewritten to use [`kokoro-js`](https://www.npmjs.com/package/kokoro-js) directly
— the original `@huggingface/transformers` env flags triggered an
`onnxruntime-node` native-library path-resolution bug on this box.

## Why a separate server?

Kokoro's ONNX model is ~291 MB and takes ~600 ms to cold-load. Keeping it
resident in a long-lived process means synthesis is ~1.5 s/sentence instead
of ~2-3 s (model reload every call). Every consumer (agentchatbox, kidstories)
shares one warm model over HTTP.

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET` | `/health` | — | `{ status, modelLoaded, dtype, voice, voiceCount }` |
| `GET` | `/voices` | — | `{ voices: string[] }` (28 Kokoro voices) |
| `POST` | `/tts` | `{ text, voice?, speed? }` | `audio/wav` — whole blob, all chunks concatenated |
| `POST` | `/tts/stream` | `{ text, voice?, speed? }` | `application/octet-stream` — chunked binary frame stream |

### Streaming (`/tts/stream`)

Long text is split into ≤500-char chunks (Kokoro's phoneme encoder has a fixed
~510-token window — anything past it is silently dropped). The non-streaming
`/tts` synthesizes all chunks then concatenates them into one WAV before
responding. `/tts/stream` instead writes each chunk's WAV **the moment it's
synthesized**, so the client can start playing the first chunk while later
chunks are still being synthesized — turning a ~20 s wait into ~5 s
time-to-first-sound on a long reply.

Binary frame layout (little-endian), written straight to the body:

```
[1 byte type][uint32 LE length N][N bytes payload]
  0x01 DATA → payload is one complete WAV file (one text chunk)
  0x00 END  → no payload; clean end of stream
  0x80 ERR  → payload is a UTF-8 error message
```

Both `/tts` and `/tts/stream` occupy one slot in the same serial queue — Kokoro
is not concurrency-safe on a single model instance, so a stream's chunks never
interleave with a concurrent request. The stream detects client disconnect
(`res.close`) and stops synthesizing remaining chunks.

## Configuration

All via environment variables (defaults shown):

| Var | Default | Meaning |
|---|---|---|
| `KOKORO_MODEL_ID` | `onnx-community/Kokoro-82M-v1.0-ONNX` | Hugging Face model id |
| `KOKORO_DTYPE` | `q4` | ONNX quantization (`q4` \| `q4f16` \| `q8` \| `fp16` \| `fp32`) |
| `KOKORO_VOICE` | `af_heart` | Default voice (warm female narrator; see `/voices` for all 28) |
| `KOKORO_HOST` | `127.0.0.1` | Bind host |
| `KOKORO_PORT` | `8181` | Bind port |

Model files are cached by `kokoro-js` under `~/.cache/huggingface/transformers/`
(XDG: `$XDG_CACHE_HOME`).

## Run

```bash
git clone https://github.com/anunkai1/pi-voice-server
cd pi-voice-server
npm install      # model auto-downloads on first boot (~291 MB)
npm start        # → http://127.0.0.1:8181
```

Quick smoke test:

```bash
curl -X POST http://127.0.0.1:8181/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world."}' \
  --output out.wav
```

## Run as a systemd service

A unit file is included (`pi-voice-server.service`) and mirrored into the
deploy manifest at [`infra/systemd/pi-voice-server.service`](https://gitea.mavali.top/admin/infra).

```bash
sudo cp pi-voice-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now pi-voice-server
```

## Files

- **`server.mjs`** — the server systemd actually runs. This is the source of truth.
- `server.ts` — an older, undeployed TypeScript rewrite with extra model-management
  endpoints (`/models/*`). Kept for reference; the systemd unit does **not** run it.
- `test-kokoro.mjs` — throwaway synthesis smoke test.

## Related

- [agentchatbox](https://github.com/anunkai1/agentchatbox) — the chat UI that consumes this (`/api/tts` → here)
- [pi-voice-reply](https://github.com/anunkai1/pi-voice-reply) — pi extension producing spoken-summary voice replies
- [`@s1m0n38/pi-voice`](https://github.com/s1m0n38/pi-voice) — upstream this was adapted from (MIT)
- [`kokoro-js`](https://www.npmjs.com/package/kokoro-js) — the Kokoro ONNX bindings

## License

MIT — see [LICENSE](LICENSE). Adapted from `@s1m0n38/pi-voice` (MIT).
