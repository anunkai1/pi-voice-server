#!/usr/bin/env python3
"""Single CPU Kokoro backend; compatibility WAV/framed HTTP, opt-in alignment."""
import asyncio
import io
import json
import math
import os
import re
import signal
import socket
import struct
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from aiohttp import web

VOICES = "af_heart af_alloy af_aoede af_bella af_jessica af_kore af_nicole af_nova af_river af_sarah af_sky am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa bf_emma bf_isabella bm_george bm_lewis bf_alice bf_lily bm_daniel bm_fable".split()
RATE = 24000
ALIGNMENT = "kokoro-pred-dur-v1"


class UnpronounceableText(ValueError):
    """Input cannot be represented by the speech model without dropping text."""


@dataclass
class Config:
    voice: str = os.getenv("KOKORO_VOICE", "af_heart")
    max_audio: int = max(1048576, int(os.getenv("KOKORO_MAX_AUDIO_BYTES", 67108864)))
    idle_ms: int = int(os.getenv("KOKORO_IDLE_TIMEOUT_MS", 600000))


def validate(body, config):
    if not isinstance(body, dict) or not isinstance(body.get("text"), str):
        raise web.HTTPBadRequest(text=json.dumps({"error": "'text' must be a string"}), content_type="application/json")
    text = body["text"].strip()
    if not text:
        raise web.HTTPBadRequest(text='{"error":"text must not be empty"}', content_type="application/json")
    text_length = len(text.encode("utf-16-le", errors="surrogatepass")) // 2
    if text_length > 30000:
        raise web.HTTPRequestEntityTooLarge(max_size=30000, actual_size=text_length)
    voice = body.get("voice", config.voice)
    if voice not in VOICES:
        raise web.HTTPBadRequest(text='{"error":"unknown voice"}', content_type="application/json")
    try:
        speed = float(body.get("speed", 1))
    except (ValueError, TypeError):
        speed = float("nan")
    if not math.isfinite(speed) or not 0.5 <= speed <= 2:
        raise web.HTTPBadRequest(text='{"error":"speed must be a number from 0.5 to 2"}', content_type="application/json")
    return text, voice, speed


def chunks(text):
    # Same sentence/line ramp as the former lib/chunker.mjs; no mid-sentence
    # cadence break unless the model phoneme context requires one.
    units = []
    for line in text.replace("\r\n", "\n").split("\n"):
        pending = ""
        for piece in re.split(r'(?<=[.!?])\s+', line.strip()):
            pending = f"{pending} {piece}".strip()
            if pending and not re.search(r'(?:^|\s)(?:e\.g|i\.e|etc|vs|approx|Mr|Mrs|Ms|Dr|St|No)\.$', pending, re.I):
                units.append(pending)
                pending = ""
        if pending:
            units.append(pending)
    result, buf, budget = [], "", 0
    for unit in units:
        if buf and (budget == 0 or len(buf) + 1 + len(unit) > budget):
            result.append(buf)
            budget = min(500, math.ceil(len(buf) * 2.2))
            buf = unit
        else:
            buf = f"{buf} {unit}".strip()
    if buf:
        result.append(buf)
    return result


def wav_bytes(pcm):
    out = io.BytesIO()
    with wave.open(out, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(pcm)
    return out.getvalue()


def frame(kind, payload=b""):
    return struct.pack("<BI", kind, len(payload)) + payload


class KokoroBackend:
    def __init__(self):
        self.model = None
        self.pipelines = {}
        self.loading = False

    def load(self):
        if self.model is not None:
            return
        self.loading = True
        try:
            import torch
            from kokoro import KModel, KPipeline
            torch.set_num_threads(int(os.getenv("KOKORO_THREADS", "4")))
            directory = Path(__file__).parent / "model"
            options = {"config": str(directory / "config.json"), "model": str(directory / "kokoro-v1_0.pth")} if directory.is_dir() else {}
            model = KModel(repo_id="hexgrad/Kokoro-82M", **options).to("cpu").eval()
            pipelines = {lang: KPipeline(lang_code=lang, repo_id="hexgrad/Kokoro-82M", model=model) for lang in "ab"}
            self.model, self.pipelines = model, pipelines
        finally:
            self.loading = False

    def prepare(self, text, voice):
        self.load()
        pipeline = self.pipelines[voice[0]]
        prepared = []
        pending = chunks(text)
        while pending:
            chunk = pending.pop(0)
            _, tokens = pipeline.g2p(chunk)
            if any(len(token.phonemes or "") > 510 for token in tokens):
                # Match the former server's hard-split safety net for an
                # over-context URL/word. Never let upstream truncate phonemes.
                if len(chunk) < 2:
                    raise ValueError("text cannot fit Kokoro's phoneme context")
                boundaries = [m.start() for m in re.finditer(r"\s+", chunk) if 0 < m.start() < len(chunk)]
                middle = min(boundaries, key=lambda i: abs(i - len(chunk) // 2)) if boundaries else len(chunk) // 2
                pending[0:0] = [chunk[:middle].strip(), chunk[middle:].strip()]
                continue
            tokenized = list(pipeline.en_tokenize(tokens))
            if not tokenized:
                raise UnpronounceableText("text contains a chunk with no pronounceable content")
            for gs, ps, tks in tokenized:
                if not ps.strip():
                    raise UnpronounceableText("text contains a chunk with no pronounceable content")
                if len(ps) <= 510:
                    prepared.append((gs, ps, tks))
                    continue
                # Upstream's punctuation waterfall can leave a remainder
                # over its own 510-phoneme limit. Repack at token boundaries
                # rather than rejecting a valid sentence or cutting a word.
                group = []
                for token in tks:
                    if group and len(pipeline.tokens_to_ps([*group, token])) > 510:
                        prepared.append((pipeline.tokens_to_text(group), pipeline.tokens_to_ps(group), group))
                        group = []
                    group.append(token)
                if group and pipeline.tokens_to_ps(group):
                    prepared.append((pipeline.tokens_to_text(group), pipeline.tokens_to_ps(group), group))
        return prepared

    def synthesize(self, item, voice, speed, offset):
        import numpy as np
        from kokoro import KPipeline
        gs, ps, tokens = item
        pipeline = self.pipelines[voice[0]]
        local_voice = Path(__file__).parent / "model" / "voices" / f"{voice}.pt"
        pack = pipeline.load_voice(str(local_voice) if local_voice.is_file() else voice)
        output = KPipeline.infer(self.model, ps, pack, speed)
        KPipeline.join_timestamps(tokens, output.pred_dur)
        samples = output.audio.detach().cpu().numpy().reshape(-1)
        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
        words = []
        for token in tokens:
            start, end = getattr(token, "start_ts", None), getattr(token, "end_ts", None)
            if start is not None and end is not None:
                words.append({"text": token.text, "whitespace": token.whitespace,
                              "startSample": offset + round(start * RATE),
                              "endSample": offset + round(end * RATE)})
        unaligned = [token.text for token in tokens if any(c.isalnum() for c in token.text)
                     and (getattr(token, "start_ts", None) is None or getattr(token, "end_ts", None) is None)]
        return pcm, {"text": gs, "words": words, "unalignedTokens": unaligned,
                     "alignmentComplete": not unaligned, "startSample": offset,
                     "endSample": offset + len(samples), "sampleRate": RATE,
                     "alignment": ALIGNMENT}


def create_app(backend=None, config=None):
    backend, config = backend or KokoroBackend(), config or Config()
    if config.voice not in VOICES or not 60000 <= config.idle_ms <= 86400000:
        raise ValueError("invalid default voice or idle timeout")
    @web.middleware
    async def json_errors(request, handler):
        try:
            return await handler(request)
        except web.HTTPException as error:
            if error.content_type == "application/json":
                raise
            return web.json_response({"error": error.reason}, status=error.status)

    app = web.Application(client_max_size=131072, middlewares=[json_errors])
    lock = asyncio.Lock()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kokoro")
    state = {"active": 0, "last": time.monotonic(), "draining": False}

    async def cpu(fn, *args):
        # A disconnected HTTP handler must not release the serial slot while
        # native CPU inference is still running. Stop before the NEXT chunk.
        future = asyncio.get_running_loop().run_in_executor(executor, fn, *args)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            try:
                await future
            finally:
                raise

    async def health(_):
        return web.json_response({"status": "draining" if state["draining"] else "ok",
                                  "modelAvailable": True, "modelLoaded": backend.model is not None,
                                  "modelResident": backend.model is not None, "modelLoading": backend.loading,
                                  "dtype": "fp32", "voice": config.voice, "voiceCount": len(VOICES),
                                  "maxTextChars": 30000, "maxAudioBytes": config.max_audio,
                                  "idleTimeoutMs": config.idle_ms, "activeSyntheses": state["active"],
                                  "alignment": ALIGNMENT})

    async def voices(_):
        return web.json_response({"voices": VOICES})

    async def tts(request):
        if state["draining"]:
            raise web.HTTPServiceUnavailable(text='{"error":"server is draining"}', content_type="application/json")
        try:
            body = await request.json()
        except (ValueError, UnicodeDecodeError):
            raise web.HTTPBadRequest(text='{"error":"invalid JSON"}', content_type="application/json")
        text, voice, speed = validate(body, config)
        aligned = request.headers.get("X-Kokoro-Alignment") == ALIGNMENT
        streaming = request.path == "/tts/stream"
        if aligned and not streaming:
            raise web.HTTPBadRequest(text='{"error":"alignment requires /tts/stream"}', content_type="application/json")
        if request.headers.get("X-Kokoro-Alignment") and not aligned:
            raise web.HTTPBadRequest(text='{"error":"unsupported alignment protocol"}', content_type="application/json")
        response = web.StreamResponse(headers={"Content-Type": "application/octet-stream", "Cache-Control": "no-store"})
        if aligned:
            response.headers["X-Kokoro-Alignment"] = ALIGNMENT
        if streaming:
            await response.prepare(request)
        state["active"] += 1
        try:
            async with lock:
                def disconnected():
                    return request.transport is None or request.transport.is_closing()
                if disconnected():
                    return response
                if state["draining"]:
                    raise web.HTTPServiceUnavailable(text='{"error":"server is draining"}', content_type="application/json")
                items = await cpu(backend.prepare, text, voice)
                if not items:
                    raise UnpronounceableText("text contains no pronounceable content")
                audio, offset, total = [], 0, 0 if streaming else 44
                for item in items:
                    if disconnected():
                        break
                    pcm, metadata = await cpu(backend.synthesize, item, voice, speed, offset)
                    if disconnected():
                        break
                    total += len(pcm) + (44 if streaming else 0)
                    if total > config.max_audio:
                        raise web.HTTPRequestEntityTooLarge(max_size=config.max_audio, actual_size=total)
                    if streaming:
                        if aligned:
                            metadata["submittedText"] = text
                            await response.write(frame(2, json.dumps(metadata, ensure_ascii=False).encode()))
                        await response.write(frame(1, wav_bytes(pcm)))
                    else:
                        audio.append(pcm)
                    offset += len(pcm) // 2
                if streaming:
                    if not disconnected():
                        await response.write(frame(0))
                        await response.write_eof()
                    return response
                return web.Response(body=wav_bytes(b"".join(audio)), content_type="audio/wav", headers={"Cache-Control": "no-store"})
        except (ConnectionError, asyncio.CancelledError):
            raise
        except Exception as error:
            if streaming:
                try:
                    await response.write(frame(128, str(error).encode()))
                    await response.write_eof()
                except ConnectionError:
                    pass
                return response
            status = error.status if isinstance(error, web.HTTPException) else 400 if isinstance(error, UnpronounceableText) else 500
            return web.json_response({"error": str(error)}, status=status)
        finally:
            state["active"] -= 1
            state["last"] = time.monotonic()

    async def lifecycle(_):
        async def idle_exit():
            while True:
                await asyncio.sleep(1)
                if not state["active"] and (time.monotonic() - state["last"]) * 1000 >= config.idle_ms:
                    os.kill(os.getpid(), signal.SIGTERM)
                    return
        task = asyncio.create_task(idle_exit())
        yield
        state["draining"] = True
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        executor.shutdown(wait=True)

    async def start_draining(_):
        state["draining"] = True

    app.on_shutdown.append(start_draining)
    app.cleanup_ctx.append(lifecycle)
    app.router.add_get("/health", health)
    app.router.add_get("/voices", voices)
    app.router.add_post("/tts", tts)
    app.router.add_post("/tts/stream", tts)
    return app


def listen_socket(env=os.environ):
    pid, fds = env.get("LISTEN_PID"), env.get("LISTEN_FDS")
    if pid is None and fds is None:
        return None
    # systemd activation variables belong only to the nominated process.
    if pid != str(os.getpid()):
        return None
    if fds != "1":
        raise ValueError("invalid systemd socket activation environment")
    if env.get("LISTEN_FDNAMES", "tts") != "tts":
        raise ValueError("unexpected systemd socket name")
    sock = socket.socket(fileno=3)
    if sock.getsockname()[0] != "127.0.0.1":
        raise ValueError("socket must be loopback-only")
    return sock


if __name__ == "__main__":
    inherited = listen_socket()
    web.run_app(create_app(), sock=inherited,
                host=None if inherited else os.getenv("KOKORO_HOST", "127.0.0.1"),
                port=None if inherited else int(os.getenv("KOKORO_PORT", "8181")),
                handler_cancellation=True, shutdown_timeout=600, access_log=None)
