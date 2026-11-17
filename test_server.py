import asyncio
import io
import json
import struct
import unittest
import wave
import os
from types import SimpleNamespace
from unittest.mock import patch

from aiohttp.test_utils import TestClient, TestServer
from server import ALIGNMENT, Config, VOICES, KokoroBackend, UnpronounceableText, chunks, create_app, frame, listen_socket, validate, wav_bytes


class FakeBackend:
    model = None
    loading = False

    def prepare(self, text, voice):
        self.model = object()
        return chunks(text)

    def synthesize(self, item, voice, speed, offset):
        if item == "Fail.":
            raise ValueError("synthesis failed")
        return b"\0\0" * 240, {"text": item, "words": [{"text": item, "startSample": offset, "endSample": offset + 240}], "sampleRate": 24000, "alignment": ALIGNMENT}


def frames(data):
    result = []
    while data:
        kind, length = struct.unpack("<BI", data[:5])
        if len(data) < 5 + length:
            raise ValueError("truncated frame")
        result.append((kind, data[5:5 + length]))
        data = data[5 + length:]
    return result


class ContractTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = TestClient(TestServer(create_app(FakeBackend())))
        await self.client.start_server()

    async def asyncTearDown(self):
        await self.client.close()

    async def test_health_and_catalogue_unloaded(self):
        health = await (await self.client.get("/health")).json()
        self.assertTrue(health["modelAvailable"])
        self.assertFalse(health["modelResident"])
        self.assertEqual(health["activeSyntheses"], 0)
        self.assertEqual(health["voiceCount"], 28)
        self.assertEqual((await (await self.client.get("/voices")).json())["voices"], VOICES)

    async def test_whole_wav_all_voices_and_speed_bounds(self):
        for voice in VOICES:
            for speed in [.5, 2]:
                response = await self.client.post("/tts", json={"text": "Hello.", "voice": voice, "speed": speed})
                self.assertEqual(response.status, 200)
                self.assertEqual(response.headers["Content-Type"], "audio/wav")
                with wave.open(io.BytesIO(await response.read())) as wav:
                    self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 24000))

    async def test_old_stream_has_no_alignment_frames(self):
        response = await self.client.post("/tts/stream", json={"text": "Hello. There."})
        self.assertEqual([kind for kind, _ in frames(await response.read())], [1, 1, 0])
        self.assertNotIn("X-Kokoro-Alignment", response.headers)

    async def test_opt_in_metadata_carries_exact_submission(self):
        text = "Hello. There."
        response = await self.client.post("/tts/stream", json={"text": text}, headers={"X-Kokoro-Alignment": ALIGNMENT})
        result = frames(await response.read())
        self.assertEqual([kind for kind, _ in result], [2, 1, 2, 1, 0])
        first, second = json.loads(result[0][1]), json.loads(result[2][1])
        self.assertEqual(first["submittedText"], text)
        self.assertEqual(first["words"][0]["endSample"], 240)
        self.assertEqual(second["words"][0]["startSample"], 240)
        self.assertEqual(response.headers["X-Kokoro-Alignment"], ALIGNMENT)

    async def test_failure_frame_not_false_end(self):
        response = await self.client.post("/tts/stream", json={"text": "Fail."})
        result = frames(await response.read())
        self.assertEqual(result, [(128, b"synthesis failed")])

    async def test_no_pronounceable_items_are_an_error_not_empty_audio(self):
        backend = FakeBackend()
        backend.prepare = lambda *_: []
        async with TestClient(TestServer(create_app(backend))) as client:
            response = await client.post("/tts", json={"text": "---"})
            self.assertEqual(response.status, 400)
            self.assertIn("pronounceable", (await response.json())["error"])
            response = await client.post("/tts/stream", json={"text": "---"})
            result = frames(await response.read())
            self.assertEqual([kind for kind, _ in result], [128])

    async def test_shutdown_marks_draining_before_cleanup(self):
        await self.client.server.app.shutdown()
        health = await (await self.client.get("/health")).json()
        self.assertEqual(health["status"], "draining")
        response = await self.client.post("/tts", json={"text": "Hello."})
        self.assertEqual(response.status, 503)

    async def test_invalid_inputs(self):
        cases = [{}, {"text": ""}, {"text": "x", "voice": "no"}, {"text": "x", "speed": .49}, {"text": "x", "speed": 2.01}, {"text": "x", "speed": "NaN"}]
        for body in cases:
            self.assertEqual((await self.client.post("/tts", json=body)).status, 400)
        self.assertEqual((await self.client.post("/tts", json={"text": "x" * 30001})).status, 413)
        self.assertEqual((await self.client.post("/tts", data="{")).status, 400)
        self.assertEqual((await self.client.post("/tts/stream", json={"text": "x"}, headers={"X-Kokoro-Alignment": "unknown"})).status, 400)

    async def test_audio_limits_whole_and_stream(self):
        async with TestClient(TestServer(create_app(FakeBackend(), Config(max_audio=500)))) as client:
            response = await client.post("/tts", json={"text": "Hello."})
            self.assertEqual(response.status, 413)
            response = await client.post("/tts/stream", json={"text": "Hello."})
            self.assertEqual(frames(await response.read())[0][0], 128)

    async def test_utf16_text_limit_and_body_limit(self):
        self.assertEqual((await self.client.post("/tts", json={"text": "😀" * 15001})).status, 413)
        self.assertEqual((await self.client.post("/tts", data=b" " * 131073)).status, 413)

    async def test_serialization(self):
        class SerialBackend(FakeBackend):
            in_flight = 0
            maximum = 0

            def synthesize(self, *args):
                import time
                self.in_flight += 1
                self.maximum = max(self.maximum, self.in_flight)
                time.sleep(.03)
                result = super().synthesize(*args)
                self.in_flight -= 1
                return result
        backend = SerialBackend()
        async with TestClient(TestServer(create_app(backend))) as client:
            responses = await asyncio.gather(*(client.post("/tts", json={"text": "Hello."}) for _ in range(3)))
            self.assertEqual([r.status for r in responses], [200] * 3)
            self.assertEqual(backend.maximum, 1)

    async def test_disconnect_stops_future_chunks_without_overlap(self):
        class SlowBackend(FakeBackend):
            calls = 0

            def synthesize(self, *args):
                import time
                self.calls += 1
                time.sleep(.1)
                return super().synthesize(*args)
        backend = SlowBackend()
        async with TestClient(TestServer(create_app(backend), handler_cancellation=True)) as client:
            response = await client.post("/tts/stream", json={"text": "One. Two. Three. Four."})
            await response.content.readexactly(5)
            response.close()
            await asyncio.sleep(.4)
            self.assertLess(backend.calls, 3)
            self.assertEqual((await (await client.get("/health")).json())["activeSyntheses"], 0)


class BackendTests(unittest.TestCase):
    def pipeline(self, tokens):
        from kokoro import KPipeline
        pipeline = object.__new__(KPipeline)
        pipeline.g2p = lambda _: ("", tokens)
        backend = KokoroBackend()
        backend.model = object()  # No weights or G2P model loaded.
        backend.pipelines = {"a": pipeline}
        return backend

    def test_real_tokenizer_overshoot_is_repacked_without_losing_words(self):
        specifications = [(161, False), (150, False), (161, False), (1, True),
                          (1, False), (150, False), (2, False), (1, True),
                          (1, True), (100, False), (250, False), (161, False)]
        tokens = [SimpleNamespace(text=f"word{i}", whitespace=" ", phonemes="," if punct else "a" * length)
                  for i, (length, punct) in enumerate(specifications)]
        backend = self.pipeline(tokens)
        self.assertTrue(any(len(ps) > 510 for _, ps, _ in backend.pipelines["a"].en_tokenize(tokens)))
        items = backend.prepare("Sentence.", "af_heart")
        self.assertTrue(all(0 < len(ps) <= 510 for _, ps, _ in items))
        self.assertEqual([t.text for _, _, group in items for t in group], [t.text for t in tokens])

    def test_phoneme_empty_chunks_are_rejected_including_mixed_input(self):
        backend = self.pipeline([])
        with self.assertRaises(UnpronounceableText):
            backend.prepare("---", "af_heart")
        pipeline = backend.pipelines["a"]
        pipeline.g2p = lambda text: ("", [] if text == "---" else [SimpleNamespace(text="Hello.", whitespace="", phonemes="hello")])
        with self.assertRaises(UnpronounceableText):
            backend.prepare("Hello.\n---", "af_heart")

    def test_oversized_token_prefers_whitespace_before_character_splitting(self):
        backend = self.pipeline([])
        seen = []
        def g2p(text):
            seen.append(text)
            # Force one oversized pass, then permit the smaller chunks.
            return "", [SimpleNamespace(text=text, whitespace="", phonemes="x" * (511 if len(seen) == 1 else 5))]
        backend.pipelines["a"].g2p = g2p
        backend.prepare("alpha extraordinaryword beta", "af_heart")
        self.assertEqual(seen[1:], ["alpha", "extraordinaryword beta"])

    def test_synthesis_metadata_offsets_and_missing_tokens(self):
        import numpy as np
        from kokoro import KPipeline
        tokens = [SimpleNamespace(text="Hello", whitespace=" ", start_ts=0, end_ts=.01),
                  SimpleNamespace(text="missing", whitespace="", start_ts=None, end_ts=None)]
        backend = self.pipeline(tokens)
        backend.pipelines["a"].load_voice = lambda _: object()
        audio = SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: np.zeros(240))))
        output = SimpleNamespace(audio=audio, pred_dur=object())
        with patch.object(KPipeline, "infer", return_value=output), patch.object(KPipeline, "join_timestamps"):
            pcm, metadata = backend.synthesize(("Hello missing", "hello", tokens), "af_heart", 1, 1000)
        self.assertEqual(len(pcm), 480)
        self.assertEqual((metadata["startSample"], metadata["endSample"]), (1000, 1240))
        self.assertEqual((metadata["words"][0]["startSample"], metadata["words"][0]["endSample"]), (1000, 1240))
        self.assertFalse(metadata["alignmentComplete"])
        self.assertEqual(metadata["unalignedTokens"], ["missing"])


class PureTests(unittest.TestCase):
    def test_socket_activation_ignores_other_process_and_validates_our_descriptor(self):
        self.assertIsNone(listen_socket({}))
        self.assertIsNone(listen_socket({"LISTEN_PID": "99999999", "LISTEN_FDS": "1"}))
        ours = {"LISTEN_PID": str(os.getpid()), "LISTEN_FDS": "1", "LISTEN_FDNAMES": "tts"}
        with self.assertRaises(ValueError):
            listen_socket({**ours, "LISTEN_FDS": "2"})
        with self.assertRaises(ValueError):
            listen_socket({**ours, "LISTEN_FDNAMES": "wrong"})
        with patch("server.socket.socket") as constructor:
            constructor.return_value.getsockname.return_value = ("127.0.0.1", 8181)
            self.assertIs(listen_socket(ours), constructor.return_value)
            constructor.assert_called_once_with(fileno=3)
            constructor.return_value.getsockname.return_value = ("0.0.0.0", 8181)
            with self.assertRaises(ValueError):
                listen_socket(ours)

    def test_espeak_data_is_moved_not_duplicated_at_shorter_path(self):
        import tempfile
        from pathlib import Path
        from prepare_model import stage_espeak_data
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / ("nested" * 25) / "espeak-ng-data"
            source.mkdir(parents=True)
            (source / "phontab").write_text("fixture")
            stage_espeak_data(root, source)
            self.assertTrue(source.is_symlink())
            self.assertEqual(source.resolve(), root / "espeak-ng-data")
            self.assertEqual((source / "phontab").read_text(), "fixture")
            stage_espeak_data(root, source)
            self.assertEqual(len(list(root.rglob("phontab"))), 1)

    def test_sentence_ramp(self):
        self.assertEqual(chunks("One. Two. Three."), ["One.", "Two.", "Three."])
        self.assertEqual(chunks("Dr. Smith left. Next."), ["Dr. Smith left.", "Next."])

    def test_no_truncation_at_limit(self):
        self.assertEqual(len(validate({"text": "x" * 30000}, Config())[0]), 30000)

    def test_frame_layout(self):
        self.assertEqual(frame(1, b"a"), b"\1\1\0\0\0a")
        self.assertEqual(len(wav_bytes(b"\0\0")), 46)


if __name__ == "__main__":
    unittest.main()
