import asyncio
import io
import json
import struct
import unittest
import wave

from aiohttp.test_utils import TestClient, TestServer
from server import ALIGNMENT, Config, VOICES, chunks, create_app, frame, validate, wav_bytes


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


class PureTests(unittest.TestCase):
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
