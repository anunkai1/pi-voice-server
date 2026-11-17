"""Real synthesis checks for the legacy contract and opt-in timing protocol."""
import io
import json
import struct
import sys
import urllib.request
import wave

BASE = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8181"


def request(path, body=None, headers=None):
    headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(BASE + path, data=None if body is None else json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read(), dict(response.headers)


def wav(data):
    with wave.open(io.BytesIO(data)) as audio:
        assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 24000)
        assert audio.getnframes() > 0
        return audio.getnframes()


def frames(data):
    result = []
    while data:
        assert len(data) >= 5
        kind, length = struct.unpack("<BI", data[:5])
        assert len(data) >= 5 + length
        result.append((kind, data[5:5 + length]))
        data = data[5 + length:]
    return result


if __name__ == "__main__":
    health = json.loads(request("/health")[0])
    assert health["modelAvailable"] and health["status"] == "ok"
    voices = json.loads(request("/voices")[0])["voices"]
    assert len(voices) == 28 and health["voice"] == "af_heart"
    for voice, speed in [("af_heart", .5), ("bf_emma", 2)]:
        wav(request("/tts", {"text": "Hello world.", "voice": voice, "speed": speed})[0])
    old = frames(request("/tts/stream", {"text": "Hello world. Another sentence."})[0])
    assert old[-1][0] == 0 and all(kind in (0, 1) for kind, _ in old)
    for kind, payload in old:
        if kind == 1:
            wav(payload)
    data, headers = request("/tts/stream", {"text": "Hello world."}, {"X-Kokoro-Alignment": "kokoro-pred-dur-v1"})
    assert headers.get("X-Kokoro-Alignment") == "kokoro-pred-dur-v1"
    aligned = frames(data)
    assert [kind for kind, _ in aligned] == [2, 1, 0]
    metadata = json.loads(aligned[0][1])
    count = wav(aligned[1][1])
    assert metadata["submittedText"] == "Hello world."
    assert metadata["alignmentComplete"]
    assert metadata["words"] and all(0 <= word["startSample"] <= word["endSample"] <= count for word in metadata["words"])
    print(json.dumps({"contract": "PASS", "legacyFrames": [kind for kind, _ in old], "alignmentExample": metadata}, indent=2))
