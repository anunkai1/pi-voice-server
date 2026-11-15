"""Run a fresh process (cached weights), then three warm requests per text."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request
import wave
import io

TEXTS = {
    "short": "Hello. I can speak to you while the tools keep working.",
    "paragraph": "The garden was quiet after the rain. A small bird landed beside the water bowl and watched the morning light move across the fence. Inside the house, someone opened a window and put the kettle on. There was no need to hurry today. We could finish the useful work first, take a short break, and come back with a clearer idea of what mattered most.",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=18181)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[0] == "--" else args.command
    env = dict(os.environ, KOKORO_PORT=str(args.port), KOKORO_HOST="127.0.0.1", KOKORO_THREADS="4")
    base = f"http://127.0.0.1:{args.port}"
    log = open(args.output + ".log", "w")
    start = time.monotonic()
    process = subprocess.Popen(command, env=env, stdout=log, stderr=log)
    result = {"command": command, "method": "fresh process; cached weights; af_heart speed=1; three warm requests, median; 4 Python Torch threads; no Server4"}
    try:
        for _ in range(1200):
            if process.poll() is not None:
                raise RuntimeError("server exited; inspect benchmark log")
            try:
                urllib.request.urlopen(base + "/health", timeout=1).read()
                break
            except OSError:
                time.sleep(.05)
        result["healthReadySeconds"] = time.monotonic() - start

        def request(text):
            begin = time.monotonic()
            req = urllib.request.Request(base + "/tts", data=json.dumps({"text": text, "voice": "af_heart", "speed": 1}).encode(), headers={"Content-Type": "application/json"})
            data = urllib.request.urlopen(req, timeout=240).read()
            elapsed = time.monotonic() - begin
            with wave.open(io.BytesIO(data)) as wav:
                duration = wav.getnframes() / wav.getframerate()
            return {"seconds": elapsed, "audioSeconds": duration, "rtf": elapsed / duration}
        result["coldShort"] = request(TEXTS["short"])
        result["processToFirstWavSeconds"] = time.monotonic() - start
        for name, text in TEXTS.items():
            trials = [request(text) for _ in range(3)]
            result[name] = {"text": text, "trials": trials, "medianSeconds": statistics.median(t["seconds"] for t in trials)}
        status = Path(f"/proc/{process.pid}/status").read_text()
        result["memory"] = [line for line in status.splitlines() if line.startswith(("VmRSS:", "VmHWM:"))]
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        process.terminate()
        try:
            process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


if __name__ == "__main__":
    main()
