"""Exercise inherited socket FD and a real minimum (60s) idle shutdown."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request

if __name__ == "__main__":
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    script = """
import os, runpy
os.dup2(int(os.environ['TEST_FD']), 3)
os.environ.update(LISTEN_PID=str(os.getpid()), LISTEN_FDS='1', LISTEN_FDNAMES='tts')
runpy.run_path('server.py', run_name='__main__')
"""
    env = dict(os.environ, TEST_FD=str(listener.fileno()), KOKORO_IDLE_TIMEOUT_MS="60000")
    start = time.monotonic()
    process = subprocess.Popen([sys.executable, "-c", script], pass_fds=(listener.fileno(),), env=env, cwd=Path(__file__).parent, stdout=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{port}/health"
        health = json.load(urllib.request.urlopen(url, timeout=10))
        assert health["status"] == "ok" and not health["modelLoaded"]
        process.wait(timeout=70)
        elapsed = time.monotonic() - start
        assert process.returncode == 0 and 60 <= elapsed < 70
        print(json.dumps({"socketActivation": "PASS", "idleExitSeconds": elapsed, "exitCode": process.returncode}))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        listener.close()
