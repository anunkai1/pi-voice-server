"""Stage pinned weights and every advertised voice, never download at runtime."""
from pathlib import Path
from huggingface_hub import snapshot_download
from server import VOICES

REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"

if __name__ == "__main__":
    directory = Path(__file__).parent / "model"
    snapshot_download("hexgrad/Kokoro-82M", revision=REVISION, local_dir=directory,
                      allow_patterns=["config.json", "kokoro-v1_0.pth"] + [f"voices/{voice}.pt" for voice in VOICES])
    (directory / "REVISION").write_text(REVISION + "\n")
