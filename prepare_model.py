"""Stage pinned weights and every advertised voice, never download at runtime."""
from pathlib import Path
import os
import shutil
import espeakng_loader
from huggingface_hub import snapshot_download
from server import VOICES

REVISION = "f3ff3571791e39611d31c381e3a41a3af07b4987"

def stage_espeak_data(root, source=None):
    # libespeak-ng has a short internal path limit. The immutable release's
    # nested site-packages path exceeds it after staging is renamed. Move the
    # same data (not another copy) to the release root; phonemizer resolves the
    # relative symlink before passing the shorter path to native eSpeak.
    source = Path(source or espeakng_loader.get_data_path())
    target = root / "espeak-ng-data"
    if source.resolve() != target.resolve():
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(source), str(target))
        source.symlink_to(os.path.relpath(target, source.parent), target_is_directory=True)


if __name__ == "__main__":
    root = Path(__file__).parent
    stage_espeak_data(root)
    directory = root / "model"
    snapshot_download("hexgrad/Kokoro-82M", revision=REVISION, local_dir=directory,
                      allow_patterns=["config.json", "kokoro-v1_0.pth"] + [f"voices/{voice}.pt" for voice in VOICES])
    (directory / "REVISION").write_text(REVISION + "\n")
