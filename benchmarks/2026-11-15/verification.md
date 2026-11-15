# Final verification — 15/11/2026

Runtime deployed: `5c2e3bb28d140dfce8360962eee3da01f037dd22`.
Infra/deployer/unit: `d73f69b`.
Later measurement/evidence-only commits do not require another voice restart.

Commands and results:

- Original backend, before retirement: `npm test` — all 17 tests passed.
- Python checkout and immutable staging build: `.venv/bin/python -m unittest -v`
  — all 14 tests passed, including serial inference, disconnect cancellation,
  exact UTF-16 text limit/body/audio limits, legacy framing, opt-in metadata,
  error frames and post-rename native data packaging.
- `uv pip check` — all 99 packages compatible.
- `uvx ruff check --select F,E9 server.py test_server.py prepare_model.py benchmark.py verify_contract.py verify_lifecycle.py`
  — passed.
- `uvx --from pip-audit==2.10.1 pip-audit --path .venv/lib/python3.12/site-packages`
  — no known vulnerabilities among matched packages; skips the direct-URL
  spaCy language model and Torch's local `+cpu` distribution name.
- `uvx --from pip-audit==2.10.1 pip-audit --disable-pip --no-deps -r .torch-audit.txt`
  with `torch==2.13.0` — no known vulnerabilities. The pinned spaCy model is not
  covered by the PyPI feed. Earlier dependency findings were resolved by pinning
  aiohttp 3.14.4 and Torch 2.13.0+cpu before final deployment.
- `.venv/bin/python verify_contract.py http://127.0.0.1:8181` — passed against
  the real sandboxed/socket-activated service. Whole WAV at speed 0.5 (US) and
  speed 2 (British); old DATA/END stream; opt-in metadata/DATA/END stream.
  Actual final output exactly matched committed `contract.json`.
- All 28 entries in `server.VOICES` were POSTed to the live `/tts` with `Hello.`,
  speed 1; every response was a nonempty WAV (`voices.json`).
- `node verify_consumers.mjs http://127.0.0.1:3500` — ACB `/api/tts` and
  `/api/tts/stream` passed. The deployed KidStories `createTtsRunner` processed
  one disposable narration page, using a deliberately unreachable primary to
  exercise fallback to real Kokoro. Temporary output was deleted; no live book,
  app job, consumer source, or tunnel was changed (`consumers.jsonl`).
- `.venv/bin/python verify_lifecycle.py` — inherited `tts` socket FD accepted;
  healthy while unloaded; process exited successfully after the real 60-second
  minimum idle timeout (`lifecycle.json`).
- `bash -n infra/scripts/usr-local-bin/deploy-pi-voice-server` — passed.
- `systemd-analyze verify` with the candidate executable path — passed. Source
  ExecStart initially points through `current` to a JS release without a venv;
  the deployer substitutes the immutable candidate path for preflight only.
- `VOICE_DEPLOY_FAIL_AT=after_health deploy-pi-voice-server` — intentional failure;
  existing shared transaction restored the prior JS release and matching unit.
- `deploy-pi-voice-server --rollback-js` — tested explicitly; q4 health and a real
  24 kHz WAV succeeded. Normal `deploy-pi-voice-server` then restored Python;
  live contract/consumer checks passed again. Saved JS pair remains available.
- Final `cmp` of source/installed voice service unit and deployer — identical;
  service active, `NRestarts=0`, only one Kokoro process, only loopback 8181.

An initial live attempt exposed eSpeak's native path-length limit after renaming
the release. JS was restored immediately. The fix moves (does not duplicate)
eSpeak data to the release root and runs synthesis smoke tests at the final,
read-only path before switching. Failed trial release directories were removed;
the validated JS rollback pair was retained.

Only the owning voice deployer/unit/runtime were reconciled. `deploy-local-bin`
installed the changed voice deploy command using its existing manifest workflow;
it restarts no application. No ACB, KidStories, KidStories2, tunnel, STT, Server4,
Qwen or image service was deployed/restarted. This backend-only approval does not
change Relay's default/selection, playback clock or saved-prefix behaviour.
