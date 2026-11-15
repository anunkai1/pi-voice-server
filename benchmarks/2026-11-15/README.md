# Server2 comparison — 15/11/2026 (AEST)

CPU: Intel Core i5-1135G7, 4 physical cores / 8 threads. No Server4 involved.

`benchmark.py` starts a fresh process on loopback 18181, waits for health, then
requests `/tts` with af_heart, speed 1: one cold short request followed by three
warm short and three warm paragraph requests. Texts and every trial are in the
raw JSON. Warm figures below are medians; cold figures are single trials.
Weights were cached before both trials; OS page cache was not flushed. Thus
“cold” means process/model cold, **not** first-time downloading or cold disk.
No other benchmark synthesis was run concurrently. These are observations on a
shared host, not statistically isolated CPU benchmarks.

Baseline: kokoro-js/q4 at source commit
`2918820b7031c17f3f5df833c9b736fcfb41d566`, its existing ONNX defaults.
Python: kokoro 0.9.4, Torch 2.13.0+cpu, pinned upstream model,
`KOKORO_THREADS=4`, `HF_HUB_OFFLINE=1`, local staged weights. Both return full
mono PCM16 24 kHz WAVs. Output lengths differ slightly because phonemisation and
model precision differ (4.500 vs 4.350 s short; 21.425 vs 21.375 s paragraph).
Python generates duration metadata internally even for legacy requests.

| Measurement | kokoro-js q4 | Python fp32 |
|---|---:|---:|
| Process → health ready | 0.250 s | 0.221 s |
| Cold first short synthesis HTTP request | 2.562 s | 8.727 s |
| Process → first complete short WAV | 2.812 s | 8.948 s |
| Warm short synthesis, median of 3 | 1.821 s | 1.326 s |
| Warm paragraph synthesis, median of 3 | 9.191 s | 6.559 s |
| Peak process RSS (`/proc/PID/status:VmHWM`) | 817.9 MiB | 1636.6 MiB |
| RSS after all trials (`VmRSS`) | 817.9 MiB | 1484.9 MiB |
| Installed runtime including weights, apparent size | 0.557 GB | 1.567 GB |

Installed-size method: `du -sb` of the pruned immutable JS release
(556,653,732 bytes, including its q4 model) and the final immutable Python release
(1,566,742,932 bytes, including all weights/voices and relocated eSpeak data).
Neither counts the system Node/Python interpreter, shared download caches or
rollback copies. Exact release IDs and byte counts are in the JSON.
Python model packs include every advertised voice, not only the benchmark one.
The source development environment used for the timings is larger than the fresh
release because it retains generated/import caches; its size is not used here.

**Conclusion:** Python was about 27% faster warm for the short text and 29%
faster warm for the paragraph here. Its first complete WAV was about 3.2×
slower from a fresh process; peak RSS was about 2× higher, installed runtime
about 2.8× larger. This does not establish streaming time-to-first-audio or
subjective Relay responsiveness; the owner still needs to judge those.

Reproduce against an old immutable JS release (or a checkout of the baseline
with `npm ci`) and a prepared Python checkout, sequentially:

```
python3 benchmark.py --output /tmp/kokoro-js.json -- node /path/to/js-release/server.mjs
HF_HUB_OFFLINE=1 python3 benchmark.py --output /tmp/kokoro-python.json -- .venv/bin/python server.py
```

`contract.json` records actual opt-in model-duration timings, `voices.json`
records synthesis with all 28 advertised voices, and `lifecycle.json` records
inherited-FD activation and successful idle exit after the 60-second minimum.
The original JS utility test suite passed all 17 tests before retiring the JS
backend; the Python contract suite is now `python -m unittest -v`.

An initial cutover passed pre-rename synthesis but failed live eSpeak startup
because the final nested package path exceeded its native path-length limit.
The explicit rollback command restored JS successfully. Packaging now moves
that data (without duplication) to the release root and smoke-tests the final,
read-only release path before switching. The injected post-health transaction
failure also restored the previous JS unit/release successfully. A final explicit
rollback round-trip verified actual q4 WAV synthesis, then reinstalled Python.
`consumers.jsonl` records real ACB whole/stream synthesis and the deployed
KidStories narration runner using a simulated Ming failure and real Kokoro
fallback, without changing any live book or job.
