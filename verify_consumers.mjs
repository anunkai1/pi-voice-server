// Read-only ACB synthesis endpoints + deployed KidStories narration runner.
// Writes only a disposable /tmp narration fixture; no live book/job is changed.
import assert from "node:assert/strict";
import { mkdtemp, writeFile, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const acb = process.argv[2] || "http://127.0.0.1:3500";
function wav(bytes) {
  assert.equal(bytes.toString("ascii", 0, 4), "RIFF");
  assert.equal(bytes.toString("ascii", 8, 12), "WAVE");
  assert(bytes.length > 44);
}
for (const route of ["/api/tts/kokoro-stream", "/api/tts/relay-stream"]) {
  const aligned = route.endsWith("/relay-stream");
  const response = await fetch(acb + route, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text: "A short compatibility check. Another sentence.", voice: "af_heart", speed: 1 }),
  });
  assert.equal(response.status, 200);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (aligned) assert.equal(response.headers.get("X-Kokoro-Alignment"), "kokoro-pred-dur-v1");
  let offset = 0;
  const kinds = [];
  while (offset < bytes.length) {
    assert(offset + 5 <= bytes.length);
    const kind = bytes[offset], length = bytes.readUInt32LE(offset + 1);
    assert(offset + 5 + length <= bytes.length);
    assert(kind === 0 || kind === 1 || (aligned && kind === 2));
    if (kind === 1) wav(bytes.subarray(offset + 5, offset + 5 + length));
    kinds.push(kind);
    offset += 5 + length;
  }
  assert.equal(kinds.at(-1), 0);
  assert(kinds.includes(1));
  if (aligned) assert(kinds.includes(2));
  console.log(JSON.stringify({ consumer: "ACB", route, status: "PASS", kinds }));
}
const { createTtsRunner } = require("/home/lepton/.local/lib/server2-releases/kidstories/current/lib/tts.js");
const directory = await mkdtemp(join(tmpdir(), "kokoro-kidstories-check-"));
try {
  const plan = join(directory, "plan.json");
  await writeFile(plan, JSON.stringify({ out_dir: directory, pages: [{ page: 1, text: "A bird visited the garden." }] }));
  const runner = createTtsRunner({ baseUrl: "http://127.0.0.1:8181", voice: "af_heart", speed: 1,
    primary: { baseUrl: "http://127.0.0.1:1" } });
  const result = await runner.runBatch(plan);
  assert.deepEqual(result, { ok: [1], failed: [] });
  wav(await readFile(join(directory, "page-1.wav")));
  console.log(JSON.stringify({ consumer: "KidStories", path: "deployed narration runner, failed primary → Kokoro", status: "PASS", result }));
} finally {
  await rm(directory, { recursive: true, force: true });
}
