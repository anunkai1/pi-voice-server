/**
 * Minimal Kokoro TTS HTTP server.
 *
 * Adapted from @s1m0n38/pi-voice (MIT) but stripped to the essentials and
 * rewritten to avoid the @huggingface/transformers env flags that trigger an
 * onnxruntime-node native-library path-resolution bug on this box.
 *
 * Loads a single Kokoro ONNX model on startup, keeps it warm in memory, and
 * exposes a tiny REST surface:
 *
 *   GET  /health            → { status, modelLoaded, voices }
 *   GET  /voices            → { voices: string[] }
 *   POST /tts               → { text, voice?, speed? } → audio/wav bytes
 *
 * Configuration via env:
 *   KOKORO_MODEL_ID  default "onnx-community/Kokoro-82M-v1.0-ONNX"
 *   KOKORO_DTYPE     default "q4"   (q4 | q4f16 | q8 | fp16 | fp32)
 *   KOKORO_VOICE     default "af_heart"
 *   KOKORO_HOST      default "127.0.0.1"
 *   KOKORO_PORT      default 8181
 *
 * Model files are cached by transformers.js under
 *   ~/.cache/huggingface/transformers/  (XDG: $XDG_CACHE_HOME)
 */

import { createServer } from "node:http";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { resolve } from "node:path";
import { KokoroTTS } from "kokoro-js";
import { phonemize } from "phonemizer";

const MODEL_ID = process.env.KOKORO_MODEL_ID || "onnx-community/Kokoro-82M-v1.0-ONNX";
const DTYPE = process.env.KOKORO_DTYPE || "q4";
const DEFAULT_VOICE = process.env.KOKORO_VOICE || "af_heart";
const HOST = process.env.KOKORO_HOST || "127.0.0.1";
const PORT = Number(process.env.KOKORO_PORT || 8181);

let tts = null;
const stateFile = resolve(homedir(), ".pi", "voice", "server-state.json");

function log(...args) {
	console.log("[kokoro-tts]", ...args);
}

function saveState(s) {
	try {
		const dir = resolve(homedir(), ".pi", "voice");
		mkdirSync(dir, { recursive: true });
		writeFileSync(stateFile, `${JSON.stringify(s, null, 2)}\n`);
	} catch {
		/* best-effort */
	}
}

async function loadModel() {
	log(`loading model ${MODEL_ID} (dtype=${DTYPE}) …`);
	const t0 = Date.now();
	tts = await KokoroTTS.from_pretrained(MODEL_ID, { dtype: DTYPE, device: "cpu" });
	const voices = Object.keys(tts.voices);
	log(`model ready in ${Date.now() - t0}ms — ${voices.length} voices available`);
	saveState({ modelLoaded: true, dtype: DTYPE, voices, at: new Date().toISOString() });
}

/** Float32 PCM → 16-bit little-endian PCM WAV Buffer. */
function float32ToWav(samples, sampleRate) {
	const numChannels = 1;
	const bitsPerSample = 16;
	const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
	const blockAlign = numChannels * (bitsPerSample / 8);
	const dataSize = samples.length * (bitsPerSample / 8);
	const buf = Buffer.alloc(44 + dataSize);
	buf.write("RIFF", 0);
	buf.writeUInt32LE(36 + dataSize, 4);
	buf.write("WAVE", 8);
	buf.write("fmt ", 12);
	buf.writeUInt32LE(16, 16);
	buf.writeUInt16LE(1, 20); // PCM
	buf.writeUInt16LE(numChannels, 22);
	buf.writeUInt32LE(sampleRate, 24);
	buf.writeUInt32LE(byteRate, 28);
	buf.writeUInt16LE(blockAlign, 32);
	buf.writeUInt16LE(bitsPerSample, 34);
	buf.write("data", 36);
	buf.writeUInt32LE(dataSize, 40);
	let off = 44;
	for (let i = 0; i < samples.length; i++) {
		const s = Math.max(-1, Math.min(1, samples[i] ?? 0));
		buf.writeInt16LE(Math.round(s * 0x7fff), off);
		off += 2;
	}
	return buf;
}

// 2 MB hard cap on any single request body. TTS text is tiny (the
// agentchatbox proxy caps at 30 000 chars; kidstories sends a page at a
// time), so this only ever trips on misuse/abuse — but without it a single
// oversized POST would accumulate unbounded chunks and OOM the warm-model
// process (a cold reload is ~600 ms + ~291 MB).
const MAX_BODY_BYTES = 2 * 1024 * 1024;

function readBody(req) {
	return new Promise((resolveP, rejectP) => {
		const chunks = [];
		let size = 0;
		let aborted = false;
		req.on("data", (c) => {
			if (aborted) return;
			size += c.length;
			if (size > MAX_BODY_BYTES) {
				// Stop ingesting, tear down the socket, and reject. The handler's
				// catch turns this into a 500 with a clear message. Destroying is
				// intentional: we don't want to buffer the rest of a runaway body.
				aborted = true;
				req.destroy();
				rejectP(new Error(`request body too large (max ${MAX_BODY_BYTES} bytes)`));
				return;
			}
			chunks.push(c);
		});
		req.on("end", () => {
			if (!aborted) resolveP(Buffer.concat(chunks).toString("utf-8"));
		});
		req.on("error", (e) => {
			if (!aborted) rejectP(e);
		});
	});
}

function sendJson(res, data, status = 200) {
	const body = JSON.stringify(data);
	res.writeHead(status, {
		"Content-Type": "application/json",
		"Content-Length": Buffer.byteLength(body),
	});
	res.end(body);
}

// ── Text chunking ──────────────────────────────────────────────────
// Kokoro's phoneme encoder has a FIXED context window: the tokenizer's
// model_max_length is 512, and kokoro-js calls it with {truncation:true}
// (generate() → tokenizer(phonemes, {truncation:true})). So any chunk whose
// PHONEME tokens exceed ~510 is SILENTLY truncated: Kokoro synthesizes only
// the first ~510 tokens of that chunk and drops the rest, while the NEXT
// chunk synthesizes in full. The audible effect is a sentence cut short
// mid-way ("truncated"), then the following chunk resumes ("continued") —
// exactly what the LongTTS button was doing on long replies.
//
// A char budget alone can't prevent this: the char→phoneme-token ratio
// ranges from ~0.7 (terse prose) to >1.0 (numbers, currency, polysyllabic
// words) and higher still for code/URLs. Real voice replies measured at
// 4/29 chunks silently truncated under the old 500-char cap. So we split on
// ACTUAL phoneme-token count: phonemize each char-budgeted chunk, tokenize it
// WITHOUT truncation (to read its true length), and re-split on word
// boundaries anything over the safe limit.
const CHUNK_MAX_CHARS = 500; // first-pass char budget (keeps paragraph/sentence/word boundaries)
const SAFE_TOKEN_LIMIT = 480; // phoneme tokens; ~30 under the ~510 content cap → margin for safety

/**
 * kokoro-js's post-phonemization cleanup, copied verbatim from
 * node_modules/kokoro-js/dist/kokoro.js (v1.2.1) so our token count matches
 * what the library feeds its tokenizer. The 1:1 glyph substitutions
 * (ʲ→j, r→ɹ, x→k, ɬ→l) don't change the token COUNT; the count-affecting
 * rules (the "hundred" space insertion, the trailing "s"/"z" join, the
 * "ninety"→"ninedi" fix) are rare but included for an exact match.
 * `lang` is "a" (en-us) or "b" (en-gb), mirroring kokoro's _validate_voice.
 */
function kokoroPostProcess(joined, lang = "a") {
	let i = joined
		.replace(/kəkˈoːɹoʊ/g, "kˈoʊkəɹoʊ")
		.replace(/kəkˈɔːɹəʊ/g, "kˈəʊkəɹoʊ")
		.replace(/ʲ/g, "j")
		.replace(/r/g, "ɹ")
		.replace(/x/g, "k")
		.replace(/ɬ/g, "l")
		.replace(/(?<=[a-zɹː])(?=hˈʌndɹɪd)/g, " ")
		.replace(/ z(?=[;:,.!?¡¿—…"«»“” ]|$)/g, "z");
	if (lang === "a") i = i.replace(/(?<=nˈaɪn)ti(?!ː)/g, "di");
	return i.trim();
}

/**
 * True phoneme-token count for `text` — what the model would actually
 * consume. Phonemizes via espeak-ng WASM (bundled in the `phonemizer` dep),
 * applies kokoro-js's post-processing, then tokenizes WITHOUT truncation so
 * the returned length is the real one, not the silently-clamped 512. Returns
 * 0 on any failure so the caller falls back to the char-budgeted chunk
 * unchanged (degrading to the pre-fix behavior rather than breaking TTS).
 */
async function phonemeTokenCount(text, lang = "a") {
	try {
		const ph = (await phonemize(text, lang === "a" ? "en-us" : "en")).join(" ");
		const processed = kokoroPostProcess(ph, lang);
		const { input_ids } = tts.tokenizer(processed, { truncation: false });
		return input_ids.dims.at(-1);
	} catch {
		return 0;
	}
}

/**
 * Re-split a single char-budgeted chunk that measured OVER the token limit,
 * greedily packing words until adding the next would cross SAFE_TOKEN_LIMIT.
 * A lone word whose own tokens exceed the limit (vanishingly rare — e.g. a
 * long URL with no spaces) is hard-split by characters, each half re-checked.
 */
async function splitByTokenBudget(chunk, lang = "a") {
	const pieces = [];
	let buf = "";
	for (const w of chunk.split(/\s+/).filter(Boolean)) {
		const cand = buf ? `${buf} ${w}` : w;
		if ((await phonemeTokenCount(cand, lang)) > SAFE_TOKEN_LIMIT) {
			if (buf) {
				pieces.push(buf);
				buf = "";
			}
			// The word alone is over budget: hard-split it character by character.
			if ((await phonemeTokenCount(w, lang)) > SAFE_TOKEN_LIMIT) {
				let half = "";
				for (const ch of w) {
					if ((await phonemeTokenCount(half + ch, lang)) > SAFE_TOKEN_LIMIT) {
						if (half) pieces.push(half);
						half = ch;
					} else {
						half += ch;
					}
				}
				if (half) pieces.push(half);
				continue;
			}
			buf = w;
		} else {
			buf = cand;
		}
	}
	if (buf) pieces.push(buf);
	return pieces;
}

/**
 * Split `text` into chunks each safe for Kokoro's context window. Two passes:
 *   1) hierarchical char-based split (paragraph → sentence → word) for
 *      natural boundaries and to keep most chunks intact, then
 *   2) a token-aware pass that measures each chunk's phoneme tokens and
 *      re-splits any that exceed SAFE_TOKEN_LIMIT.
 * `voice` selects the phonemization language (a*=en-us, b*=en-gb), matching
 * what kokoro-js itself uses.
 */
async function chunkText(text, voice = DEFAULT_VOICE) {
	const lang = voice.at(0) === "b" ? "b" : "a";
	const charChunks = charSplit(text, CHUNK_MAX_CHARS);
	const out = [];
	for (const ch of charChunks) {
		if ((await phonemeTokenCount(ch, lang)) <= SAFE_TOKEN_LIMIT) {
			out.push(ch);
		} else {
			out.push(...(await splitByTokenBudget(ch, lang)));
		}
	}
	return out;
}

/**
 * Pure synchronous char-budget splitter (paragraph → sentence → word wrap).
 * Extracted from the original chunkText so the token-aware pass above can
 * start from good boundaries before measuring phoneme tokens.
 */
function charSplit(text, maxChars) {
	const clean = text.replace(/\r\n/g, "\n").trim();
	if (!clean) return [];
	if (clean.length <= maxChars) return [clean];

	const chunks = [];
	const pushIf = (s) => {
		const t = s.trim();
		if (t) chunks.push(t);
	};

	// Split on blank lines (paragraphs) first.
	for (const para of clean.split(/\n\s*\n/)) {
		if (para.length <= maxChars) {
			pushIf(para);
			continue;
		}
		// Paragraph too big: split on sentence enders, keeping punctuation.
		const sentences = para.match(/[^.!?]*[.!?]+(?:["')\]]+)?|[^.!?]+$/g) ?? [para];
		let buf = "";
		for (const sent of sentences) {
			const s = sent.trim();
			if (!s) continue;
			if (s.length > maxChars) {
				// Single sentence longer than budget: flush, then hard-wrap by word.
				if (buf) {
					pushIf(buf);
					buf = "";
				}
				let wbuf = "";
				for (const w of s.split(/\s+/)) {
					if ((wbuf + " " + w).trim().length > maxChars) {
						pushIf(wbuf);
						wbuf = w;
					} else {
						wbuf = (wbuf + " " + w).trim();
					}
				}
				if (wbuf) pushIf(wbuf);
				continue;
			}
			if ((buf + " " + s).length > maxChars) {
				pushIf(buf);
				buf = s;
			} else {
				buf = (buf + " " + s).trim();
			}
		}
		if (buf) pushIf(buf);
	}
	return chunks;
}

// Serialize /tts calls — Kokoro synthesis is not concurrency-safe on one model.
let ttsChain = Promise.resolve();
function enqueueTts(fn) {
	const next = ttsChain.then(fn, fn);
	ttsChain = next.catch(() => {});
	return next;
}

const server = createServer(async (req, res) => {
	try {
		const url = new URL(req.url ?? "/", `http://${HOST}:${PORT}`);
		const path = url.pathname;

		if (path === "/health" && req.method === "GET") {
			return sendJson(res, {
				status: "ok",
				modelLoaded: tts !== null,
				dtype: DTYPE,
				voice: DEFAULT_VOICE,
				voiceCount: tts ? Object.keys(tts.voices).length : 0,
			});
		}

		if (path === "/voices" && req.method === "GET") {
			if (!tts) return sendJson(res, { error: "model not loaded" }, 503);
			return sendJson(res, { voices: Object.keys(tts.voices) });
		}

		if (path === "/tts" && req.method === "POST") {
			if (!tts) return sendJson(res, { error: "model not loaded" }, 503);
			const body = JSON.parse(await readBody(req));
			const text = (body.text ?? "").trim();
			if (!text) return sendJson(res, { error: "missing 'text'" }, 400);
			const voice = body.voice || DEFAULT_VOICE;
			const speed = Number(body.speed ?? 1.0);

			const result = await enqueueTts(async () => {
				const chunks = await chunkText(text, voice);
				log(`tts: chars=${text.length} chunks=${chunks.length} voice=${voice} speed=${speed}`);
				let sampleRate = 0;
				const parts = [];
				for (let i = 0; i < chunks.length; i++) {
					const audio = await tts.generate(chunks[i], { voice, speed });
					sampleRate = audio.sampling_rate;
					parts.push(audio.audio);
					// ~200ms silence between chunks so sentences don't run together.
					if (i < chunks.length - 1) {
						parts.push(new Float32Array(Math.floor(sampleRate * 0.2)));
					}
				}
				const total = parts.reduce((n, p) => n + p.length, 0);
				const merged = new Float32Array(total);
				let off = 0;
				for (const p of parts) {
					merged.set(p, off);
					off += p.length;
				}
				return float32ToWav(merged, sampleRate);
			});
			res.writeHead(200, {
				"Content-Type": "audio/wav",
				"Content-Length": result.length,
				"Cache-Control": "no-store",
			});
			return res.end(result);
		}

		// Streaming variant: synthesize chunks one at a time and write each
		// chunk's WAV the moment it's ready, so the client can start playing
		// the first chunk while later chunks are still being synthesized.
		// The whole call still occupies one slot in the serial ttsChain so
		// its chunks never interleave with a concurrent /tts request.
		//
		// Binary frame layout (little-endian), written straight to the body:
		//   [1 byte type][uint32 LE length N][N bytes payload]
		//     type 0x01 DATA → payload = one complete WAV file (one text chunk)
		//     type 0x00 END  → no payload; clean end of stream
		//     type 0x80 ERR  → payload = UTF-8 error message
		if (path === "/tts/stream" && req.method === "POST") {
			if (!tts) return sendJson(res, { error: "model not loaded" }, 503);
			const body = JSON.parse(await readBody(req));
			const text = (body.text ?? "").trim();
			if (!text) return sendJson(res, { error: "missing 'text'" }, 400);
			const voice = body.voice || DEFAULT_VOICE;
			const speed = Number(body.speed ?? 1.0);

			res.writeHead(200, {
				"Content-Type": "application/octet-stream",
				"Cache-Control": "no-store",
			});
			// Client disconnect detection: stop synthesizing remaining chunks
			// if the browser navigates away or hits stop mid-stream.
			let clientGone = false;
			res.on("close", () => {
				clientGone = true;
			});
			const writeFrame = (type, payload) => {
				if (clientGone) return;
				const plen = payload ? payload.length : 0;
				const buf = Buffer.allocUnsafe(5 + plen);
				buf[0] = type;
				buf.writeUInt32LE(plen, 1);
				if (payload && plen) payload.copy(buf, 5);
				res.write(buf);
			};

			await enqueueTts(async () => {
				try {
					const chunks = await chunkText(text, voice);
					log(`tts/stream: chars=${text.length} chunks=${chunks.length} voice=${voice} speed=${speed}`);
					let sampleRate = 0;
					for (let i = 0; i < chunks.length; i++) {
						if (clientGone) break;
						const out = await tts.generate(chunks[i], { voice, speed });
						sampleRate = out.sampling_rate;
						let samples = out.audio;
						// Bake ~200ms of trailing silence into every chunk except
						// the last, so sentences don't run together when the
						// client chains the WAVs back to back.
						if (i < chunks.length - 1) {
							const sil = new Float32Array(Math.floor(sampleRate * 0.2));
							const cat = new Float32Array(samples.length + sil.length);
							cat.set(samples, 0);
							cat.set(sil, samples.length);
							samples = cat;
						}
						writeFrame(0x01, float32ToWav(samples, sampleRate));
					}
					writeFrame(0x00, null); // END
				} catch (err) {
					log("tts/stream synth error:", err);
					writeFrame(0x80, Buffer.from(err instanceof Error ? err.message : String(err), "utf8"));
				} finally {
					try {
						res.end();
					} catch {
						/* socket already gone */
					}
				}
			});
			return;
		}

		return sendJson(res, { error: "not found" }, 404);
	} catch (err) {
		log("request error:", err);
		if (!res.headersSent) {
			sendJson(res, { error: err instanceof Error ? err.message : String(err) }, 500);
		}
	}
});

// Load model on startup, then listen. If the model load fails, the process
// exits and systemd will restart it (Restart=on-failure).
loadModel()
	.then(() => {
		server.listen(PORT, HOST, () => {
			log(`listening on http://${HOST}:${PORT}`);
		});
	})
	.catch((err) => {
		log("FATAL: model load failed:", err);
		process.exit(1);
	});
