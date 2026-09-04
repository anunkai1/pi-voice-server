/**
 * Minimal Kokoro TTS HTTP server.
 *
 * Adapted from @s1m0n38/pi-voice (MIT) but stripped to the essentials and
 * rewritten to avoid the @huggingface/transformers env flags that trigger an
 * onnxruntime-node native-library path-resolution bug on this box.
 *
 * Lazily loads a single Kokoro ONNX model for synthesis, keeps it warm during
 * a burst, and exposes a tiny REST surface:
 *
 *   GET  /health            → capability, residency, and voice metadata
 *   GET  /voices            → { voices: string[] }
 *   POST /tts               → { text, voice?, speed? } → audio/wav bytes
 *
 * Configuration via env:
 *   KOKORO_MODEL_ID  default "onnx-community/Kokoro-82M-v1.0-ONNX"
 *   KOKORO_DTYPE     default "q4"   (q4 | q4f16 | q8 | fp16 | fp32)
 *   KOKORO_VOICE     default "af_heart"
 *   KOKORO_HOST      default "127.0.0.1"
 *   KOKORO_PORT      default 8181 (used without a systemd socket)
 *   KOKORO_IDLE_TIMEOUT_MS default 600000 (10 minutes)
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
import {
	HttpError,
	MAX_TEXT_CHARS,
	parseTtsRequest,
	readBody,
	writeWithBackpressure,
} from "./lib/http-utils.mjs";
import {
	createIdleShutdownTimer,
	modelHealthState,
	parseIdleTimeout,
	resolveListenTarget,
} from "./lib/lifecycle.mjs";

const MODEL_ID = process.env.KOKORO_MODEL_ID || "onnx-community/Kokoro-82M-v1.0-ONNX";
const DTYPE = process.env.KOKORO_DTYPE || "q4";
const DEFAULT_VOICE = process.env.KOKORO_VOICE || "af_heart";
const HOST = process.env.KOKORO_HOST || "127.0.0.1";
const PORT = Number(process.env.KOKORO_PORT || 8181);
const MAX_AUDIO_BYTES = Math.max(1024 * 1024, Number(process.env.KOKORO_MAX_AUDIO_BYTES) || 64 * 1024 * 1024);
const SHUTDOWN_DRAIN_TIMEOUT_MS = Math.max(
	10000,
	Number(process.env.SHUTDOWN_DRAIN_TIMEOUT_MS) || 600000,
);
const IDLE_TIMEOUT_MS = parseIdleTimeout(process.env.KOKORO_IDLE_TIMEOUT_MS);
const LISTEN_TARGET = resolveListenTarget({ host: HOST, port: PORT });

// Keep voice validation and /voices available while the expensive model is
// absent. This is the exact Kokoro 82M v1 voice catalogue; model load verifies
// it before serving synthesis so upstream drift fails closed.
const KNOWN_VOICES = [
	"af_heart", "af_alloy", "af_aoede", "af_bella", "af_jessica", "af_kore", "af_nicole",
	"af_nova", "af_river", "af_sarah", "af_sky", "am_adam", "am_echo", "am_eric", "am_fenrir",
	"am_liam", "am_michael", "am_onyx", "am_puck", "am_santa", "bf_emma", "bf_isabella",
	"bm_george", "bm_lewis", "bf_alice", "bf_lily", "bm_daniel", "bm_fable",
];

let tts = null;
let modelLoadPromise = null;
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
	const loaded = await KokoroTTS.from_pretrained(MODEL_ID, { dtype: DTYPE, device: "cpu" });
	const voices = Object.keys(loaded.voices);
	if (JSON.stringify(voices) !== JSON.stringify(KNOWN_VOICES)) {
		await loaded.model.dispose();
		throw new Error("loaded Kokoro voice catalogue does not match the reviewed server catalogue");
	}
	tts = loaded;
	log(`model ready in ${Date.now() - t0}ms — ${voices.length} voices available`);
	saveState({ modelLoaded: true, dtype: DTYPE, voices, at: new Date().toISOString() });
	return loaded;
}

async function ensureModel() {
	if (tts) return tts;
	modelLoadPromise ??= loadModel().finally(() => { modelLoadPromise = null; });
	try {
		return await modelLoadPromise;
	} catch (error) {
		// Preserve the request error, then let systemd replace a process whose
		// native/model initialisation failed. The socket queues later callers.
		const timer = setTimeout(() => process.exit(1), 1000);
		timer.unref();
		throw error;
	}
}

/** Float32 PCM → 16-bit little-endian PCM WAV Buffer. */
function float32ToWav(samples, sampleRate) {
	const numChannels = 1;
	const bitsPerSample = 16;
	const byteRate = sampleRate * numChannels * (bitsPerSample / 8);
	const blockAlign = numChannels * (bitsPerSample / 8);
	const dataSize = samples.length * (bitsPerSample / 8);
	if (44 + dataSize > MAX_AUDIO_BYTES) {
		throw new HttpError(413, `generated audio exceeds KOKORO_MAX_AUDIO_BYTES (${44 + dataSize} > ${MAX_AUDIO_BYTES})`);
	}
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

/** Convert multiple PCM parts directly into one WAV without first allocating
 * a second merged Float32Array. This halves the large-response copy overhead. */
function float32PartsToWav(parts, sampleRate) {
	const totalSamples = parts.reduce((sum, part) => sum + part.length, 0);
	const dataSize = totalSamples * 2;
	if (44 + dataSize > MAX_AUDIO_BYTES) {
		throw new HttpError(413, `generated audio exceeds KOKORO_MAX_AUDIO_BYTES (${44 + dataSize} > ${MAX_AUDIO_BYTES})`);
	}
	const buf = Buffer.alloc(44 + dataSize);
	buf.write("RIFF", 0);
	buf.writeUInt32LE(36 + dataSize, 4);
	buf.write("WAVE", 8);
	buf.write("fmt ", 12);
	buf.writeUInt32LE(16, 16);
	buf.writeUInt16LE(1, 20);
	buf.writeUInt16LE(1, 22);
	buf.writeUInt32LE(sampleRate, 24);
	buf.writeUInt32LE(sampleRate * 2, 28);
	buf.writeUInt16LE(2, 32);
	buf.writeUInt16LE(16, 34);
	buf.write("data", 36);
	buf.writeUInt32LE(dataSize, 40);
	let offset = 44;
	for (const samples of parts) {
		for (let i = 0; i < samples.length; i++) {
			const sample = Math.max(-1, Math.min(1, samples[i] ?? 0));
			buf.writeInt16LE(Math.round(sample * 0x7fff), offset);
			offset += 2;
		}
	}
	return buf;
}

async function readTtsRequest(req) {
	const contentType = String(req.headers["content-type"] || "").toLowerCase();
	if (!contentType.includes("application/json")) {
		throw new HttpError(415, "Content-Type must be application/json");
	}
	let body;
	try {
		body = JSON.parse(await readBody(req));
	} catch (error) {
		if (error instanceof HttpError) throw error;
		throw new HttpError(400, "invalid JSON body");
	}
	return parseTtsRequest(body, {
		voices: KNOWN_VOICES,
		defaultVoice: DEFAULT_VOICE,
		maxTextChars: MAX_TEXT_CHARS,
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
let activeSyntheses = 0;
let shuttingDown = false;
const idleShutdown = createIdleShutdownTimer({
	timeoutMs: IDLE_TIMEOUT_MS,
	isIdle: () => activeSyntheses === 0,
	onIdle: () => void shutdown("idle timeout"),
});
function enqueueTts(fn) {
	idleShutdown.workStarted();
	activeSyntheses += 1;
	const next = ttsChain.then(fn, fn);
	ttsChain = next.catch(() => {});
	const settled = () => {
		activeSyntheses -= 1;
		idleShutdown.workFinished();
	};
	void next.then(settled, settled);
	return next;
}

const server = createServer(async (req, res) => {
	try {
		const url = new URL(req.url ?? "/", `http://${HOST}:${PORT}`);
		const path = url.pathname;

		if (path === "/health" && req.method === "GET") {
			return sendJson(res, {
				...modelHealthState({
					resident: tts !== null,
					loading: modelLoadPromise !== null,
				}),
				dtype: DTYPE,
				voice: DEFAULT_VOICE,
				voiceCount: KNOWN_VOICES.length,
				maxTextChars: MAX_TEXT_CHARS,
				maxAudioBytes: MAX_AUDIO_BYTES,
				idleTimeoutMs: IDLE_TIMEOUT_MS,
				activeSyntheses,
				status: shuttingDown ? "draining" : "ok",
			});
		}

		if (path === "/voices" && req.method === "GET") {
			return sendJson(res, { voices: KNOWN_VOICES });
		}

		if (path === "/tts" && req.method === "POST") {
			const { text, voice, speed } = await readTtsRequest(req);
			let clientGone = false;
			req.once("aborted", () => { clientGone = true; });
			res.once("close", () => { if (!res.writableEnded) clientGone = true; });

			const result = await enqueueTts(async () => {
				if (clientGone) throw new HttpError(499, "client disconnected");
				const activeTts = await ensureModel();
				const chunks = await chunkText(text, voice);
				log(`tts: chars=${text.length} chunks=${chunks.length} voice=${voice} speed=${speed}`);
				let sampleRate = 0;
				const parts = [];
				let estimatedWavBytes = 44;
				for (let i = 0; i < chunks.length; i++) {
					if (clientGone) throw new HttpError(499, "client disconnected");
					const audio = await activeTts.generate(chunks[i], { voice, speed });
					if (clientGone) throw new HttpError(499, "client disconnected");
					sampleRate = audio.sampling_rate;
					// The whole-blob endpoint returns one continuous WAV. Kokoro's
					// generated audio already contains natural sentence/paragraph
					// timing, so do not add an artificial inter-chunk pause here.
					parts.push(audio.audio);
					estimatedWavBytes += audio.audio.length * 2;
					if (estimatedWavBytes > MAX_AUDIO_BYTES) {
						throw new HttpError(413, `generated audio exceeds KOKORO_MAX_AUDIO_BYTES (${estimatedWavBytes} > ${MAX_AUDIO_BYTES})`);
					}
				}
				return float32PartsToWav(parts, sampleRate);
			});
			if (clientGone) return;
			res.writeHead(200, {
				"Content-Type": "audio/wav",
				"Content-Length": result.length,
				"Cache-Control": "no-store",
			});
			await writeWithBackpressure(res, result, () => clientGone);
			return res.end();
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
			const { text, voice, speed } = await readTtsRequest(req);

			res.writeHead(200, {
				"Content-Type": "application/octet-stream",
				"Cache-Control": "no-store",
			});
			// Client disconnect detection: stop synthesizing remaining chunks
			// if the browser navigates away or hits stop mid-stream.
			let clientGone = false;
			let streamedAudioBytes = 0;
			req.once("aborted", () => { clientGone = true; });
			res.on("close", () => { clientGone = true; });
			const writeFrame = async (type, payload) => {
				if (clientGone) throw new HttpError(499, "client disconnected");
				const plen = payload ? payload.length : 0;
				if (type === 0x01) {
					streamedAudioBytes += plen;
					if (streamedAudioBytes > MAX_AUDIO_BYTES) {
						throw new HttpError(413, `streamed audio exceeds KOKORO_MAX_AUDIO_BYTES (${streamedAudioBytes} > ${MAX_AUDIO_BYTES})`);
					}
				}
				const header = Buffer.allocUnsafe(5);
				header[0] = type;
				header.writeUInt32LE(plen, 1);
				await writeWithBackpressure(res, header, () => clientGone);
				if (payload && plen) await writeWithBackpressure(res, payload, () => clientGone);
			};

			await enqueueTts(async () => {
				try {
					if (clientGone) return;
					const activeTts = await ensureModel();
					const chunks = await chunkText(text, voice);
					log(`tts/stream: chars=${text.length} chunks=${chunks.length} voice=${voice} speed=${speed}`);
					let sampleRate = 0;
					for (let i = 0; i < chunks.length; i++) {
						if (clientGone) break;
						const out = await activeTts.generate(chunks[i], { voice, speed });
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
						await writeFrame(0x01, float32ToWav(samples, sampleRate));
					}
					if (!clientGone) await writeFrame(0x00, null); // END
				} catch (err) {
					if (clientGone || (err instanceof HttpError && err.status === 499)) return;
					log("tts/stream synth error:", err);
					await writeFrame(0x80, Buffer.from(err instanceof Error ? err.message : String(err), "utf8"));
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
		if (err instanceof HttpError && err.status === 499) return;
		log("request error:", err);
		if (!res.headersSent) {
			const status = err instanceof HttpError ? err.status : 500;
			sendJson(res, { error: err instanceof Error ? err.message : String(err) }, status);
		} else {
			res.destroy();
		}
	}
});

// Listen without loading the model. Health and voice-list probes therefore
// remain cheap; the first synthesis request loads the model inside the serial
// work queue. After a synthesis burst, clean idle exit releases the process.
server.listen(LISTEN_TARGET.options, () => {
	log(`listening on ${LISTEN_TARGET.source}; lazy model; idle shutdown=${IDLE_TIMEOUT_MS}ms`);
});
server.on("error", (error) => {
	log("FATAL: HTTP listener failed:", error);
	process.exit(1);
});

async function shutdown(signal) {
	if (shuttingDown) return;
	shuttingDown = true;
	idleShutdown.stop();
	log(`received ${signal}; draining ${activeSyntheses} synthesis request(s)`);

	const closed = server.listening
		? new Promise((resolveClose, rejectClose) => {
			server.close((error) => (error ? rejectClose(error) : resolveClose()));
		})
		: Promise.resolve();
	let timer;
	const timeout = new Promise((_, reject) => {
		timer = setTimeout(
			() => reject(new Error(`shutdown drain exceeded ${SHUTDOWN_DRAIN_TIMEOUT_MS}ms`)),
			SHUTDOWN_DRAIN_TIMEOUT_MS,
		);
	});

	try {
		await Promise.race([Promise.all([closed, ttsChain]), timeout]);
		clearTimeout(timer);
		process.exit(0);
	} catch (error) {
		log("shutdown failed:", error instanceof Error ? error.message : String(error));
		process.exit(1);
	}
}

process.once("SIGTERM", () => void shutdown("SIGTERM"));
process.once("SIGINT", () => void shutdown("SIGINT"));
