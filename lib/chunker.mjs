/**
 * Text chunking for Kokoro synthesis.
 *
 * Kokoro's tokenizer has a fixed context window, so long text must be split
 * into chunks, and each chunk is spoken as a complete utterance: its pitch
 * falls at the end and the next chunk starts fresh. A chunk boundary inside a
 * sentence is therefore heard as a break in the middle of a thought. This
 * module only ever cuts at the end of a sentence or a line; server.mjs owns
 * the token-aware pass that splits the rare sentence too long for the model's
 * window (it needs the loaded tokenizer, so it can't live here).
 *
 * Sizing. A streamed chunk only becomes playable once it is fully synthesized,
 * so the first chunk is the first sentence alone (the listener hears something
 * quickly), and each later chunk packs whole sentences up to RAMP_GROWTH x the
 * previous chunk's length (capped at CHUNK_MAX_CHARS). Synthesis runs at about
 * 3x realtime, so a chunk up to RAMP_GROWTH x the one playing is ready before
 * that one ends. A single sentence longer than its budget is still taken
 * whole: if playback briefly runs dry there, the wait lands on a sentence end,
 * where a pause sounds natural.
 */

/** The most text packed into one chunk (single long sentences may exceed it). */
export const CHUNK_MAX_CHARS = 500;

/** Bound on per-chunk growth; under the ~3x synthesis-to-playback ratio. */
export const RAMP_GROWTH = 2.2;

/** A sentence end: terminal punctuation, optional closing quote/bracket, then a space. */
const SENTENCE_END = /(?<=[.!?]["')\]”’]*)\s+/;

/** Words whose trailing full stop is not a sentence end. */
const ABBREVIATION = /(?:^|\s)(?:e\.g|i\.e|etc|vs|approx|Mr|Mrs|Ms|Dr|St|No)\.$/i;

/**
 * Split `text` into sentences and lines. A full stop only ends a sentence when
 * whitespace follows it, so "AGENTS.md", "1.25" and "example.com" stay whole.
 */
export function sentences(text) {
	const units = [];
	for (const line of text.replace(/\r\n/g, "\n").split("\n")) {
		let pending = "";
		for (const piece of line.trim().split(SENTENCE_END)) {
			if (!piece) continue;
			pending = pending ? `${pending} ${piece}` : piece;
			if (!ABBREVIATION.test(pending)) {
				units.push(pending);
				pending = "";
			}
		}
		if (pending) units.push(pending);
	}
	return units;
}

/** Pack whole sentences into chunks under the ramped budget described above. */
export function charSplit(text) {
	const chunks = [];
	let buf = "";
	let budget = 0; // 0 → the first chunk is exactly the first sentence
	for (const s of sentences(text)) {
		if (buf && (budget === 0 || buf.length + 1 + s.length > budget)) {
			chunks.push(buf);
			budget = Math.min(CHUNK_MAX_CHARS, Math.ceil(buf.length * RAMP_GROWTH));
			buf = s;
		} else {
			buf = buf ? `${buf} ${s}` : s;
		}
	}
	if (buf) chunks.push(buf);
	return chunks;
}
