/**
 * Text chunking for Kokoro synthesis.
 *
 * Kokoro's tokenizer has a fixed context window, so long text must be split
 * into chunks. This module owns the pure char-level split and the two sizing
 * policies; server.mjs owns the token-aware pass that re-splits anything whose
 * true phoneme-token count would overflow the window (it needs the loaded
 * tokenizer, so it can't live here).
 *
 * Steady vs ramped:
 *   - `steadyBudget` is a uniform cap at CHUNK_MAX_CHARS. The whole-blob route
 *     uses it: it waits for every chunk before responding, so a small opening
 *     chunk would buy nothing.
 *   - `rampBudget` opens small and grows. The streaming route uses it, because
 *     the listener is waiting in silence until the FIRST chunk is synthesized
 *     (~0.4s of CPU per second of audio, so a 500-char opening chunk — roughly
 *     29s of audio — costs ~12s of silence), and a ~100-char opening chunk (one
 *     sentence, where the text has one that short) costs ~2s.
 *
 * Why the growth is bounded — the part that is easy to get wrong: a chunk is
 * sent only once it is FULLY synthesized, so playback runs dry whenever the
 * next chunk needs longer to synthesize than the current one takes to play.
 * Synthesis runs at ~2.5x realtime, which means a chunk's playback lasts about
 * 2.5x as long as its own synthesis, so the next chunk may be at most ~2.5x its
 * length. Past the first pair the constraint relaxes (every played chunk banks
 * slack, since synthesis outruns playback), so a single bounded growth step per
 * chunk is enough. RAMP_GROWTH stays under the 2.5x with margin.
 *
 * Note the growth is measured against the PREVIOUS CHUNK'S ACTUAL LENGTH rather
 * than its budget: a sentence boundary can end a chunk well short of its budget
 * (e.g. a 30-char opener), and jumping from a 30-char chunk straight to a
 * 500-char one is exactly the case that underflows.
 */

/** Steady cap: the most text we aim to put in one chunk. */
export const CHUNK_MAX_CHARS = 500;

/** Opening chunk for the ramped policy: ~2s of synthesis on this box. */
export const FIRST_CHUNK_MAX_CHARS = 100;

/**
 * Cap for the opening chunk when the text's first sentence fits under it. A
 * complete sentence is a much better join than a mid-sentence cut, and it also
 * avoids the orphan that a plain char budget produces: a 119-char opening
 * sentence against a 100-char budget gets word-wrapped into 99 chars plus a
 * 20-char leftover. Longer than this and the sentence is wrapped as usual.
 * 120 chars is ~2.5s of synthesis (measured ~0.021s/char on this box), keeping
 * the opening sentence — when we take one whole — within sight of the ~2s that
 * FIRST_CHUNK_MAX_CHARS targets.
 */
export const FIRST_SENTENCE_MAX_CHARS = 120;

/** Bound on per-chunk growth (see the module comment); < the ~2.5x realtime ratio. */
export const RAMP_GROWTH = 2.2;

/** Uniform cap (whole-blob route). */
export function steadyBudget() {
	return CHUNK_MAX_CHARS;
}

/**
 * Cap for the opening chunk: the first sentence when it is short enough to
 * speak on its own, else the plain opening cap (see FIRST_SENTENCE_MAX_CHARS).
 */
export function openingChunkBudget(text) {
	const clean = text.replace(/\r\n/g, "\n").trim();
	const firstSentence = /^[^.!?]*[.!?]+(?:["')\]]+)?/.exec(clean)?.[0]?.trim();
	if (firstSentence && firstSentence.length <= FIRST_SENTENCE_MAX_CHARS) {
		return firstSentence.length;
	}
	return FIRST_CHUNK_MAX_CHARS;
}

/**
 * Ramped policy (streaming route): the opening chunk gets `openingChars`, and
 * each later one gets RAMP_GROWTH x the previous chunk's ACTUAL length, up to
 * the steady cap. Returned as a `(lastChunkChars) => cap` policy — the length
 * argument is 0 for the first chunk.
 *
 * Growth is measured against the previous chunk's real length rather than its
 * budget because a sentence boundary can end a chunk well short of its budget,
 * and that is exactly when a fixed next-budget over-reaches into underflow.
 */
export function rampBudget(openingChars = FIRST_CHUNK_MAX_CHARS) {
	return (lastChunkChars) =>
		lastChunkChars <= 0
			? openingChars
			: Math.min(CHUNK_MAX_CHARS, Math.ceil(lastChunkChars * RAMP_GROWTH));
}

/** Ramped policy for `text`, sized from its opening sentence. */
export function rampBudgetFor(text) {
	return rampBudget(openingChunkBudget(text));
}

/**
 * Split `text` on natural boundaries (paragraph → sentence → word) so that no
 * chunk exceeds the budget the policy hands out at that point. `nextBudget` is
 * called with the length of the chunk just emitted (0 before the first) and
 * returns the cap for the next one — see steadyBudget/rampBudget.
 *
 * Boundary logic is deliberately unchanged from the single-cap version this
 * replaces, so the steady policy produces byte-identical chunking.
 */
export function charSplit(text, nextBudget = steadyBudget) {
	const clean = text.replace(/\r\n/g, "\n").trim();
	if (!clean) return [];

	const chunks = [];
	let budget = nextBudget(0);
	if (clean.length <= budget) return [clean];

	const pushIf = (s) => {
		const t = s.trim();
		if (!t) return;
		chunks.push(t);
		budget = nextBudget(t.length);
	};

	// Split on blank lines (paragraphs) first.
	for (const para of clean.split(/\n\s*\n/)) {
		if (para.length <= budget) {
			pushIf(para);
			continue;
		}
		// Paragraph too big: split on sentence enders, keeping punctuation.
		const sentences = para.match(/[^.!?]*[.!?]+(?:["')\]]+)?|[^.!?]+$/g) ?? [para];
		let buf = "";
		for (const sent of sentences) {
			const s = sent.trim();
			if (!s) continue;
			if (s.length > budget) {
				// Single sentence longer than budget: flush what we have, then hard-wrap
				// by word. The wrapped tail becomes the START of the next chunk rather
				// than a chunk of its own — a sentence barely over the budget would
				// otherwise leave a ~20-char orphan behind it (a 121-char opening
				// sentence against a 120-char opening cap did exactly that).
				if (buf) {
					pushIf(buf);
					buf = "";
				}
				let wbuf = "";
				for (const w of s.split(/\s+/)) {
					if ((wbuf + " " + w).trim().length > budget) {
						pushIf(wbuf);
						wbuf = w;
					} else {
						wbuf = (wbuf + " " + w).trim();
					}
				}
				buf = wbuf;
				continue;
			}
			if ((buf + " " + s).length > budget) {
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
