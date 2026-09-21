/**
 * chunker.mjs — text splitting and chunk sizing policies.
 *
 * The sizing policy is not cosmetic: a chunk only becomes playable once it is
 * FULLY synthesized, so the streaming route needs (a) a small opening chunk so
 * the listener hears something quickly and (b) a bound on how much each chunk
 * may grow, or playback runs dry between chunks. The last test here encodes
 * that invariant directly, so a future edit to the ramp can't quietly reintroduce
 * gaps in the audio.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import {
	CHUNK_MAX_CHARS,
	FIRST_CHUNK_MAX_CHARS,
	FIRST_SENTENCE_MAX_CHARS,
	RAMP_GROWTH,
	charSplit,
	openingChunkBudget,
	rampBudget,
	rampBudgetFor,
	steadyBudget,
} from "../lib/chunker.mjs";

/** A sentence of roughly `chars` characters. */
function sentence(chars) {
	const base = "The quick brown fox jumps over the lazy dog and keeps going. ";
	let s = "";
	while (s.length < chars) s += base;
	return s.slice(0, chars).trimEnd().replace(/[,.]$/, "") + ".";
}

function paragraph(sentences, chars) {
	return Array.from({ length: sentences }, () => sentence(chars)).join(" ");
}

/** One sentence of roughly `chars` characters, with no internal punctuation. */
function singleSentence(chars) {
	const words = "the quick brown fox jumps over the lazy dog ".repeat(20);
	return `${words.slice(0, Math.max(1, chars - 6)).trimEnd()} stop.`;
}

describe_chunker();

function describe_chunker() {
	test("short text stays a single chunk under both policies", () => {
		const text = "A short reply.";
		assert.deepEqual(charSplit(text, steadyBudget), [text]);
		assert.deepEqual(charSplit(text, rampBudgetFor(text)), [text]);
	});

	test("empty and whitespace-only text produce no chunks", () => {
		assert.deepEqual(charSplit(""), []);
		assert.deepEqual(charSplit("   \n\n  "), []);
	});

	test("CRLF is normalised and the text is trimmed", () => {
		assert.deepEqual(charSplit("  hello\r\nworld  ", steadyBudget), ["hello\nworld"]);
	});

	test("steady policy never exceeds its cap and splits on sentence ends", () => {
		const text = paragraph(12, 90); // ~1080 chars, one paragraph
		const chunks = charSplit(text, steadyBudget);
		assert.ok(chunks.length >= 3, `expected several chunks, got ${chunks.length}`);
		for (const c of chunks) {
			assert.ok(c.length <= CHUNK_MAX_CHARS, `chunk of ${c.length} exceeds the cap`);
			assert.ok(/[.!?]$/.test(c), `chunk should end at a sentence boundary: ${JSON.stringify(c.slice(-40))}`);
		}
		assert.equal(chunks.join(" "), text);
	});

	test("paragraph breaks are preserved and never merged into one chunk", () => {
		const paras = [paragraph(2, 150), paragraph(2, 150), paragraph(2, 150)];
		const chunks = charSplit(paras.join("\n\n"), steadyBudget);
		assert.equal(chunks.length, 3, `expected one chunk per paragraph, got ${chunks.length}`);
		chunks.forEach((c, i) => assert.equal(c, paras[i]));
	});

	test("a sentence longer than the cap wraps on word boundaries", () => {
		const long = `${"word ".repeat(240)}end.`; // no sentence end until the very end
		for (const budget of [steadyBudget, rampBudget()]) {
			const chunks = charSplit(long, budget);
			for (const c of chunks) {
				assert.ok(c.length <= CHUNK_MAX_CHARS, `chunk of ${c.length} exceeds the cap`);
			}
			assert.equal(chunks.join(" "), long.trim());
		}
	});

	test("ramp opens at one short sentence, not a full steady chunk", () => {
		const text = paragraph(20, 80); // ~1600 chars in one paragraph
		const steady = charSplit(text, steadyBudget);
		const ramped = charSplit(text, rampBudgetFor(text));
		assert.ok(
			steady[0].length > FIRST_CHUNK_MAX_CHARS,
			"the steady opening chunk should be well over the ramped cap",
		);
		assert.ok(
			ramped[0].length <= FIRST_CHUNK_MAX_CHARS,
			`ramped first chunk is ${ramped[0].length} chars, over the ${FIRST_CHUNK_MAX_CHARS} cap`,
		);
		assert.equal(ramped.join(" "), text, "no text may be lost or duplicated");
	});

	test("ramp growth is bounded by the previous chunk's ACTUAL length", () => {
		for (const size of [60, 120, 400, 1600, 6000]) {
			const chunks = charSplit(paragraph(Math.ceil(size / 60), 60), rampBudget());
			for (let i = 1; i < chunks.length; i++) {
				const prev = chunks[i - 1].length;
				// The cap is derived from the previous chunk; the boundary logic may
				// come in under it, never over it (up to the steady ceiling).
				assert.ok(
					chunks[i].length <= Math.max(FIRST_CHUNK_MAX_CHARS, Math.ceil(prev * RAMP_GROWTH)),
					`chunk ${i} (${chunks[i].length}) is not bounded by ${prev} x ${RAMP_GROWTH}`,
				);
				assert.ok(chunks[i].length <= CHUNK_MAX_CHARS);
			}
		}
	});

	test("ramp saturates at the steady cap and stays lossless", () => {
		const text = paragraph(30, 90); // long enough to reach the ceiling
		const ramped = charSplit(text, rampBudgetFor(text));
		const steady = charSplit(text, steadyBudget);
		assert.equal(ramped.join(" "), text);
		// Ramping trades a fast start for more, smaller opening chunks...
		assert.ok(ramped.length > steady.length, "the ramp should produce more chunks");
		assert.ok(ramped[0].length <= FIRST_CHUNK_MAX_CHARS);
		// ...but the tail behaves like a steady split once the cap is reached.
		const late = ramped.slice(3);
		assert.ok(late.length > 0, "fixture should have chunks past the ramp");
		for (const c of late) {
			assert.ok(c.length > FIRST_CHUNK_MAX_CHARS, `chunk did not grow back: ${c.length}`);
			assert.ok(c.length <= CHUNK_MAX_CHARS);
		}
	});

	test("ramp growth keeps synthesis ahead of playback (the gapless invariant)", () => {
		// Kokoro synthesizes ~2.5x realtime on this box, i.e. a chunk worth A
		// seconds of audio costs 0.4*A seconds to synthesize. Chunks are emitted in
		// order, each fully synthesized, so chunk j only plays gaplessly if its
		// synthesis fits inside what has been banked by the time chunk j-1 finishes
		// playing: rtf*A(j) <= A(0) + (1-rtf)*sum(A(1..j-1)). Chars are a faithful
		// proxy for audio seconds (same voice and speed), so the check is in chars.
		const rtf = 0.4;
		const texts = [
			paragraph(4, 40), // short reply
			paragraph(10, 90), // one long paragraph
			paragraph(6, 60) + `\n\n${paragraph(6, 60)}`, // multi-paragraph
			`${sentence(30)} ${paragraph(8, 120)}`, // very short opener, then long ones
			paragraph(40, 90), // very long reply
		];
		for (const text of texts) {
			const chunks = charSplit(text, rampBudgetFor(text));
			assert.ok(chunks.length > 1, "fixture should split into several chunks");
			for (let j = 1; j < chunks.length; j++) {
				const banked =
					chunks[0].length +
					(1 - rtf) * chunks.slice(1, j).reduce((sum, c) => sum + c.length, 0);
				const needed = rtf * chunks[j].length;
				assert.ok(
					banked >= needed,
					`underflow risk at chunk ${j}: needs ${needed.toFixed(0)} of synthesis but only ` +
						`${banked.toFixed(0)} chars of audio will be banked (text: ${text.slice(0, 40)}…, ` +
						`sizes: ${chunks.map((c) => c.length).join(",")})`,
				);
			}
		}
	});

	test("the opening chunk is a whole sentence when it is short enough", () => {
		// A 119-char opening sentence against a plain 100-char budget would be
		// word-wrapped mid-sentence and leave a ~20-char orphan behind it.
		const opener = singleSentence(119);
		assert.equal(opener.match(/[.!?]/g).length, 1, "fixture must be a single sentence");
		const text = `${opener} ${paragraph(4, 90)}`;
		assert.equal(openingChunkBudget(text), opener.length);
		const chunks = charSplit(text, rampBudgetFor(text));
		assert.equal(chunks[0], opener, "the opening chunk should be the whole first sentence");
		assert.ok(chunks[1].length > 20, `second chunk is an orphan: ${JSON.stringify(chunks[1])}`);
		assert.equal(chunks.join(" "), text);
	});

	test("an opening sentence past the sentence cap falls back to the plain cap", () => {
		// One long sentence with no internal punctuation, past the cap.
		const longOpener = `${"alpha ".repeat(35)}omega.`;
		assert.ok(longOpener.length > FIRST_SENTENCE_MAX_CHARS);
		assert.equal(openingChunkBudget(`${longOpener} More text here.`), FIRST_CHUNK_MAX_CHARS);
		const chunks = charSplit(`${longOpener} ${paragraph(6, 90)}`, rampBudgetFor(longOpener));
		assert.ok(chunks[0].length <= FIRST_CHUNK_MAX_CHARS);
		// Every chunk is still a reasonable size — no tiny leftover pieces.
		for (const c of chunks) assert.ok(c.length > 30, `tiny chunk: ${c.length}`);
	});

	test("a sentence just over the cap leaves no orphan chunk", () => {
		// The case that bit the deployed ramp: a 121-char opening sentence against
		// the 120-char sentence cap word-wraps, and pushing the ~20-char tail on
		// its own made a chunk that was far too short to speak.
		const opener = singleSentence(FIRST_SENTENCE_MAX_CHARS + 1);
		const text = `${opener} ${paragraph(4, 90)}`;
		const chunks = charSplit(text, rampBudgetFor(text));
		assert.ok(chunks[0].length <= FIRST_CHUNK_MAX_CHARS);
		assert.ok(chunks[1].length > 60, `orphan chunk: ${chunks[1].length} chars`);
		assert.equal(chunks.join(" "), text);
	});

	test("openingChunkBudget handles text with no sentence end at all", () => {
		assert.equal(openingChunkBudget("one long clause with no punctuation at all"), FIRST_CHUNK_MAX_CHARS);
		assert.equal(openingChunkBudget("Short opener. Then more text follows."), 13);
	});

	test("budget helpers behave at their edges", () => {
		assert.equal(steadyBudget(), CHUNK_MAX_CHARS);
		const policy = rampBudget();
		assert.equal(policy(0), FIRST_CHUNK_MAX_CHARS);
		assert.equal(policy(-5), FIRST_CHUNK_MAX_CHARS);
		assert.equal(policy(10), Math.ceil(10 * RAMP_GROWTH));
		assert.equal(policy(10_000), CHUNK_MAX_CHARS, "growth must never pass the steady cap");
		assert.equal(rampBudget(137)(0), 137, "the opening cap is configurable");
		assert.ok(RAMP_GROWTH < 2.5, "growth must stay under the ~2.5x synthesis/playback ratio");
	});
}
