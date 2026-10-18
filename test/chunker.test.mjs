/**
 * chunker.mjs — sentence splitting and chunk sizing.
 *
 * Two invariants matter. Every chunk is spoken as a complete utterance, so a
 * chunk must never end mid-sentence (that is heard as a break in the middle of
 * a thought). And a chunk only becomes playable once fully synthesized, so the
 * first chunk is one sentence and later chunks grow by a bounded factor, or
 * playback runs dry between chunks.
 */

import assert from "node:assert/strict";
import { test } from "node:test";
import { CHUNK_MAX_CHARS, RAMP_GROWTH, charSplit, sentences } from "../lib/chunker.mjs";

/** A sentence of roughly `chars` characters. */
function sentence(chars) {
	const base = "The quick brown fox jumps over the lazy dog and keeps going ";
	let s = "";
	while (s.length < chars) s += base;
	return `${s.slice(0, chars).trimEnd()}.`;
}

function paragraph(count, chars) {
	return Array.from({ length: count }, () => sentence(chars)).join(" ");
}

test("short text stays a single chunk", () => {
	assert.deepEqual(charSplit("A short reply."), ["A short reply."]);
});

test("empty or whitespace text yields no chunks", () => {
	assert.deepEqual(charSplit(""), []);
	assert.deepEqual(charSplit("   \n\n  "), []);
});

test("dots inside names, numbers and abbreviations do not end a sentence", () => {
	assert.deepEqual(sentences("The AGENTS.md updates are done at speed 1.25 on example.com today."), [
		"The AGENTS.md updates are done at speed 1.25 on example.com today.",
	]);
	assert.deepEqual(sentences("Use a tool, e.g. the browser. Then stop."), [
		"Use a tool, e.g. the browser.",
		"Then stop.",
	]);
	assert.deepEqual(sentences('He said "done." Then left! Why? Because.'), [
		'He said "done."',
		"Then left!",
		"Why?",
		"Because.",
	]);
});

test("line breaks are boundaries and CRLF is normalised", () => {
	assert.deepEqual(sentences("first line\r\nsecond line\n\nthird"), ["first line", "second line", "third"]);
});

test("no chunk ever ends mid-sentence", () => {
	const text = [paragraph(3, 40), paragraph(4, 260), paragraph(2, 700), paragraph(6, 90)].join(" ");
	const units = new Set(sentences(text));
	for (const chunk of charSplit(text)) {
		for (const piece of sentences(chunk)) assert.ok(units.has(piece), `split inside: ${piece.slice(0, 40)}`);
		assert.match(chunk, /[.!?]$/);
	}
});

test("the first chunk is the first sentence alone", () => {
	const text = `${sentence(30)} ${paragraph(5, 120)}`;
	assert.equal(charSplit(text)[0], sentence(30));
});

test("growth is bounded, except for a single sentence taken whole", () => {
	for (const size of [20, 60, 150, 400]) {
		const chunks = charSplit(paragraph(40, size));
		for (let i = 1; i < chunks.length; i++) {
			const single = sentences(chunks[i]).length === 1;
			const cap = Math.min(CHUNK_MAX_CHARS, Math.ceil(chunks[i - 1].length * RAMP_GROWTH));
			assert.ok(single || chunks[i].length <= cap, `chunk ${i} (${chunks[i].length}) exceeds ${cap}`);
		}
	}
});

test("regression: the reply that broke mid-sentence on 18/10/2026", () => {
	const text =
		"The AGENTS.md updates are done, with both copies matching and pushed. The email rule now sends through Browser Harness to Gmail with Proton as backup, there's a new Telegram notification rule for long tasks, the personal details rule moved into Defaults with no wording change, and the first Changes rule now points to the map instead of naming a section. No other reply rules were added.";
	const chunks = charSplit(text);
	assert.equal(chunks[0], "The AGENTS.md updates are done, with both copies matching and pushed.");
	assert.ok(chunks.every((c) => /[.!?]$/.test(c)));
	assert.equal(chunks.join(" "), text);
});
