import test from "node:test";
import assert from "node:assert/strict";
import { EventEmitter } from "node:events";

import {
	HttpError,
	parseTtsRequest,
	readBody,
	writeWithBackpressure,
} from "../lib/http-utils.mjs";

test("parseTtsRequest validates text, voice, speed, and text length", () => {
	const opts = { voices: ["af_heart", "bf_emma"], defaultVoice: "af_heart", maxTextChars: 10 };
	assert.deepEqual(parseTtsRequest({ text: " hello ", speed: 0.7 }, opts), {
		text: "hello",
		voice: "af_heart",
		speed: 0.7,
	});
	assert.throws(() => parseTtsRequest({ text: "hello", voice: "unknown" }, opts), /unknown voice/);
	assert.throws(() => parseTtsRequest({ text: "hello", speed: 4 }, opts), /0.5 to 2/);
	assert.throws(() => parseTtsRequest({ text: "12345678901" }, opts), (error) => error.status === 413);
	assert.throws(() => parseTtsRequest({ text: 123 }, opts), /must be a string/);
});

test("readBody returns 413 and drains an oversized request instead of destroying it", async () => {
	const req = new EventEmitter();
	let resumed = false;
	req.resume = () => { resumed = true; };
	const result = readBody(req, 4);
	req.emit("data", Buffer.from("12345"));
	req.emit("end");
	await assert.rejects(result, (error) => error instanceof HttpError && error.status === 413);
	assert.equal(resumed, true);
});

test("writeWithBackpressure waits for drain", async () => {
	const res = new EventEmitter();
	let writes = 0;
	res.write = () => {
		writes += 1;
		return false;
	};
	let settled = false;
	const pending = writeWithBackpressure(res, Buffer.from("audio")).then(() => { settled = true; });
	await new Promise((resolve) => setImmediate(resolve));
	assert.equal(settled, false);
	res.emit("drain");
	await pending;
	assert.equal(settled, true);
	assert.equal(writes, 1);
});

test("writeWithBackpressure cancels on client close", async () => {
	const res = new EventEmitter();
	res.write = () => false;
	const pending = writeWithBackpressure(res, Buffer.from("audio"));
	res.emit("close");
	await assert.rejects(pending, (error) => error instanceof HttpError && error.status === 499);
});
