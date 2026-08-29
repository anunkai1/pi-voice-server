import test from "node:test";
import assert from "node:assert/strict";

import {
	createIdleShutdownTimer,
	modelHealthState,
	parseIdleTimeout,
	resolveListenTarget,
} from "../lib/lifecycle.mjs";

test("resolveListenTarget uses host/port without matching activation variables", () => {
	assert.deepEqual(
		resolveListenTarget({ env: {}, pid: 42, host: "127.0.0.1", port: 8181 }),
		{ options: { host: "127.0.0.1", port: 8181 }, source: "http://127.0.0.1:8181" },
	);
	assert.deepEqual(
		resolveListenTarget({
			env: { LISTEN_PID: "41", LISTEN_FDS: "1", LISTEN_FDNAMES: "tts" },
			pid: 42,
			host: "127.0.0.1",
			port: 8181,
		}),
		{ options: { host: "127.0.0.1", port: 8181 }, source: "http://127.0.0.1:8181" },
	);
});

test("resolveListenTarget adopts exactly the named systemd socket", () => {
	assert.deepEqual(
		resolveListenTarget({
			env: { LISTEN_PID: "42", LISTEN_FDS: "1", LISTEN_FDNAMES: "tts" },
			pid: 42,
			host: "127.0.0.1",
			port: 8181,
		}),
		{ options: { fd: 3 }, source: "systemd socket fd 3" },
	);
	assert.throws(
		() => resolveListenTarget({ env: { LISTEN_PID: "42" }, pid: 42, host: "x", port: 1 }),
		/incomplete/,
	);
	assert.throws(
		() => resolveListenTarget({
			env: { LISTEN_PID: "42", LISTEN_FDS: "2" }, pid: 42, host: "x", port: 1,
		}),
		/exactly one/,
	);
	assert.throws(
		() => resolveListenTarget({
			env: { LISTEN_PID: "42", LISTEN_FDS: "1", LISTEN_FDNAMES: "wrong" },
			pid: 42,
			host: "x",
			port: 1,
		}),
		/unexpected systemd socket name/,
	);
});

test("model health separates capability from residency", () => {
	assert.deepEqual(modelHealthState({ resident: false, loading: false }), {
		modelAvailable: true,
		modelLoaded: false,
		modelResident: false,
		modelLoading: false,
	});
	assert.deepEqual(modelHealthState({ resident: true, loading: false }), {
		modelAvailable: true,
		modelLoaded: true,
		modelResident: true,
		modelLoading: false,
	});
});

test("parseIdleTimeout applies a bounded ten-minute default", () => {
	assert.equal(parseIdleTimeout(undefined), 600_000);
	assert.equal(parseIdleTimeout("120000"), 120_000);
	assert.throws(() => parseIdleTimeout("59999"), /must be an integer/);
	assert.throws(() => parseIdleTimeout("not-a-number"), /must be an integer/);
});

test("idle shutdown timer is disarmed for work and rearmed after the queue drains", () => {
	let active = 0;
	let idleCalls = 0;
	let nextId = 0;
	const timers = new Map();
	const cleared = [];
	const timer = createIdleShutdownTimer({
		timeoutMs: 600_000,
		isIdle: () => active === 0,
		onIdle: () => { idleCalls += 1; },
		setTimer: (fn, delay) => {
			const handle = { id: ++nextId, delay, fn, unref() {} };
			timers.set(handle.id, handle);
			return handle;
		},
		clearTimer: (handle) => {
			cleared.push(handle.id);
			timers.delete(handle.id);
		},
	});

	timer.start();
	assert.equal(timers.size, 1);
	assert.equal([...timers.values()][0].delay, 600_000);
	active += 1;
	timer.workStarted();
	assert.equal(timers.size, 0);
	assert.deepEqual(cleared, [1]);
	active -= 1;
	timer.workFinished();
	assert.equal(timers.size, 1);
	const armed = [...timers.values()][0];
	armed.fn();
	assert.equal(idleCalls, 1);
	timer.stop();
});
