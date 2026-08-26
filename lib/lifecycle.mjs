const SYSTEMD_FD_START = 3;
const DEFAULT_IDLE_TIMEOUT_MS = 10 * 60 * 1000;
const MIN_IDLE_TIMEOUT_MS = 60 * 1000;
const MAX_IDLE_TIMEOUT_MS = 24 * 60 * 60 * 1000;

export function parseIdleTimeout(
	value,
	{
		defaultMs = DEFAULT_IDLE_TIMEOUT_MS,
		minMs = MIN_IDLE_TIMEOUT_MS,
		maxMs = MAX_IDLE_TIMEOUT_MS,
	} = {},
) {
	if (value === undefined || value === "") return defaultMs;
	const parsed = Number(value);
	if (!Number.isSafeInteger(parsed) || parsed < minMs || parsed > maxMs) {
		throw new Error(`KOKORO_IDLE_TIMEOUT_MS must be an integer from ${minMs} to ${maxMs}`);
	}
	return parsed;
}

/** Resolve either the ordinary host/port listener or systemd's first passed FD. */
export function resolveListenTarget({ env = process.env, pid = process.pid, host, port }) {
	const listenPidRaw = env.LISTEN_PID;
	const listenFdsRaw = env.LISTEN_FDS;
	if (listenPidRaw === undefined && listenFdsRaw === undefined) {
		return { options: { host, port }, source: `http://${host}:${port}` };
	}
	if (listenPidRaw === undefined || listenFdsRaw === undefined) {
		throw new Error("incomplete systemd socket activation environment");
	}

	const listenPid = Number(listenPidRaw);
	if (!Number.isSafeInteger(listenPid) || listenPid <= 0) {
		throw new Error("invalid LISTEN_PID");
	}
	// Match sd_listen_fds(): inherited activation variables for another process
	// are ignored instead of causing this process to adopt an unrelated FD.
	if (listenPid !== pid) {
		return { options: { host, port }, source: `http://${host}:${port}` };
	}

	const listenFds = Number(listenFdsRaw);
	if (listenFds !== 1) {
		throw new Error(`expected exactly one systemd socket, received ${listenFdsRaw}`);
	}
	if (env.LISTEN_FDNAMES && env.LISTEN_FDNAMES !== "tts") {
		throw new Error(`unexpected systemd socket name: ${env.LISTEN_FDNAMES}`);
	}
	return { options: { fd: SYSTEMD_FD_START }, source: "systemd socket fd 3" };
}

/**
 * Arm shutdown only while the caller reports no active or queued work.
 * Health/voice-list requests deliberately do not reset the timer; synthesis
 * activity calls workStarted/workFinished around the complete serial queue.
 */
export function createIdleShutdownTimer({
	timeoutMs,
	isIdle,
	onIdle,
	setTimer = setTimeout,
	clearTimer = clearTimeout,
}) {
	let timer = null;
	let stopped = false;

	const disarm = () => {
		if (timer === null) return;
		clearTimer(timer);
		timer = null;
	};
	const arm = () => {
		disarm();
		if (stopped || !isIdle()) return;
		timer = setTimer(() => {
			timer = null;
			if (stopped) return;
			if (!isIdle()) {
				arm();
				return;
			}
			onIdle();
		}, timeoutMs);
		timer?.unref?.();
	};

	return {
		start: arm,
		workStarted: disarm,
		workFinished: arm,
		stop() {
			stopped = true;
			disarm();
		},
	};
}
