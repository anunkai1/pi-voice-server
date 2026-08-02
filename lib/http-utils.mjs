export class HttpError extends Error {
	constructor(status, message) {
		super(message);
		this.status = status;
	}
}

export const MAX_BODY_BYTES = 128 * 1024;
export const MAX_TEXT_CHARS = 30_000;

export function readBody(req, maxBytes = MAX_BODY_BYTES) {
	return new Promise((resolve, reject) => {
		const chunks = [];
		let size = 0;
		let settled = false;
		const fail = (error) => {
			if (settled) return;
			settled = true;
			reject(error);
		};
		req.on("data", (chunk) => {
			if (settled) return;
			size += chunk.length;
			if (size > maxBytes) {
				fail(new HttpError(413, `request body too large (max ${maxBytes} bytes)`));
				// Drain rather than destroy so the server can return the 413 cleanly.
				req.resume?.();
				return;
			}
			chunks.push(chunk);
		});
		req.on("end", () => {
			if (settled) return;
			settled = true;
			resolve(Buffer.concat(chunks).toString("utf8"));
		});
		req.on("aborted", () => fail(new HttpError(499, "client disconnected")));
		req.on("error", fail);
	});
}

export function parseTtsRequest(body, { voices, defaultVoice, maxTextChars = MAX_TEXT_CHARS }) {
	if (!body || typeof body !== "object" || Array.isArray(body)) {
		throw new HttpError(400, "request body must be a JSON object");
	}
	if (typeof body.text !== "string") throw new HttpError(400, "'text' must be a string");
	const text = body.text.trim();
	if (!text) throw new HttpError(400, "missing 'text'");
	if (text.length > maxTextChars) {
		throw new HttpError(413, `text too long (max ${maxTextChars} characters)`);
	}
	const voice = body.voice === undefined ? defaultVoice : body.voice;
	if (typeof voice !== "string" || !voices.includes(voice)) {
		throw new HttpError(400, "unknown voice");
	}
	const speed = body.speed === undefined ? 1 : Number(body.speed);
	if (!Number.isFinite(speed) || speed < 0.5 || speed > 2) {
		throw new HttpError(400, "speed must be a number from 0.5 to 2");
	}
	return { text, voice, speed };
}

export async function writeWithBackpressure(res, chunk, isGone = () => false) {
	if (isGone()) throw new HttpError(499, "client disconnected");
	if (res.write(chunk)) return;
	await new Promise((resolve, reject) => {
		const cleanup = () => {
			res.off("drain", onDrain);
			res.off("close", onClose);
			res.off("error", onError);
		};
		const onDrain = () => {
			cleanup();
			resolve();
		};
		const onClose = () => {
			cleanup();
			reject(new HttpError(499, "client disconnected"));
		};
		const onError = (error) => {
			cleanup();
			reject(error);
		};
		res.once("drain", onDrain);
		res.once("close", onClose);
		res.once("error", onError);
	});
}
