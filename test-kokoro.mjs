import { KokoroTTS } from "kokoro-js";
console.log("kokoro-js imported OK");
try {
  const t = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-v1.0-ONNX", { dtype: "q4", device: "cpu" });
  console.log("MODEL OK, voices:", Object.keys(t.voices).length);
  const audio = await t.generate("Hello world, this is a test.", { voice: "af_heart" });
  console.log("SYNTH OK, samples:", audio.audio.length, "rate:", audio.sampling_rate);
} catch (e) {
  console.log("FAIL:", e.message.split("\n").slice(0,3).join(" | "));
}
