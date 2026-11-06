import { test } from "node:test";
import assert from "node:assert/strict";
import { importBatch, matchesRecording } from "./workflow.mjs";

test("a 100-file import continues after failures and counts duplicates", async () => {
  const files = Array.from({ length: 100 }, (_, i) => ({ name: `${i}.wav` }));
  const seen = [];
  const result = await importBatch(
    files,
    async (file) => {
      seen.push(file.name);
      if (file.name === "3.wav") throw new Error("Unsupported audio");
      return { duplicate: file.name === "9.wav" };
    },
    () => {},
  );
  assert.equal(seen.length, 100);
  assert.equal(result.imported, 98);
  assert.equal(result.duplicates, 1);
  assert.deepEqual(result.failures, [
    { name: "3.wav", message: "Unsupported audio" },
  ]);
});
test("stop after current file retains completed imports", async () => {
  let stop = false;
  const result = await importBatch(
    [{ name: "a" }, { name: "b" }],
    async () => {
      stop = true;
      return {};
    },
    () => {},
    () => stop,
  );
  assert.equal(result.done, 1);
  assert.equal(result.stopped, true);
});
test("metadata search matches saved section wording, not unaccepted suggestions", () => {
  const track = {
    name: "Piece.wav",
    caption: "Piano",
    family: "suite",
    bpm: "80",
    sections: [{ label: "Coda", caption: "Sparse bowed strings" }],
    suggestion: { caption: "Brass" },
  };
  assert.equal(matchesRecording(track, "STRINGS sparse"), true);
  assert.equal(matchesRecording(track, "suite 80"), true);
  assert.equal(matchesRecording(track, "brass"), false);
});
