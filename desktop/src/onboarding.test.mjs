import { test } from "node:test";
import assert from "node:assert/strict";
import { onboardingFacts, restoreOnboarding, stepPage } from "./onboarding.mjs";
const track = (id, split = "train", reviewed = false) => ({
  id,
  family: id,
  split,
  reviewed,
});
const project = {
  model_id: "sa3-small",
  tracks: [track("a"), track("b"), track("c", "validation")],
  assets: [],
};
test("fresh collections enter import; existing work is not forced into onboarding", () => {
  assert.deepEqual(restoreOnboarding({ tracks: [] }), {
    enabled: true,
    step: 0,
  });
  assert.equal(restoreOnboarding(project).enabled, false);
});
test("approval progress excludes excluded audio, never counts unreviewed included audio", () => {
  const p = {
    ...project,
    tracks: [
      ...project.tracks.map((t) => ({ ...t, reviewed: true })),
      track("excluded", "exclude"),
    ],
  };
  assert.equal(onboardingFacts(p).firstIncomplete, 3);
  p.tracks[0].reviewed = false;
  assert.equal(onboardingFacts(p).firstIncomplete, 1);
});
test("wrong-model checkpoints and MIDI exports do not complete training or audio steps", () => {
  const p = {
    ...project,
    tracks: project.tracks.map((t) => ({ ...t, reviewed: true })),
    assets: [{ path: "a.mid" }],
  };
  const facts = onboardingFacts(p, [{ model_id: "sa3-medium" }]);
  assert.equal(facts.done[3], false);
  assert.equal(facts.done[4], false);
  assert.equal(onboardingFacts(p, [{ model_id: "sa3-small" }]).done[3], true);
});
test("resume returns to changed prerequisites and remembers dismissal", () => {
  assert.deepEqual(
    restoreOnboarding({ ...project, onboarding: { enabled: true, step: 4 } }),
    { enabled: true, step: 1 },
  );
  assert.equal(
    restoreOnboarding({ ...project, onboarding: { enabled: false, step: 4 } })
      .enabled,
    false,
  );
  assert.deepEqual([0, 1, 2, 3, 4].map(stepPage), [
    "collection",
    "collection",
    "collection",
    "train",
    "compose",
  ]);
});
