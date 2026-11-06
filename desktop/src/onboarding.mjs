export const stepPage = (step) =>
  step < 3 ? "collection" : step === 3 ? "train" : "compose";

export function onboardingFacts(project, checkpoints = [], readiness = null) {
  const tracks = project?.tracks ?? [];
  const included = tracks.filter((t) => t.split !== "exclude");
  const reviewed = included.filter((t) => t.reviewed).length;
  const held = included.filter((t) => t.split === "validation").length;
  const training = included.filter((t) => t.split === "train").length;
  const hasVersion = checkpoints.some((c) => c.model_id === project?.model_id);
  const done = [
    tracks.length > 0,
    included.length > 0 && reviewed === included.length,
    held > 0 && training > 0,
    hasVersion,
    (project?.assets ?? []).some((a) => a.path?.endsWith(".wav")),
  ];
  const first = done.findIndex((v) => !v);
  return {
    done,
    firstIncomplete: first < 0 ? 4 : first,
    count: tracks.length,
    included: included.length,
    reviewed,
    held,
    training,
    families: new Set(included.map((t) => t.family)).size,
    ready: !!readiness?.ready,
  };
}

export function restoreOnboarding(project, checkpoints = [], readiness = null) {
  const saved = project?.onboarding;
  const facts = onboardingFacts(project, checkpoints, readiness);
  if (!saved)
    return { enabled: !project?.tracks?.length, step: facts.firstIncomplete };
  return {
    enabled: saved.enabled === true,
    step:
      Number.isInteger(saved.step) && saved.step >= 0 && saved.step <= 4
        ? Math.min(saved.step, facts.firstIncomplete)
        : facts.firstIncomplete,
  };
}
