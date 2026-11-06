<script lang="ts">
  import { onMount, tick } from "svelte";
  import TrainingPreview from "./TrainingPreview.svelte";
  import MetadataReview from "./MetadataReview.svelte";
  import CollectionPicker from "./CollectionPicker.svelte";
  import Icon from "./Icon.svelte";
  import WorkflowGuide from "./WorkflowGuide.svelte";
  import {
    onboardingFacts,
    restoreOnboarding,
    stepPage,
  } from "./onboarding.mjs";
  import { matchesRecording, importBatch } from "./workflow.mjs";
  let guided = $state(false);
  let guideStep = $state(0);
  let guideProject = "";
  async function moveGuide(step: number) {
    if (busy || reviewDirty || !!active) return;
    guideStep = step;
    if (step === 4 && !adapter)
      adapter =
        checkpoints.find((c) => c.model_id === project?.model_id)?.id ?? "";
    page = stepPage(step);
    search = "";
    reviewFilter = "all";
    await save({ onboarding: { enabled: true, step } });
    await tick();
    document.querySelector<HTMLElement>(".workflow-guide h1")?.focus();
  }
  async function leaveGuide(next = page) {
    if (busy || reviewDirty) return;
    guided = false;
    await save({ onboarding: { enabled: false, step: guideStep } });
    await go(next);
  }
  async function resumeGuide() {
    if (busy || reviewDirty || !!active) return;
    guided = true;
    await moveGuide(guideFacts.firstIncomplete);
  }
  let readiness = $state<any>(null);
  let status = $state("");
  let importState = $state<any>(null);
  let stopImport = $state(false);
  let readinessKey = "";
  let previousActiveId = "";
  async function go(next: string) {
    if (reviewDirty || busy) return;
    if (guided && next !== "setup") {
      guideStep = next === "train" ? 3 : next === "compose" ? 4 : 1;
    }
    page = next;
    await tick();
    document.querySelector<HTMLElement>("main h1")?.focus();
  }
  async function createCollection() {
    if (reviewDirty || busy) return;
    await act(async () => {
      const p = await api("projects", { name: name.trim() || "My recordings" });
      pid = p.id;
      name = "";
      search = "";
      reviewFilter = "all";
      reviewId = "";
      source = "";
      adapter = "";
      trainingAdapter = "";
      checkedTracks = [];
      newCollectionOpen = false;
      page = "collection";
      importState = null;
    });
    await tick();
    document.querySelector<HTMLElement>("main h1")?.focus();
  }
  async function reviewPending() {
    search = "";
    reviewFilter = "pending";
    reviewId = project?.tracks.find((t) => !t.reviewed)?.id ?? reviewId;
    await go("collection");
    await tick();
    const heading = document.querySelector<HTMLElement>(".metadata-editor h2");
    heading?.focus();
    heading?.scrollIntoView({ block: "start" });
  }
  async function comparisonPieces() {
    if (project?.tracks.some((t) => t.split === "validation")) {
      search = "";
      reviewFilter = "comparison";
      reviewId = project.tracks.find((t) => t.split === "validation")?.id ?? "";
      await go("collection");
    } else await act(() => api("split", { project_id: pid }));
  }

  let search = $state("");
  let newCollectionOpen = $state(false);
  const pageNames: Record<string, string> = {
    collection: "Recordings",
    compose: "Compose",
    train: "Personal sound",
    setup: "Settings",
  };
  function displayName(name: string) {
    return name
      .replace(/^\d+_copilot\.mp3\s+/i, "")
      .replace(/\.(wav|mp3|flac|aiff?|ogg)$/i, "");
  }
  function duration(value: number) {
    return `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, "0")}`;
  }
  let taggerAccepted = $state(false);
  let reviewDirty = $state(false);
  let bulkText = $state("");
  let reviewId = $state("");
  let reviewFilter = $state("all");
  let checkedTracks = $state<string[]>([]);
  type Item = {
    reviewed?: boolean;
    metadata_revision?: number;
    id: string;
    name: string;
    path: string;
    duration: number;
    peaks?: number[];
    caption?: string;
    sections?: { label: string; caption: string }[];
    family?: string;
    split?: string;
    bpm?: string;
    suggested_bpm?: number;
    adapter_id?: string;
    seed?: number;
    notes?: { pitch: number; start: number; end: number }[];
  };
  type Project = {
    id: string;
    updated: number;
    name: string;
    description: string;
    profile: string;
    model_id: string;
    tracks: Item[];
    assets: Item[];
    selected_asset: string;
    onboarding?: { enabled: boolean; step: number };
    selection_history?: (string | null)[];
  };
  let studio = $state<any>(null),
    pid = $state(""),
    page = $state("collection"),
    error = $state(""),
    busy = $state(false);
  let fileInput = $state<HTMLInputElement>();
  let name = $state(""),
    prompt = $state(""),
    seconds = $state(15),
    seed = $state(42),
    steps = $state(1000),
    adapter = $state(""),
    trainingAdapter = $state(""),
    strength = $state(1);
  let source = $state(""),
    start = $state(0),
    end = $state(10),
    bpm = $state(120),
    low = $state(36),
    high = $state(96),
    quantize = $state(false);
  let checkpoints = $state<any[]>([]),
    accepted = $state(false),
    hfToken = $state("");
  const token =
    document.querySelector<HTMLMetaElement>('meta[name="ravi-token"]')
      ?.content ?? "";
  let project = $derived<Project | undefined>(
    studio?.projects.find((p: Project) => p.id === pid),
  );
  let guideFacts = $derived(onboardingFacts(project, checkpoints, readiness));
  let startingStep = $derived(
    checkpoints.find((c) => c.id === trainingAdapter)?.step ?? 0,
  );
  let validTrainingTarget = $derived(
    Number.isInteger(steps) && steps > startingStep && steps <= 10000,
  );
  let modelReady = $derived(
    studio?.models.find((m: any) => m.id === project?.model_id)?.ready ?? false,
  );
  let active = $derived(
    studio?.jobs.find((j: any) =>
      ["queued", "running", "cancelling"].includes(j.status),
    ),
  );
  let items = $derived(
    project
      ? [
          ...project.tracks,
          ...project.assets.filter((a) => !a.path.endsWith(".mid")),
        ]
      : [],
  );
  let selected = $derived(items.find((a) => a.id === source));
  async function api(path: string, body?: unknown) {
    const r = await fetch("/api/" + path, {
      method: body ? "POST" : "GET",
      headers: {
        "X-Ravi-Token": token,
        ...(body ? { "Content-Type": "application/json" } : {}),
      },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Operation failed");
    return data;
  }
  async function refresh() {
    studio = await api("state");
    if (previousActiveId) {
      const finished = studio.jobs.find((j: any) => j.id === previousActiveId);
      if (
        finished &&
        !["queued", "running", "cancelling"].includes(finished.status)
      ) {
        status = `${finished.kind}: ${finished.status}. ${finished.message}`;
        previousActiveId = "";
      }
    }
    const running = studio.jobs.find((j: any) =>
      ["queued", "running", "cancelling"].includes(j.status),
    );
    if (running) previousActiveId = running.id;
    if (!pid && studio.projects.length) pid = studio.projects[0].id;
    if (pid) {
      checkpoints = await api("checkpoints?project_id=" + pid);
      const current = studio.projects.find((p: Project) => p.id === pid);
      const nextKey = `${pid}:${current?.updated}`;
      if (nextKey !== readinessKey) {
        readiness = null;
        const result = await api("readiness?project_id=" + pid);
        if (pid === result.project_id) {
          readiness = result;
          readinessKey = nextKey;
        }
      }
      if (current && guideProject !== pid) {
        const restored = restoreOnboarding(current, checkpoints, readiness);
        guided = restored.enabled;
        guideStep = restored.step;
        guideProject = pid;
        page = guided ? stepPage(guideStep) : "collection";
        if (guided && !current.onboarding) {
          await api("project", {
            project_id: pid,
            onboarding: { enabled: true, step: guideStep },
          });
        }
      }
      if (current && !current.tracks.some((t: Item) => t.id === reviewId))
        reviewId = current.tracks[0]?.id ?? "";
    }
  }
  async function act(fn: () => Promise<unknown>) {
    if (busy) return;
    busy = true;
    error = "";
    try {
      await fn();
      await refresh();
    } catch (e) {
      error = String(e instanceof Error ? e.message : e);
    } finally {
      busy = false;
    }
  }
  async function save(changes: object) {
    await act(() => api("project", { project_id: pid, ...changes }));
  }
  async function job(kind: string, extra: object = {}) {
    await act(() =>
      api("jobs", {
        kind,
        project_id: pid,
        prompt,
        seconds,
        seed,
        steps,
        adapter_id: kind === "train" ? trainingAdapter : adapter,
        strength,
        source_id: source,
        start,
        end,
        bpm,
        min_note: low,
        max_note: high,
        quantize,
        ...extra,
      }),
    );
  }
  let visibleTracks = $derived(
    (project?.tracks ?? []).filter(
      (t) =>
        (reviewFilter === "all" ||
          (reviewFilter === "comparison"
            ? t.split === "validation"
            : reviewFilter === "reviewed"
              ? t.reviewed
              : !t.reviewed)) &&
        matchesRecording(t, search),
    ),
  );
  function url(a: Item, download = false) {
    return (
      "/api/asset?" +
      new URLSearchParams({
        project_id: pid,
        asset_id: a.id,
        token,
        ...(download ? { download: "1" } : {}),
      })
    );
  }
  async function upload(event: Event) {
    const input = event.target as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    if (!files.length || busy) return;
    const destination = pid;
    busy = true;
    error = "";
    stopImport = false;
    try {
      importState = await importBatch(
        files,
        async (file: File) => {
          const r = await fetch(
            "/api/import?" +
              new URLSearchParams({ project_id: destination, name: file.name }),
            { method: "POST", headers: { "X-Ravi-Token": token }, body: file },
          );
          const data = await r.json();
          if (!r.ok) throw new Error(data.error || "Import failed");
          return data;
        },
        (progress: any) => (importState = progress),
        () => stopImport,
      );
      status = `Import finished: ${importState.imported} added, ${importState.duplicates} duplicates, ${importState.failures.length} failed.`;
    } catch (e) {
      error = String(e);
    } finally {
      input.value = "";
      try {
        await refresh();
      } catch (e) {
        error = String(e);
      }
      busy = false;
    }
  }
  onMount(() => {
    void act(refresh);
    const timer = setInterval(() => {
      if (!busy) refresh().catch((e) => (error = String(e)));
    }, 2500);
    return () => clearInterval(timer);
  });
</script>

<svelte:head><title>rAvI, Composer studio</title></svelte:head>
<a class="skip-link" href="#main">Skip to workspace</a>
<div class="shell">
  <aside>
    <div class="brand">rAvI<span>Composer studio</span></div>
    <CollectionPicker
      projects={studio?.projects ?? []}
      value={pid}
      disabled={reviewDirty || busy}
      select={(id) => {
        pid = id;
        search = "";
        reviewFilter = "all";
        reviewId = "";
        checkedTracks = [];
        importState = null;
        source = "";
        adapter = "";
        trainingAdapter = "";
        void act(refresh);
      }}
    />
    <details class="new-collection" bind:open={newCollectionOpen}>
      <summary>New collection</summary>
      <form
        onsubmit={(e) => {
          e.preventDefault();
          void createCollection();
        }}
      >
        <input
          aria-label="New collection name"
          placeholder="New collection name"
          bind:value={name}
        /><button disabled={!name.trim() || busy || reviewDirty}
          >Create collection</button
        >
      </form>
    </details>
    {#if !guided}<nav aria-label="Studio">
        {#each [["collection", "Recordings"], ["compose", "Compose"], ["train", "Personal sound"], ["setup", "Settings"]] as [key, label]}<button
            disabled={reviewDirty || busy}
            class:current={page === key}
            aria-current={page === key ? "page" : undefined}
            onclick={() => go(key)}
            ><Icon name={key} /><span>{label}</span></button
          >{/each}
      </nav>
      {#if project}<button disabled={busy || reviewDirty} onclick={resumeGuide}
          >Guided setup</button
        >{/if}
    {:else}<p class="aside-note">Your first personal sound</p>
      <button
        disabled={busy || reviewDirty}
        onclick={() => leaveGuide("collection")}>Leave guided setup</button
      >{/if}
    <p class="aside-note">
      <Icon name="local" /> Saved on this Mac.<br />Works offline after setup.
    </p>
  </aside>
  <main id="main" tabindex="-1">
    <p class="visually-hidden" role="status" aria-atomic="true">{status}</p>
    <header>
      <span>{project?.name ?? "No collection selected"}</span><span
        class="header-location">{pageNames[page]}</span
      >
    </header>
    {#if error}<div class="error" role="alert">
        {error}<button onclick={() => (error = "")} aria-label="Dismiss error"
          >×</button
        >
      </div>{/if}
    {#if active}<section class="job" aria-label="Current local job">
        <div>
          <strong role="status">{active.message}</strong>
          <p>
            {Math.round(active.elapsed_seconds)} seconds elapsed{#if active.step},
              update {active.step} of {active.steps}{/if}
          </p>
        </div>
        <progress
          aria-label="Local job progress"
          max="1"
          value={active.progress}
        ></progress><button
          onclick={() => act(() => api("cancel", { id: active.id }))}
          disabled={active.status === "cancelling"}>Stop</button
        >
      </section>{/if}
    {#if project && guided}
      <WorkflowGuide
        setup={page === "setup"}
        step={guideStep}
        facts={guideFacts}
        disabled={busy || reviewDirty || !!active}
        dirty={reviewDirty}
        move={moveGuide}
        finish={() => leaveGuide("compose")}
      />
      {#if page === "setup"}<button
          disabled={busy || reviewDirty}
          onclick={() => moveGuide(guideStep)}>Back to guided setup</button
        >{/if}
    {/if}
    {#if project && !modelReady && ["train", "compose"].includes(page)}<div
        class="notice"
      >
        The selected model is not ready on this computer. <button
          onclick={() => go("setup")}>Set up model</button
        >
      </div>{/if}
    {#if !studio}<div class="loading-state" aria-live="polite">
        <h1 tabindex="-1">Opening your studio…</h1>
        <div class="skeleton"></div>
        <div class="skeleton short"></div>
      </div>
    {:else if !project && page !== "setup"}<section class="welcome">
        <h1 tabindex="-1">Make a personal sound from your recordings</h1>
        <p class="lede">
          Turn your recordings into new passages. We'll guide you one step at a
          time.
        </p>
        <form
          onsubmit={(e) => {
            e.preventDefault();
            void createCollection();
          }}
        >
          <details>
            <summary>Name this collection (optional)</summary><label>
              Collection name<input
                bind:value={name}
                placeholder="My recordings"
                maxlength="120"
              />
            </label>
          </details>
          <button class="primary" disabled={busy}>Get started</button>
        </form>
        <p>Your audio stays on this computer.</p>
        <details>
          <summary>What you'll need</summary>
          <p>
            A few recordings to begin. Training needs a local model download;
            some models require an account and license acceptance.
          </p>
        </details>
      </section>
    {:else if project && page === "collection"}
      {#if !guided}<div class="page-title">
          <div>
            <h1 tabindex="-1">Recordings</h1>
            <p class="lede">
              Listen, describe, and mark the passages that matter.
            </p>
          </div>
          <button
            class="primary"
            onclick={() => fileInput?.click()}
            disabled={busy || reviewDirty}><Icon name="add" /> Add audio</button
          >
        </div>{/if}
      {#if importState}<section class="import-status" aria-label="Audio import">
          <p role="status">
            {importState.done} of {importState.total} files processed. {importState.imported}
            added, {importState.duplicates} duplicates, {importState.failures
              .length} failed.
          </p>
          {#if busy}<p>{importState.current}</p>
            <progress
              aria-label="Files processed"
              max={importState.total}
              value={importState.done}
            ></progress><button
              onclick={() => (stopImport = true)}
              disabled={stopImport}>Stop after this file</button
            >{/if}
          {#if importState.failures.length}<details>
              <summary
                >Files that need attention ({importState.failures
                  .length})</summary
              >
              <ul>
                {#each importState.failures as failure}<li>
                    {failure.name}: {failure.message}
                  </li>{/each}
              </ul>
              <p>
                Correct these files and add them again. Completed imports are
                safe; exact duplicates are skipped.
              </p>
            </details>{/if}
        </section>{/if}
      {#if guided && guideStep === 0}
        <section class="guided-action">
          <button
            class="primary"
            disabled={busy || reviewDirty}
            onclick={() => fileInput?.click()}>Choose audio files</button
          >
          <p>Your originals stay untouched.</p>
          <small
            >WAV, AIFF, FLAC, MP3 or OGG. Select several files together.</small
          >
        </section>
      {/if}
      {#if guided && guideStep === 2}
        <section class="guided-action">
          <p>
            {guideFacts.held}
            {guideFacts.held === 1 ? "recording" : "recordings"} set aside; {guideFacts.training}
            available for training.
          </p>
          <button
            class="primary"
            disabled={busy || !!active || guideFacts.families < 3}
            onclick={() => act(() => api("split", { project_id: pid }))}
          >
            {guideFacts.held
              ? "Choose comparisons again"
              : "Set aside comparison pieces"}</button
          >
          {#if guideFacts.families < 3}<p>
              Add at least three different pieces to use automatic selection.
            </p>{/if}
          <details>
            <summary>Choose which pieces are held out</summary>
            <p>
              Keep alternate takes in the same piece family. Review their roles
              in Recording details and use.
            </p>
            <button disabled={busy} onclick={() => moveGuide(1)}
              >Review recording roles</button
            >
            <ul>
              {#each project.tracks.filter((t) => t.split === "validation") as track}<li
                >
                  {displayName(track.name)}
                </li>{/each}
            </ul>
          </details>
        </section>
      {/if}
      <section class="collection-tools">
        <input
          bind:this={fileInput}
          hidden
          type="file"
          accept=".wav,.aif,.aiff,.flac,.mp3,.ogg"
          multiple
          onchange={upload}
        />
        {#if !guided}<details class="collection-notes">
            <summary>Collection notes and tools</summary>
            <label
              >Collection name<input
                disabled={busy || reviewDirty}
                value={project.name}
                onchange={(e) => {
                  if (e.currentTarget.value.trim())
                    void save({ name: e.currentTarget.value.trim() });
                }}
              /></label
            >
            <p class="muted">
              WAV, AIFF, FLAC, MP3, OGG. Originals are retained.
            </p>
            <label class="spaced"
              >Notes for this collection<textarea
                disabled={busy || reviewDirty}
                value={project.description}
                placeholder="For example: intimate chamber textures, bowed strings, sparse piano, slow harmonic movement…"
                onchange={(e) => save({ description: e.currentTarget.value })}
              ></textarea></label
            >
            <div class="row">
              <button
                disabled={busy ||
                  !project.tracks.length ||
                  !!active ||
                  !studio.tagger?.ready ||
                  reviewDirty}
                onclick={() => job("analyze")}>Suggest descriptions</button
              ><button
                disabled={busy || reviewDirty}
                onclick={() => act(() => api("split", { project_id: pid }))}
                >Set aside comparison pieces</button
              >
            </div>
            <p class="muted">
              {project.tracks.filter((t) => t.reviewed).length} / {project
                .tracks.length} reviewed. Suggestions leave your edits intact. {#if !studio.tagger?.ready}Install
                the optional description helper in Settings, or describe pieces
                manually.{/if}
            </p>
            <p class="muted">
              Collection notes are not used as training descriptions. Keep
              alternate takes in the same piece family.
            </p>
          </details>
        {/if}
      </section>
      {#if !guided || guideStep === 1}
        {#if guided}<div class="row spaced">
            <button
              disabled={busy ||
                !!active ||
                reviewDirty ||
                !studio.tagger?.ready}
              onclick={() => job("analyze")}
              >Suggest descriptions locally</button
            >
            <details>
              <summary>About suggestions</summary>
              <p>
                Drafts can be wrong. Listen, edit, then approve. You can also
                write descriptions yourself.
              </p>
              {#if !studio.tagger?.ready}<button
                  disabled={busy || reviewDirty}
                  onclick={() => go("setup")}>Set up description helper</button
                >{/if}
            </details>
          </div>{/if}
        <div class="review-layout">
          <section class="review-list" aria-label="Recordings to review">
            <div class="list-toolbar">
              <label class="search-label"
                >Find a recording<input
                  type="search"
                  bind:value={search}
                  placeholder="Title, description, section, tempo…"
                /></label
              ><label
                >Review status<select bind:value={reviewFilter}
                  ><option value="all">All pieces</option><option
                    value="pending">Needs review</option
                  ><option value="reviewed">Reviewed</option><option
                    value="comparison">Comparison pieces</option
                  ></select
                ></label
              >
              <p class="list-count" role="status">
                {visibleTracks.length} of {project.tracks.length} recordings
                <span
                  >{project.tracks.filter((t) => t.reviewed).length} reviewed</span
                >
              </p>
            </div>
            {#each visibleTracks as track (track.id)}
              <div class="review-entry">
                <input
                  type="checkbox"
                  aria-label={`Select ${track.name} for approval`}
                  value={track.id}
                  bind:group={checkedTracks}
                />
                <button
                  disabled={reviewDirty || busy}
                  class:current={reviewId === track.id}
                  aria-pressed={reviewId === track.id}
                  title={track.name}
                  onclick={() => (reviewId = track.id)}
                  ><span class="recording-name">{displayName(track.name)}</span
                  ><small
                    ><span>{track.reviewed ? "Reviewed" : "Needs review"}</span
                    ><time>{duration(track.duration)}</time></small
                  ></button
                >
              </div>
            {/each}
            {#if checkedTracks.length}<div class="bulk-tools">
                <h3>{checkedTracks.length} selected</h3>
                <button
                  disabled={busy || reviewDirty || !checkedTracks.length}
                  onclick={() => save({ review_tracks: checkedTracks })}
                  >Approve selected ({checkedTracks.length})</button
                >
                <label
                  >Append to selected descriptions<input
                    bind:value={bulkText}
                    placeholder="Shared instrumentation or context"
                  /></label
                >
                <button
                  disabled={busy ||
                    reviewDirty ||
                    !checkedTracks.length ||
                    !bulkText.trim()}
                  onclick={() =>
                    save({
                      append_captions: { ids: checkedTracks, text: bulkText },
                    })}>Append text to selected</button
                >
                <p class="muted">Bulk edits mark metadata for review again.</p>
                <p class="muted">
                  Bulk approval uses saved descriptions. Listen and check them
                  first.
                </p>
              </div>{/if}
            {#if project.tracks.length && !project.tracks.some((t) => (reviewFilter === "all" || (reviewFilter === "comparison" ? t.split === "validation" : reviewFilter === "reviewed" ? t.reviewed : !t.reviewed)) && matchesRecording(t, search))}<div
                class="empty"
                role="status"
              >
                <p>No recordings match these filters.</p>
                <button
                  onclick={() => {
                    search = "";
                    reviewFilter = "all";
                  }}>Clear filters</button
                >
              </div>{/if}
          </section>
          <section class="review-detail">
            {#each project.tracks.filter((t) => t.id === reviewId) as track (track.id)}
              <MetadataReview
                {guided}
                hasNext={visibleTracks.findIndex((t) => t.id === track.id) >=
                  0 &&
                  visibleTracks.findIndex((t) => t.id === track.id) <
                    visibleTracks.length - 1}
                onNext={() => {
                  const next =
                    visibleTracks[
                      visibleTracks.findIndex((t) => t.id === track.id) + 1
                    ];
                  if (next) reviewId = next.id;
                }}
                onDirty={(value) => (reviewDirty = value)}
                {track}
                audioUrl={url(track)}
                {busy}
                save={async (changes) => {
                  busy = true;
                  error = "";
                  try {
                    await api("project", { project_id: pid, ...changes });
                    await refresh();
                  } catch (e) {
                    error = String(e instanceof Error ? e.message : e);
                    throw e;
                  } finally {
                    busy = false;
                  }
                }}
              />
              <details class="spaced">
                <summary>Recording details and use</summary>
                <p class="muted">
                  A piece family groups alternate takes of the same composition.
                  All takes in a family share the same training or comparison
                  role. Tempo is reference information; include it in the
                  description if the model should learn that wording.
                </p>
                <div class="fields">
                  <label
                    >Piece family<input
                      disabled={busy || reviewDirty}
                      value={track.family}
                      onchange={(e) =>
                        save({
                          track: {
                            id: track.id,
                            family: e.currentTarget.value,
                          },
                        })}
                    /></label
                  >
                  <label
                    >Use for<select
                      disabled={busy || reviewDirty}
                      value={track.split}
                      onchange={(e) =>
                        save({
                          track: { id: track.id, split: e.currentTarget.value },
                        })}
                      ><option value="train">Learn from this recording</option
                      ><option value="validation">Keep for comparison</option
                      ><option value="exclude">Exclude</option></select
                    ></label
                  >
                  <label
                    >Tempo<input
                      disabled={busy || reviewDirty}
                      value={track.bpm}
                      placeholder="Optional BPM"
                      onchange={(e) =>
                        save({
                          track: { id: track.id, bpm: e.currentTarget.value },
                        })}
                    /></label
                  >
                </div>
              </details>
            {/each}
            {#if !project.tracks.some((t) => t.id === reviewId)}<p
                class="empty"
              >
                Choose a recording to listen, describe it, and check its
                sections.
              </p>{/if}
          </section>
        </div>
      {/if}
      {#if !guided && !project.tracks.length}<div class="empty">
          Bring in at least three different pieces to begin. You can review
          every description before training.
        </div>{/if}
    {:else if project && page === "train"}
      {#if !guided}<div class="page-title">
          <div>
            <h1 tabindex="-1">Personal sound</h1>
            <p class="lede">
              Adapt a model to your reviewed recordings. Listen before choosing
              a version.
            </p>
          </div>
        </div>
      {/if}
      <div class="personalize-layout" class:guided-training={guided}>
        <div class="training-controls">
          <section class="panel">
            <section class="readiness" aria-label="Training preparation">
              <h2>Before training</h2>
              {#if !readiness}<p role="status">
                  Checking this collection…
                </p>{:else if readiness.ready}<p>
                  {readiness.counts.train_tracks} recordings ready, {readiness
                    .counts.training_clips} planned excerpts. {readiness.counts
                    .validation_tracks} recordings stay out of training.
                </p>{:else}<p>
                  Resolve these items before starting. Your saved work stays
                  intact.
                </p>
                <ul>
                  {#each readiness.blockers.slice(0, 5) as blocker}<li>
                      {blocker.message}{#if blocker.track_id}<button
                          disabled={busy}
                          onclick={() => {
                            reviewId = blocker.track_id;
                            search = "";
                            reviewFilter = "all";
                            void go("collection");
                          }}>Open recording</button
                        >{/if}
                    </li>{/each}
                </ul>
                {#if readiness.blockers.length > 5}<p>
                    {readiness.blockers.length - 5} more items need attention. Review
                    the remaining recordings to resolve them.
                  </p>{/if}<button
                  disabled={busy}
                  onclick={() => go("collection")}>Return to recordings</button
                >{/if}
            </section>
            <h2>
              {guided ? "Start with a short test" : "Make a personal version"}
            </h2>
            <p class="muted">
              This test prepares your recordings and runs 20 training steps. It
              measures how long a small run takes on your computer. Listen to
              the result before deciding how much more to train.
            </p>
            {#if guided}<button
                class="primary"
                disabled={!!active || busy || !modelReady || !readiness?.ready}
                onclick={() => job("calibrate", { adapter_id: "" })}
                >Run a short test</button
              >{/if}
            <details open={!guided}>
              <summary>Longer training and model settings</summary>
              <details>
                <summary>Model and training settings</summary>
                <div class="fields">
                  <label
                    >Workload<select
                      value={project.profile}
                      onchange={(e) => save({ profile: e.currentTarget.value })}
                      >{#each Object.entries(studio.profiles) as [key, p]}<option
                          value={key}>{(p as any).label}</option
                        >{/each}</select
                    ></label
                  ><label
                    >Model<select
                      value={project.model_id}
                      onchange={(e) => {
                        adapter = "";
                        trainingAdapter = "";
                        void save({ model_id: e.currentTarget.value });
                      }}
                      >{#each studio.models.filter((m: any) => m.id !== "ace-step") as m}<option
                          value={m.id}>{m.name}</option
                        >{/each}</select
                    ></label
                  ><label
                    >Target training steps<input
                      type="number"
                      min="1"
                      max="10000"
                      bind:value={steps}
                    /></label
                  >
                </div>
                <p class="muted">
                  Continuing starts from the saved version and step count, with
                  a fresh optimizer. This recipe trains on whole excerpts. Check
                  passage edits by listening after training.
                </p>
              </details>
              <label
                >Start from<select bind:value={trainingAdapter}
                  ><option value="">Original model, new training run</option
                  >{#each checkpoints.filter((c) => c.model_id === project?.model_id) as c}<option
                      value={c.id}>{c.name}</option
                    >{/each}</select
                ></label
              >
              <p class="run-summary" role="status">
                {studio.models.find((m: any) => m.id === project.model_id)
                  ?.name}: target {steps ?? "—"} updates{#if validTrainingTarget},
                  {steps - startingStep} new updates from {startingStep}{/if}.
                Change the target in Model and training settings.
              </p>
              {#if !validTrainingTarget}<p class="notice">
                  Choose a target step count greater than {startingStep}, up to
                  10,000 updates, before training.
                </p>{/if}
              <div class="row">
                <button
                  disabled={!!active ||
                    busy ||
                    !modelReady ||
                    !readiness?.ready}
                  onclick={() => job("calibrate", { adapter_id: "" })}
                  >Run a short test</button
                ><button
                  class="primary"
                  disabled={!!active ||
                    busy ||
                    !modelReady ||
                    !readiness?.ready ||
                    !validTrainingTarget}
                  onclick={() => job("train")}>Train this version</button
                >
              </div>
            </details>
          </section>
          <details>
            <summary>Memory and run limits</summary>
            <p class="notice">
              This app has not yet been tested on a physical 8 GB Mac. Your Mac
              has {studio.hardware.memory_gb} GB. Jobs stop at 12 hours and keep completed
              checkpoints.
            </p>
          </details>
          <p class="muted">Training uses only the recordings you approved.</p>
        </div>
        <div class="training-material">
          <details open={!guided}>
            <summary>Preview the training excerpts</summary><TrainingPreview
              {project}
              {api}
              {url}
            />
          </details>
        </div>
      </div>
      {#each studio.jobs
        .filter((j: any) => j.project_id === pid && j.kind === "calibrate" && j.status === "completed")
        .slice(0, 1) as trial}<section class="notice">
          <h2>Short test complete</h2>
          <p>
            This collection took {Math.round(trial.elapsed_seconds)} seconds including
            any preparation and encoding. Cached preparation can make later runs faster.
            A longer run does not scale directly from this total.
          </p>
          <p>
            You can audition the test version in Compose, then choose a longer
            training run and judge the result by listening.
          </p>
          <button
            disabled={busy || !checkpoints.some((c) => c.job_id === trial.id)}
            onclick={() => {
              adapter =
                checkpoints.find((c) => c.job_id === trial.id)?.id ?? "";
              void go("compose");
            }}>Make a take with this version</button
          >
        </section>{/each}
      <details class="history">
        <summary>Training history</summary>
        {#each studio.jobs.filter((j: any) => j.project_id === pid && ["train", "calibrate", "prepare"].includes(j.kind)) as j}<section
            class="track"
          >
            <strong>{j.kind}, {j.status}</strong>
            <p class="log">{j.message}</p>
            <small
              >{Math.round(j.elapsed_seconds)} s, peak worker RSS {j.peak_rss_gb}
              GB , swap growth {j.swap_growth_gb ?? 0} GB</small
            >
          </section>{/each}
      </details>
    {:else if project && page === "compose"}
      {#if !guided}<div class="page-title">
          <div>
            <h1 tabindex="-1">Compose</h1>
            <p class="lede">
              Make a passage, listen, and keep what belongs in your work.
            </p>
          </div>
        </div>
      {/if}
      <div class="compose-layout">
        <div class="compose-controls">
          <section class="panel">
            <details>
              <summary>How descriptions guide a take</summary>
              <p>
                Use the same musical vocabulary you used for your recordings. If
                you described "muted piano, slow pulse, sparse strings," try
                "muted piano with a slow pulse and a rising string line."
                Descriptions guide a result; they are not exact score
                instructions.
              </p>
              <p>
                Compare the original and personal versions with identical
                settings. Listen for control, variety, and unwanted repetition
                before deciding that training helped.
              </p>
            </details>
            <label
              >Describe the passage<textarea
                bind:value={prompt}
                placeholder="A quiet string passage, gradually opening into a bright, unresolved harmony…"
              ></textarea></label
            >
            <div class="fields">
              <label
                >Sound version<select bind:value={adapter}
                  ><option value="">Original model</option
                  >{#each checkpoints.filter((c) => c.model_id === project?.model_id) as c}<option
                      value={c.id}>{c.name}</option
                    >{/each}</select
                ></label
              ><label
                >Duration (seconds)<input
                  type="number"
                  min="1"
                  max={studio.profiles[project.profile].max_seconds}
                  bind:value={seconds}
                /></label
              >
            </div>
            <details>
              <summary>Variation settings</summary>
              <div class="fields">
                <label
                  >Variation seed<input
                    type="number"
                    min="0"
                    max="2147483647"
                    bind:value={seed}
                  /></label
                >
              </div>
              <label
                >Influence of personal version, {strength}<input
                  type="range"
                  min="0"
                  max="1.5"
                  step="0.05"
                  bind:value={strength}
                /></label
              >
            </details>
            <button
              class="primary"
              disabled={!!active || busy || !prompt.trim() || !modelReady}
              onclick={() => job("generate")}>Make a take</button
            >
            <p class="muted">
              Compare versions using the same description, duration, and
              variation seed.
            </p>
          </section>
          <section class="panel">
            <details>
              <summary>Rework a recording</summary>
              <label
                >Source audio<select
                  bind:value={source}
                  onchange={() => {
                    start = 0;
                    end = Math.min(
                      10,
                      items.find((a) => a.id === source)?.duration ?? 10,
                    );
                  }}
                  ><option value="">Choose a recording or render</option
                  >{#each items as a}<option value={a.id}>{a.name}</option
                    >{/each}</select
                ></label
              >{#if selected}<svg
                  class="wave"
                  viewBox="0 0 480 100"
                  role="img"
                  aria-label="Source waveform with selected edit region"
                  >{#each selected.peaks ?? [] as peak, i}<line
                      x1={i * 2}
                      x2={i * 2}
                      y1={50 - peak * 45}
                      y2={50 + peak * 45}
                    />{/each}<rect
                    x={(start / selected.duration) * 480}
                    width={(Math.max(0, end - start) / selected.duration) * 480}
                    y="0"
                    height="100"
                  /></svg
                ><audio
                  aria-label={`Source recording: ${selected.name}`}
                  controls
                  src={url(selected)}
                ></audio>
                <div class="fields">
                  <label
                    >Selection start (s)<input
                      type="number"
                      min="0"
                      max={selected.duration}
                      step="0.1"
                      bind:value={start}
                    /></label
                  ><label
                    >Selection end (s)<input
                      type="number"
                      min="0"
                      max={selected.duration}
                      step="0.1"
                      bind:value={end}
                    /></label
                  >
                </div>
                <button
                  disabled={!!active || !prompt.trim() || !modelReady}
                  onclick={() => job("edit")}>Rework selection</button
                >
                <p class="muted">
                  Audio outside the selection is preserved. Crossfades stay
                  inside its boundaries.
                </p>
                <details>
                  <summary>Create a MIDI draft</summary>
                  <p>
                    Best for a single instrument. Review the detected notes in
                    your DAW.
                  </p>
                  <div class="fields">
                    <label
                      >Lowest MIDI note<input
                        type="number"
                        min="0"
                        max="126"
                        bind:value={low}
                      /></label
                    ><label
                      >Highest MIDI note<input
                        type="number"
                        min="1"
                        max="127"
                        bind:value={high}
                      /></label
                    ><label
                      >Tempo<input
                        disabled={busy || reviewDirty}
                        type="number"
                        min="20"
                        max="300"
                        bind:value={bpm}
                      /></label
                    >
                  </div>
                  <label class="check"
                    ><input type="checkbox" bind:checked={quantize} /> Quantize to
                    sixteenth notes</label
                  ><button disabled={!!active} onclick={() => job("transcribe")}
                    >Find notes locally</button
                  >
                </details>{/if}
            </details>
          </section>
        </div>
        <section class="takes" aria-label="Saved takes">
          <div class="row">
            <h2>Takes <span class="count">{project.assets.length}</span></h2>
            <button
              disabled={busy || !project.selection_history?.length}
              onclick={() =>
                act(() => api("select", { project_id: pid, undo: true }))}
              >Undo keep</button
            >
          </div>
          <details>
            <summary>Taking audio into your DAW</summary>
            <p>
              Export WAV saves a new audio file. Import it into an audio track
              in your DAW; Keep take only marks your preference here. Set
              placement and tempo in your DAW, since generated audio is not
              guaranteed to align to bars. MIDI exports are note drafts for an
              instrument track.
            </p>
          </details>
          {#if !project.assets.length}<div class="empty takes-empty">
              <Icon name="compose" />
              <h3>Your takes will appear here</h3>
              <p>
                Describe a passage and make your first take. Each version is
                saved separately.
              </p>
            </div>{/if}
          {#each [...project.assets].reverse() as a, i}<section
              class="track"
              class:kept={project.selected_asset === a.id}
            >
              <div class="row">
                <h3>
                  {a.path.endsWith(".mid") ? "MIDI draft" : "Take"}
                  {project.assets.length - i}
                </h3>
                <span class="badge"
                  >{a.adapter_id ? "Personal version" : "Original model"}</span
                >
              </div>
              <p class="take-description">{a.name || "Untitled variation"}</p>
              {#if a.notes}<svg
                  class="piano"
                  viewBox="0 0 600 128"
                  role="img"
                  aria-label="Detected MIDI notes"
                  >{#each a.notes as note}<rect
                      x={(note.start / Math.max(0.1, a.duration)) * 600}
                      y={127 - note.pitch}
                      width={Math.max(
                        1,
                        ((note.end - note.start) / Math.max(0.1, a.duration)) *
                          600,
                      )}
                      height="2"
                    />{/each}</svg
                >
                <p>{a.notes.length} notes, MIDI draft</p>{:else}<audio
                  aria-label={`Listen to take ${project.assets.length - i}: ${a.name}`}
                  controls
                  preload="none"
                  src={url(a)}
                ></audio>
                <details class="take-details">
                  <summary>Take details</summary>
                  <p class="muted">
                    Variation seed {a.seed}. Duration {duration(a.duration)}.
                    Sound version: {a.adapter_id
                      ? (checkpoints.find((c) => c.id === a.adapter_id)?.name ??
                        "Personal version (checkpoint unavailable)")
                      : "Original model"}.
                  </p>
                </details>{/if}
              <div class="row">
                <a
                  class="button"
                  aria-label={`Export take ${project.assets.length - i}, ${a.adapter_id ? "personal" : "original"} version, ${a.path.endsWith(".mid") ? "MIDI" : "WAV"}`}
                  href={url(a, true)}
                  download>Export {a.path.endsWith(".mid") ? "MIDI" : "WAV"}</a
                ><button
                  onclick={() =>
                    act(() =>
                      api("select", { project_id: pid, asset_id: a.id }),
                    )}
                  >{project.selected_asset === a.id
                    ? "Kept"
                    : "Keep take"}</button
                >
              </div>
            </section>{/each}
        </section>
      </div>
    {:else}
      {#if !guided}<div class="page-title">
          <div>
            <h1 tabindex="-1">Settings</h1>
            <p class="lede">Downloads and tools for this Mac.</p>
          </div>
        </div>{/if}
      <div class="settings-layout" class:guided-settings={guided}>
        {#if !guided || guideStep === 1}<section class="panel">
            <h2>Description helper</h2>
            {#if !project}<p class="notice">
                Create a collection in Recordings before installing tools. You
                can organize and listen to audio without any downloads.
              </p>{/if}
            <p>
              This optional tool suggests descriptions and section boundaries.
              The download is about 165 MB. It runs on this computer. You can
              write descriptions without it.
            </p>
            {#if studio.tagger?.ready}<p>Installed, available offline</p>{:else}
              <p>
                <a
                  href="https://huggingface.co/laion/clap-htsat-unfused"
                  target="_blank"
                  rel="noreferrer">Model information and license</a
                >
              </p>
              <p>
                The link opens in your browser. Read the license there, then
                return to Studio. If you agree, check the box and download the
                helper. No recordings are uploaded.
              </p>
              <label class="check"
                ><input type="checkbox" bind:checked={taggerAccepted} /> I have read
                and accept the description helper's model terms.</label
              >
              <button
                disabled={!project || !!active || busy || !taggerAccepted}
                onclick={() =>
                  job("install_tagger", { license_accepted: taggerAccepted })}
                >Download description helper (165 MB)</button
              >
            {/if}
            {#if studio.tagger?.ready && project?.tracks.length}<button
                disabled={!!active || busy}
                onclick={() => job("analyze", { force_analysis: true })}
                >Make new suggestions</button
              >
              <p class="muted">
                Your saved descriptions and sections are preserved.
              </p>{/if}
          </section>
        {/if}
        {#if !guided}<section class="panel privacy">
            <h2>Local processing</h2>
            <p class="lede">
              You choose when to download a model. Audio processing runs on this
              computer with network access blocked.
            </p>
          </section>
          <section class="panel">
            <h2>This computer</h2>
            <p>
              {studio.hardware.system}, {studio.hardware.machine}, {studio
                .hardware.memory_gb} GB memory
            </p>
            <p class="muted">Saved locally in {studio.root}</p>
            <p>
              Model training and audio generation currently work on Apple
              Silicon Macs. Windows, Linux, and DAW plug-ins are not supported
              yet.
            </p>
          </section>
        {/if}
        {#if !guided || guideStep !== 1}<section class="panel">
            {#if guided}<h2>
                {studio.models.find((m: any) => m.id === project?.model_id)
                  ?.name}
              </h2>
              <p>
                About {studio.models.find(
                  (m: any) => m.id === project?.model_id,
                )?.download_gb} GB. Internet is needed for this download; training
                and generation run locally.
              </p>{/if}
            <ol class="download-instructions">
              <li>
                <strong>Read the terms in your browser</strong>
                <p>
                  These links open browser pages. Studio stays here, and your
                  recordings are not sent to either site.
                </p>
                <p>
                  <a
                    href="https://huggingface.co/stabilityai/stable-audio-3-optimized"
                    target="_blank"
                    rel="noreferrer"
                    >Open the Stable Audio model page on Hugging Face</a
                  >
                </p>
                <p>
                  <a
                    href="https://ai.google.dev/gemma/terms"
                    target="_blank"
                    rel="noreferrer">Open the Gemma terms on Google</a
                  >
                </p>
                <p>
                  Stable Audio makes the audio. The included Gemma text encoder
                  reads your descriptions, so both sets of terms apply.
                </p>
              </li>
              <li>
                <strong>Return to Studio</strong>
                <p>
                  If you agree to the terms, check the box below. Opening a link
                  does not accept them or start a download.
                </p>
                <label class="check"
                  ><input type="checkbox" bind:checked={accepted} /> I have read and
                  accept the Stable Audio model license and Gemma terms.</label
                >
              </li>
              <li>
                <strong>Download the model</strong>
                <p>
                  Use the download button below. When it finishes, continue to
                  training. You can leave setup and keep listening or editing
                  descriptions without a model.
                </p>
              </li>
            </ol>
            <details>
              <summary>If Hugging Face asks you to sign in</summary>
              <p>
                Hugging Face hosts the model files. If its model page asks for
                an account, sign in or create one there. Review any access
                request and wait for approval if required.
              </p>
              <p>
                If Studio's download then reports an access error, create a
                read-only access token in your Hugging Face account. A token is
                an access key that lets Studio download the model using your
                account.
              </p>
              <p>
                <a
                  href="https://huggingface.co/settings/tokens"
                  target="_blank"
                  rel="noreferrer"
                  >Open Hugging Face access tokens in your browser</a
                >
              </p>
              <p>
                Give the token read access to this model, then paste it below
                and retry the download. Studio uses it for the download and does
                not save it.
              </p>
              <label
                >Hugging Face access token (optional)<input
                  type="password"
                  autocomplete="off"
                  bind:value={hfToken}
                  placeholder="Paste a read-only access token"
                /></label
              >
            </details>
            {#each studio.models.filter((m: any) => !guided || m.id === project?.model_id) as m}<div
                class="model"
              >
                <div>
                  {#if !guided}<h3>{m.name}</h3>{/if}
                  <p>
                    {m.ready
                      ? "Installed. Available offline."
                      : guided
                        ? "Ready to download after license review."
                        : m.reason}
                  </p>
                  <small>{m.validation}</small>
                </div>
                {#if m.id !== "ace-step"}<button
                    disabled={!project ||
                      !!active ||
                      !accepted ||
                      busy ||
                      m.ready}
                    onclick={async () => {
                      await job("install", {
                        model_id: m.id,
                        license_accepted: accepted,
                        hf_token: hfToken,
                      });
                      hfToken = "";
                    }}
                    >{m.ready
                      ? "Installed"
                      : `Download model (~${m.download_gb} GB)`}</button
                  >{/if}
              </div>{/each}
            {#if guided && modelReady}<button
                class="primary"
                disabled={busy || !!active}
                onclick={() => moveGuide(3)}>Continue to training</button
              >{/if}
          </section>{/if}
      </div>
    {/if}
    {#if studio?.jobs.filter((j: any) => j.project_id === pid && j.status === "failed").length}<details
        class="failures"
      >
        <summary>Recent job errors</summary>{#each studio.jobs
          .filter((j: any) => j.project_id === pid && j.status === "failed")
          .slice(0, 3) as j}<pre>{j.kind}: {j.message}</pre>{/each}
      </details>{/if}
  </main>
</div>
