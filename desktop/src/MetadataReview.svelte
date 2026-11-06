<script lang="ts">
  import { tick, untrack } from "svelte";
  import MusicalDetails from "./MusicalDetails.svelte";
  let {
    track,
    audioUrl,
    save,
    busy,
    onDirty,
    onNext,
    hasNext,
    guided = false,
  }: {
    track: any;
    audioUrl: string;
    save: (changes: any) => Promise<void>;
    busy: boolean;
    onDirty: (dirty: boolean) => void;
    onNext: () => void;
    hasNext: boolean;
    guided?: boolean;
  } = $props();
  let caption = $state(untrack(() => track.caption ?? ""));
  let sections = $state<any[]>(
    untrack(() => structuredClone($state.snapshot(track.sections ?? []))),
  );
  let musicalDetails = $state<any>(untrack(() => structuredClone($state.snapshot(track.musical_details ?? {}))));
  let audio = $state<HTMLAudioElement>();
  let dirty = $state(false);
  let playbackEnd = $state<number | null>(null);
  type Draft = { caption: string; sections: any[]; musicalDetails: any };
  let history = $state<Draft[]>([
    untrack(() => ({
      caption,
      musicalDetails: structuredClone($state.snapshot(musicalDetails)),
      sections: structuredClone($state.snapshot(sections)),
    })),
  ]);
  let historyIndex = $state(0);
  let savedDraft = untrack(() => JSON.stringify(history[0]));
  let syncedRevision = untrack(() => track.metadata_revision);
  let playhead = $state(0);
  let playing = $state(false);
  let shortcutsEnabled = $state(false);
  let saveStatus = $state("");
  let playbackError = $state("");
  const editorId = `review-${crypto.randomUUID()}`;
  let editor: HTMLDivElement;
  async function focusSection(index: number) {
    await tick();
    const inputs = editor.querySelectorAll<HTMLInputElement>(
      ".section-editor input",
    );
    const section =
      editor.querySelectorAll<HTMLFieldSetElement>(".section-editor")[index];
    (
      section?.querySelector<HTMLInputElement>("input") ??
      inputs[0] ??
      editor.querySelector<HTMLButtonElement>(".section-map-start")
    )?.focus();
  }
  function snapshot(): Draft {
    return { caption, musicalDetails: structuredClone($state.snapshot(musicalDetails)), sections: structuredClone($state.snapshot(sections)) };
  }
  function reset() {
    caption = track.caption ?? "";
    musicalDetails = structuredClone($state.snapshot(track.musical_details ?? {}));
    sections = structuredClone($state.snapshot(track.sections ?? []));
    history = [snapshot()];
    historyIndex = 0;
    savedDraft = JSON.stringify(history[0]);
    syncedRevision = track.metadata_revision;
    dirty = false;
    onDirty(false);
  }
  $effect(() => {
    const revision = track.metadata_revision;
    untrack(() => {
      if (!dirty && revision !== syncedRevision) reset();
    });
  });
  function changedState() {
    dirty = JSON.stringify(snapshot()) !== savedDraft;
    onDirty(dirty);
  }
  function change() {
    saveStatus = "";
    const current = snapshot();
    if (JSON.stringify(current) === JSON.stringify(history[historyIndex]))
      return;
    history = [...history.slice(0, historyIndex + 1), current].slice(-100);
    historyIndex = history.length - 1;
    changedState();
  }
  function undo() {
    if (busy || historyIndex === 0) return;
    historyIndex--;
    restore();
  }
  function redo() {
    if (busy || historyIndex === history.length - 1) return;
    historyIndex++;
    restore();
  }
  function restore() {
    const d = structuredClone($state.snapshot(history[historyIndex]));
    caption = d.caption;
    musicalDetails = d.musicalDetails;
    sections = d.sections;
    changedState();
  }
  function togglePlayback() {
    if (!audio) return;
    playbackEnd = null;
    if (audio.paused)
      audio
        .play()
        .catch(
          () =>
            (playbackError =
              "Playback could not start. Try the recording controls again."),
        );
    else audio.pause();
  }
  function seek(delta: number) {
    if (audio) {
      playbackEnd = null;
      audio.currentTime = Math.max(
        0,
        Math.min(track.duration, audio.currentTime + delta),
      );
    }
  }
  function splitAtPlayhead() {
    const i = sections.findIndex((s) => playhead > s.start && playhead < s.end);
    if (i >= 0) split(i);
  }
  function keyboard(e: KeyboardEvent) {
    if (!shortcutsEnabled) return;
    const target = e.target as HTMLElement;
    if (
      target.closest(
        'input,textarea,select,button,a,summary,audio,[contenteditable="true"]',
      )
    )
      return;
    if (busy || e.altKey) return;
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
      e.preventDefault();
      e.shiftKey ? redo() : undo();
      return;
    }
    if (e.metaKey || e.ctrlKey) return;
    if (e.code === "Space") {
      e.preventDefault();
      togglePlayback();
    }
    if (e.key === "ArrowLeft") {
      e.preventDefault();
      seek(-5);
    }
    if (e.key === "ArrowRight") {
      e.preventDefault();
      seek(5);
    }
    if (e.key.toLowerCase() === "n" && hasNext && !dirty) {
      e.preventDefault();
      onNext();
    }
    if (e.key.toLowerCase() === "s") {
      e.preventDefault();
      splitAtPlayhead();
    }
  }
  function hasDetails(details: any) {
    return Object.values(details ?? {}).some((d: any) => d.reviewed && d.value?.trim());
  }
  let warnings = $derived.by(() => {
    const messages: string[] = [];
    if (!sections.length && !caption.trim() && !hasDetails(musicalDetails))
      messages.push("Add a description before approving this recording.");
    const ordered = [...sections].sort((a, b) => a.start - b.start);
    let end = 0,
      gaps = 0;
    for (const s of ordered) {
      if (
        !Number.isFinite(s.start) ||
        !Number.isFinite(s.end) ||
        s.start < 0 ||
        s.end > track.duration ||
        s.end <= s.start
      )
        messages.push("Fix invalid section times.");
      if (s.start < end) messages.push("Sections overlap.");
      if (s.start > end) gaps += s.start - end;
      end = Math.max(end, s.end);
      if (!s.caption?.trim() && !hasDetails(s.musical_details))
        messages.push("A section is missing its description.");
      if (s.needs_review)
        messages.push("Check inherited descriptions for changed sections.");
    }
    if (ordered.length) {
      gaps += Math.max(0, track.duration - end);
      if (gaps > 0.001)
        messages.push(
          `${gaps.toFixed(2)} seconds in gaps will be omitted from training.`,
        );
    }
    return [...new Set(messages)];
  });
  function audition(start: number, end: number) {
    if (!audio) return;
    audio.currentTime = start;
    playbackEnd = end;
    audio
      .play()
      .catch(
        () =>
          (playbackError =
            "Playback could not start. Try the recording controls again."),
      );
  }
  function split(index: number) {
    const s = sections[index];
    const point =
      audio && audio.currentTime > s.start && audio.currentTime < s.end
        ? audio.currentTime
        : (s.start + s.end) / 2;
    sections.splice(
      index,
      1,
      { ...s, musical_details: structuredClone($state.snapshot(s.musical_details ?? {})), end: point, needs_review: true },
      {
        ...s,
        id: crypto.randomUUID(),
        musical_details: structuredClone($state.snapshot(s.musical_details ?? {})),
        start: point,
        label: s.label + " continuation",
        needs_review: true,
      },
    );
    change();
  }
  function merge(index: number) {
    const a = sections[index],
      b = sections[index + 1];
    sections.splice(index, 2, {
      ...a,
      end: b.end,
      needs_review: true,
      musical_details: {},
      caption: [...new Set([a, b].map((section) => [section.caption, ...Object.entries(section.musical_details ?? {}).filter(([, d]: any) => d.reviewed).map(([key, d]: any) => `${key}: ${d.value}.`)].filter(Boolean).join(" ")))].join(" "),
    });
    change();
    void focusSection(index);
  }
  async function persist(reviewed = false) {
    saveStatus = "Saving description…";
    try {
      await save({ track: { id: track.id, caption, sections, musical_details: musicalDetails, reviewed } });
      // The parent's save throws on failure; preserve the draft until it succeeds.
      reset();
      saveStatus = reviewed
        ? "Description approved and saved."
        : "Draft saved.";
    } catch (error) {
      saveStatus = "Could not save. Your unsaved draft is still in the editor.";
      throw error;
    }
  }
</script>

<svelte:window
  onkeydown={keyboard}
  onbeforeunload={(e) => {
    if (dirty) {
      e.preventDefault();
      e.returnValue = "";
    }
  }}
/>
<div class="metadata-editor" bind:this={editor}>
  <div class="row">
    <h2 tabindex="-1" title={track.name}>
      {track.name
        .replace(/^\d+_copilot\.mp3\s+/i, "")
        .replace(/\.(wav|aiff?|flac|mp3|ogg)$/i, "")}
    </h2>
    <span class="badge" role="status"
      >{dirty
        ? "Unsaved draft"
        : track.reviewed
          ? "Reviewed"
          : "Needs review"}</span
    >
  </div>
  <div class="review-player">
    {#if !guided || dirty}<div class="row">
        <button disabled={busy || historyIndex === 0} onclick={undo}
          >Undo edit</button
        ><button
          disabled={busy || historyIndex === history.length - 1}
          onclick={redo}>Redo edit</button
        >
      </div>{/if}
    <svg
      class="wave"
      viewBox="0 0 480 100"
      role="img"
      aria-label="Waveform and section boundaries"
    >
      {#each track.peaks ?? [] as peak, i}<line
          x1={i * 2}
          x2={i * 2}
          y1={50 - peak * 45}
          y2={50 + peak * 45}
        />{/each}
      {#each sections as section}<rect
          x={(section.start / track.duration) * 480}
          width={(Math.max(0, section.end - section.start) / track.duration) *
            480}
          y="0"
          height="100"
        />{/each}
    </svg>
    <audio
      bind:this={audio}
      controls
      aria-label={`Recording: ${track.name}`}
      preload="metadata"
      onplay={() => (playing = true)}
      onpause={() => (playing = false)}
      src={audioUrl}
      ontimeupdate={() => {
        if (audio) playhead = audio.currentTime;
        if (audio && playbackEnd !== null && audio.currentTime >= playbackEnd) {
          audio.pause();
          playbackEnd = null;
        }
      }}
    ></audio>
    {#if playbackError}<p class="error" role="alert">{playbackError}</p>{/if}
    <details open={!guided}>
      <summary>Playback and editing tools</summary>
      <div class="row">
        <button disabled={busy || dirty || !hasNext} onclick={onNext}
          >Next recording</button
        >
        <button onclick={togglePlayback}>{playing ? "Pause" : "Play"}</button>
        <button onclick={() => seek(-5)}>Back 5 s</button><button
          onclick={() => seek(5)}>Forward 5 s</button
        >
        <button
          disabled={busy ||
            !sections.some((s) => playhead > s.start && playhead < s.end)}
          onclick={splitAtPlayhead}>Split at {playhead.toFixed(2)} s</button
        >
      </div>
      <details class="keyboard-help">
        <summary>Keyboard shortcuts</summary><label class="check"
          ><input type="checkbox" bind:checked={shortcutsEnabled} /> Enable keyboard
          shortcuts for this editor</label
        >
        <p class="muted">
          Outside form controls: Space plays/pauses, ← / → seek, S splits, N
          moves to next saved recording, ⌘/Ctrl Z undoes (Shift redoes).
        </p>
      </details>
    </details>
  </div>
  <p class="muted">
    Describe what you hear. Approved descriptions guide your personal version.
  </p>
  <label
    >Description<textarea
      disabled={busy}
      bind:value={caption}
      oninput={change}
      placeholder="What is audible throughout this piece?"></textarea></label
  >
  <MusicalDetails bind:value={musicalDetails} suggestions={track.suggestion?.musical_details ?? {}} disabled={busy} {change} />
  {#if track.suggestion}
    <details>
      <summary>Suggested description (unverified)</summary>
      <p>{track.suggestion.caption}</p>
      <button
        disabled={busy}
        onclick={() => {
          caption = track.suggestion.caption;
          change();
        }}>Use draft in editor</button
      >
      <p class="muted">
        The helper chooses from a fixed set of descriptions and can miss what
        matters in your piece. Check the draft by listening. Scores show how
        closely a label matches the audio; they are not confidence percentages.
      </p>
      {#each Object.entries(track.suggestion.tags ?? {}) as [category, values]}
        <p>
          <strong>{category}:</strong>
          {(values as any[])
            .map((v) => `${v.label} (${v.similarity})`)
            .join(", ")}
        </p>
      {/each}
      <p class="muted">
        {track.suggestion.model}, {track.suggestion.windows} audio windows, {track
          .suggestion.version}
      </p>
    </details>
  {/if}
  <details open={sections.length > 0}>
    <summary>Musical sections (optional)</summary>
    <div class="row">
      <h3>Musical sections</h3>
      <button
        class="section-map-start"
        disabled={busy || sections.length > 0}
        onclick={() => {
          sections = [
            {
              id: crypto.randomUUID(),
              start: 0,
              end: track.duration,
              label: "Section 1",
              caption,
              needs_review: true,
            },
          ];
          change();
        }}>Start section map</button
      >
    </div>
    <p class="muted">
      Section descriptions and musical details replace the whole-piece metadata in training.
      Training excerpts stop at section boundaries and skip gaps. Name sections
      in whatever way makes sense for your piece.
    </p>
    {#if track.suggestion?.sections?.length}
      <details>
        <summary>Suggested section boundaries</summary>
        <p>
          {track.suggestion.sections
            .map((s: any) => `${s.start.toFixed(1)}–${s.end.toFixed(1)} s`)
            .join(", ")}
        </p>
        <p class="muted">
          Computed in 10-second windows. Listen and adjust; these are not
          verified formal sections. Using this draft replaces the section map in
          the editor.
        </p>
        <button
          onclick={() => {
            sections = structuredClone(
              $state.snapshot(track.suggestion.sections),
            ).map((s: any) => ({ ...s, needs_review: true }));
            change();
          }}>Use suggested map in editor</button
        >
      </details>
    {/if}
    <div id={`${editorId}-warnings`} aria-live="polite">
      {#each warnings as warning}<p class="notice">
          {warning}
        </p>{/each}
    </div>
    {#each sections as section, i (section.id)}
      <fieldset class="section-editor" disabled={busy}>
        <legend>{section.label || `Section ${i + 1}`}</legend>
        <div class="fields">
          <label
            >Label<input bind:value={section.label} oninput={change} /></label
          >
          <label
            >Start (s)<input
              type="number"
              min="0"
              max={track.duration}
              step="0.01"
              bind:value={section.start}
              oninput={() => {
                section.needs_review = true;
                change();
              }}
            /></label
          >
          <label
            >End (s)<input
              type="number"
              min="0"
              max={track.duration}
              step="0.01"
              bind:value={section.end}
              oninput={() => {
                section.needs_review = true;
                change();
              }}
            /></label
          >
        </div>
        <label
          >What happens in this section?<textarea
            bind:value={section.caption}
            oninput={change}></textarea></label
        >
        <MusicalDetails bind:value={section.musical_details} disabled={busy} {change} />
        <label class="check"
          ><input
            type="checkbox"
            checked={!section.needs_review}
            onchange={(e) => {
              section.needs_review = !e.currentTarget.checked;
              change();
            }}
          /> Description fits this section</label
        >
        <div class="row">
          <button onclick={() => audition(section.start, section.end)}
            >Listen to section</button
          >
          <button onclick={() => split(i)}>Split at playhead / midpoint</button>
          {#if i < sections.length - 1}<button onclick={() => merge(i)}
              >Merge with next</button
            >{/if}
          <button
            onclick={() => {
              sections.splice(i, 1);
              change();
              void focusSection(Math.max(0, i - 1));
            }}>Remove section</button
          >
        </div>
      </fieldset>
    {/each}
  </details>
  <div class="row spaced">
    <button disabled={busy || !dirty} onclick={reset}
      >Discard unsaved changes</button
    >
    <button disabled={busy || !dirty} onclick={() => persist().catch(() => {})}
      >Save draft</button
    >
    <button
      class="primary"
      disabled={busy || warnings.some((w) => !w.includes("gaps will"))}
      aria-describedby={warnings.length ? `${editorId}-warnings` : undefined}
      onclick={async () => {
        const advance = guided && hasNext;
        try {
          await persist(true);
          if (advance) {
            onNext();
            await tick();
            document.querySelector<HTMLElement>(".metadata-editor h2")?.focus();
          }
        } catch {}
      }}
      >{guided && hasNext
        ? "Approve & next recording"
        : "Approve description"}</button
    >
  </div>
  <p role="status" aria-live="polite">{saveStatus}</p>
  <p class="muted">
    Revision {track.metadata_revision ?? 0}, {track.metadata_source ??
      "Imported metadata"}. Changes require a new training run; previous runs
    keep the descriptions they were trained with.
  </p>
  <details>
    <summary
      >Earlier descriptions ({track.metadata_history?.length ?? 0})</summary
    >
    {#each [...(track.metadata_history ?? [])].reverse() as previous}
      <p>
        Revision {previous.metadata_revision ?? 0}: {previous.caption ||
          "No whole-piece description"}, {previous.sections?.length ?? 0} sections
      </p>
    {/each}
  </details>
</div>
