<script lang="ts">
  let {
    project,
    api,
    url,
  }: {
    project: any;
    api: (path: string) => Promise<any>;
    url: (item: any) => string;
  } = $props();
  let data = $state<any>(null),
    error = $state(""),
    selectedId = $state("");
  let audio = $state<HTMLAudioElement>(),
    stopAt = $state<number | null>(null);
  let selected = $derived(data?.tracks.find((t: any) => t.id === selectedId));
  let original = $derived(project.tracks.find((t: any) => t.id === selectedId));
  let limit = $state(50);
  let previewKey = $derived(
    `${project.id}:${project.updated}:${project.profile}`,
  );
  $effect(() => {
    const key = previewKey.split(":")[0];
    let cancelled = false;
    data = null;
    error = "";
    api("training-preview?project_id=" + key)
      .then((result) => {
        if (cancelled) return;
        data = result;
        if (!result.tracks.some((t: any) => t.id === selectedId))
          selectedId = result.tracks[0]?.id ?? "";
      })
      .catch((e) => {
        if (!cancelled) error = String(e);
      });
    return () => {
      cancelled = true;
    };
  });
  function listen(start: number, end: number) {
    if (!audio) return;
    stopAt = end;
    audio.currentTime = start;
    audio.play().catch((e) => (error = String(e)));
  }
</script>

<section class="panel">
  <h2>Listen before training</h2>
  <p>Review the excerpts and saved descriptions for this version.</p>
  {#if error}<p role="alert" class="error">{error}</p>{/if}
  {#if !data && !error}<p aria-live="polite">Loading excerpts…</p>{/if}
  {#if data}
    <p>
      {data.training_clips} excerpts. Training listens to {data.crop_seconds}-second
      stretches of each, so every excerpt keeps its real start and ending.
    </p>
    <label
      >Recording and role<select
        bind:value={selectedId}
        onchange={() => {
          limit = 50;
          stopAt = null;
          audio?.pause();
        }}
      >
        {#each data.tracks as track}<option value={track.id}
            >{track.name}, {track.split === "validation"
              ? "Comparison"
              : track.split === "exclude"
                ? "Excluded"
                : "Training"}{track.reviewed ? "" : ", needs approval"}</option
          >{/each}
      </select></label
    >
    {#if selected && original}
      <p>
        <strong
          >{selected.split === "validation"
            ? "Comparison recording. Not used for training."
            : selected.split === "exclude"
              ? "Excluded from training"
              : "Training material"}</strong
        >
      </p>
      {#each selected.issues as issue}<p class="notice">{issue}</p>{/each}
      {#if selected.gap_seconds > 0}<p class="notice">
          Training skips {selected.gap_seconds.toFixed(2)} seconds between sections.
        </p>{/if}
      {#if selected.short_tail_seconds > 0}<p class="notice">
          {selected.short_tail_seconds.toFixed(2)} s omitted because remaining clips
          are shorter than one second.
        </p>{/if}
      {#key selectedId}
        <audio
          aria-label={`Training source: ${selected.name}`}
          bind:this={audio}
          controls
          preload="metadata"
          src={url(original)}
          ontimeupdate={() => {
            if (audio && stopAt !== null && audio.currentTime >= stopAt) {
              audio.pause();
              stopAt = null;
            }
          }}
        ></audio>
      {/key}
      {#each selected.clips.slice(0, limit) as clip, i}
        <article class="crop-preview">
          <div class="row">
            <strong
              >{i + 1}. {clip.label}, {clip.start.toFixed(2)}–{clip.end.toFixed(
                2,
              )} s</strong
            ><button
              aria-label={`Listen to excerpt ${i + 1}, ${clip.label}, ${clip.start.toFixed(2)} to ${clip.end.toFixed(2)} seconds`}
              onclick={() => listen(clip.start, clip.end)}>Listen</button
            >
          </div>
          <p>{clip.caption}</p>
        </article>
      {/each}
      {#if selected.clips.length > limit}<button onclick={() => (limit += 50)}
          >Show more excerpts</button
        >{/if}
    {/if}
    <p class="muted">
      {data.note} You can listen to comparison recordings here to check their descriptions.
      They stay out of training.
    </p>
  {/if}
</section>
