<script lang="ts">
  import { tick } from "svelte";
  let {
    projects,
    value,
    disabled = false,
    select,
  }: {
    projects: { id: string; name: string; tracks?: unknown[] }[];
    value: string;
    disabled?: boolean;
    select: (id: string) => void;
  } = $props();
  const uid = $props.id();
  let trigger: HTMLButtonElement;
  let panel: HTMLDivElement;
  let opened = $state(false);
  let search = $state("");
  let left = $state(0),
    top = $state(0),
    width = $state(320);
  let current = $derived(projects.find((p) => p.id === value));
  let filtered = $derived(
    projects.filter((p) =>
      p.name.toLowerCase().includes(search.trim().toLowerCase()),
    ),
  );
  function close() {
    panel?.hidePopover();
  }
  async function open() {
    if (disabled) return;
    if (opened) {
      close();
      return;
    }
    const rect = trigger.getBoundingClientRect();
    width = Math.min(340, window.innerWidth - 32);
    left = Math.max(16, Math.min(rect.left, window.innerWidth - width - 16));
    top = Math.max(16, Math.min(rect.bottom + 8, window.innerHeight - 336));
    search = "";
    panel.showPopover();
    await tick();
    (
      panel.querySelector<HTMLInputElement>("input") ??
      panel.querySelector<HTMLButtonElement>('[aria-current="true"]') ??
      panel.querySelector<HTMLButtonElement>("button")
    )?.focus();
  }
  function choose(id: string) {
    close();
    trigger.focus();
    if (id !== value) select(id);
  }
  function keys(event: KeyboardEvent) {
    if (event.key === "Escape") {
      event.preventDefault();
      close();
      trigger.focus();
      return;
    }
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    if (
      event.target instanceof HTMLInputElement &&
      ["Home", "End"].includes(event.key)
    )
      return;
    const choices = Array.from(
      panel.querySelectorAll<HTMLButtonElement>(".collection-choice"),
    );
    if (!choices.length) return;
    event.preventDefault();
    const index = choices.indexOf(document.activeElement as HTMLButtonElement);
    const next =
      event.key === "Home"
        ? 0
        : event.key === "End"
          ? choices.length - 1
          : event.key === "ArrowDown"
            ? (index + 1) % choices.length
            : index < 0 ? choices.length - 1 : (index - 1 + choices.length) % choices.length;
    choices[next].focus();
  }
</script>

<svelte:window onresize={close} />
<div class="collection-picker">
  <span class="collection-label" id={`${uid}-label`}>Collection</span>
  <button
    class="collection-trigger"
    bind:this={trigger}
    {disabled}
    aria-label={`Change collection. ${current?.name ?? "No collection selected"}`}
    aria-haspopup="dialog"
    aria-expanded={opened}
    aria-controls={`${uid}-panel`}
    onclick={open}
    onkeydown={(event) => {
      if (event.key === "ArrowDown") {
        event.preventDefault();
        void open();
      }
    }}
  >
    <span>{current?.name ?? "Choose a collection"}</span>
    <svg
      aria-hidden="true"
      width="16"
      height="16"
      viewBox="0 0 16 16"
      fill="none"
      ><path
        d="m4 6 4 4 4-4"
        stroke="currentColor"
        stroke-width="1.5"
        stroke-linecap="round"
        stroke-linejoin="round"
      /></svg
    >
  </button>
</div>
<div
  class="collection-popover"
  id={`${uid}-panel`}
  bind:this={panel}
  popover="auto"
  role="dialog"
  aria-label="Choose a collection"
  tabindex="-1"
  style:left={`${left}px`}
  style:top={`${top}px`}
  style:width={`${width}px`}
  ontoggle={(event) => (opened = event.newState === "open")}
  onkeydown={keys}
>
  <p class="collection-popover-title">Your collections</p>
  {#if projects.length > 6}<input
      aria-label="Find a collection"
      placeholder="Find a collection"
      bind:value={search}
    />{/if}
  <div class="collection-options">
    {#each filtered as project}<button
        class="collection-choice"
        aria-current={project.id === value ? "true" : undefined}
        onclick={() => choose(project.id)}
      >
        <span
          ><span class="collection-name">{project.name}</span><small
            >{project.tracks?.length ?? 0} recordings</small
          ></span
        >
        {#if project.id === value}<span
            class="collection-check"
            aria-label="Selected">✓</span
          >{/if}
      </button>{/each}
    {#if !filtered.length}<p class="collection-empty">
        {projects.length
          ? "No matching collections."
          : "Create a collection to add recordings."}
      </p>{/if}
  </div>
</div>
