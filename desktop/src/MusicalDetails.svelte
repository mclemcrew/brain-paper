<script lang="ts">
  let { value = $bindable({}), suggestions = {}, disabled = false, change }: {
    value: Record<string, any>; suggestions?: Record<string, any>; disabled?: boolean; change: () => void;
  } = $props();
  const fields = [['instruments', 'Instruments'], ['vocals', 'Vocals'], ['mood', 'Mood'], ['texture', 'Texture'], ['techniques', 'Playing techniques'], ['tempo', 'Tempo']];
  function edit(key: string, text: string) {
    value = { ...value, [key]: { ...value[key], value: text, source: 'Composer', reviewed: true, dismissed: false } };
    change();
  }
  function accept(key: string, suggestion: any) {
    value = { ...value, [key]: { ...structuredClone($state.snapshot(suggestion)), reviewed: true } };
    change();
  }
</script>
<details>
  <summary>Musical details (optional)</summary>
  <p class="muted">Add only what you can hear. Leave anything uncertain blank. These details join your description in training.</p>
  <div class="fields">
    {#each fields as [key, label]}
      <div>
        <label>{label}<input {disabled} value={value[key]?.value ?? ''} oninput={(e) => edit(key, e.currentTarget.value)} placeholder={key === 'tempo' ? 'e.g. slow, free rhythm, or 90 BPM' : 'Optional'} /></label>
        {#if value[key]?.value}
          <small class="muted">{value[key].reviewed ? (value[key].source === 'Composer' ? 'Entered by you' : 'Suggestion checked by you') : 'Unverified; omitted from training'}</small>
          {#if !value[key].reviewed}<button {disabled} onclick={() => accept(key, value[key])}>Confirm {label.toLowerCase()}</button>{/if}
          <button {disabled} onclick={() => edit(key, '')}>Clear {label.toLowerCase()}</button>
        {/if}
        {#if suggestions[key] && !value[key]?.value && !value[key]?.dismissed}
          <p class="muted">Suggestion: {suggestions[key].value}</p>
          <button {disabled} onclick={() => accept(key, suggestions[key])}>Use {label.toLowerCase()} suggestion</button>
          <button {disabled} onclick={() => { value = { ...value, [key]: { ...structuredClone($state.snapshot(suggestions[key])), value: "", reviewed: false, dismissed: true } }; change(); }}>Dismiss {label.toLowerCase()} suggestion</button>
        {/if}
      </div>
    {/each}
  </div>
</details>

<style>
  .fields { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 210px), 1fr)); gap: 1rem; }
  small { display: block; margin-block: .35rem; }
</style>
