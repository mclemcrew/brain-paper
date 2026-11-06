<script lang="ts">
  let {
    step,
    facts,
    disabled,
    dirty,
    move,
    finish,
    setup = false,
  }: {
    setup?: boolean;
    step: number;
    facts: any;
    disabled: boolean;
    dirty: boolean;
    move: (step: number) => void;
    finish: () => void;
  } = $props();
  const labels = [
    "Recordings",
    "Descriptions",
    "Comparison",
    "Training",
    "Listen & export",
  ];
  const titles = [
    "Add your recordings",
    "Describe what you hear",
    "Keep a few pieces for comparison",
    "Train your first personal version",
    "Listen, compare, and export",
  ];
  const hints = [
    "Choose a few pieces or all your recordings. You can listen before training.",
    "Listen, edit the description, then approve it. These words teach the model what to respond to.",
    "These recordings stay out of training, so you can compare against material the model hasn't learned.",
    "Training adapts a model to your recordings. Start with a short run, then listen before training longer.",
    "Make a passage with your personal version. Compare it with the original model, then export a WAV for your DAW.",
  ];
  let progress = $derived(
    step === 0
      ? `${facts.count} recordings added`
      : step === 1
        ? `${facts.reviewed} of ${facts.included} included recordings approved`
        : step === 2
          ? `${facts.held} comparison ${facts.held === 1 ? "recording" : "recordings"}`
          : step === 3
            ? facts.done[3]
              ? "Personal version available"
              : "No personal version yet"
            : facts.done[4]
              ? "Your takes are saved below"
              : "Your first take will appear below",
  );
</script>

<section class="workflow-guide" aria-label="Guided setup">
  <nav aria-label="Setup steps">
    <ol class="step-indicator">
      {#each labels as label, i}<li>
          <button
            aria-current={step === i ? "step" : undefined}
            disabled={disabled || i > facts.firstIncomplete}
            onclick={() => move(i)}
          >
            <span aria-hidden="true">{facts.done[i] ? "✓" : i + 1}</span>{label}
          </button>
        </li>{/each}
    </ol>
  </nav>
  <p class="step-position">Step {step + 1} of 5</p>
  <h1 tabindex="-1">
    {setup
      ? step === 1
        ? "Set up description suggestions"
        : "Set up your music model"
      : titles[step]}
  </h1>
  <p class="step-hint">
    {setup
      ? "Download once. Your recordings stay on this computer."
      : hints[step]}
  </p>
  {#if step === 1 && !setup}<details>
      <summary>What makes a useful description?</summary>
      <p>
        For example: "Muted piano, slow pulse, sparse strings." Describe audible
        instruments, character, and movement. You can add sections later.
      </p>
    </details>{/if}
  {#if !setup}<div class="step-navigation">
      <p role="status">{progress}</p>
      <div class="row">
        {#if step > 0}<button {disabled} onclick={() => move(step - 1)}
            >Back</button
          >{/if}
        {#if step < 4}<button
            class:primary={facts.done[step]}
            disabled={disabled || !facts.done[step]}
            onclick={() => move(step + 1)}
          >
            Continue to {labels[step + 1].toLowerCase()}</button
          >
        {:else}<button disabled={disabled || !facts.done[4]} onclick={finish}
            >Finish guided setup</button
          >{/if}
      </div>
    </div>
  {/if}
  {#if dirty}<p role="status">
      Save or discard your description changes before moving to another step.
    </p>{/if}
</section>
