<script lang="ts">
  let { name }: { name: string } = $props();
  const paths: Record<string, string> = {
    collection: "M4 5h16M4 12h16M4 19h10",
    compose: "M4 10v4m4-9v14m4-11v8m4-13v18m4-11v4",
    train: "M4 7h16M4 17h16M8 4v6m8 4v6",
    setup:
      "M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8M12 3v2m0 14v2M3 12h2m14 0h2M5.6 5.6 7 7m10 10 1.4 1.4M5.6 18.4 7 17m10-10 1.4-1.4",
    local: "M7 10V7a5 5 0 0 1 10 0v3M5 10h14v11H5zM12 14v3",
    add: "M12 5v14M5 12h14",
  };
</script>

<svg
  class="icon"
  width="20"
  height="20"
  viewBox="0 0 24 24"
  fill="none"
  stroke="currentColor"
  stroke-width="1.5"
  stroke-linecap="round"
  stroke-linejoin="round"
  aria-hidden="true"><path d={paths[name] || paths.collection} /></svg
>
