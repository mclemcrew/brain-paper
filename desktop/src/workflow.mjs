// Composer-owned text only: unaccepted machine suggestions are not search results.
export function matchesRecording(track, query) {
  const text = [
    track.name,
    track.caption,
    track.family,
    track.bpm,
    ...(track.sections ?? []).flatMap((section) => [
      section.label,
      section.caption,
    ]),
  ]
    .filter(Boolean)
    .join(" ")
    .toLocaleLowerCase();
  return query
    .trim()
    .toLocaleLowerCase()
    .split(/\s+/)
    .every((term) => text.includes(term));
}

// Keep successful imports when a file fails. Capture the destination once in the caller.
export async function importBatch(
  files,
  upload,
  update,
  stopped = () => false,
) {
  const result = {
    total: files.length,
    done: 0,
    imported: 0,
    duplicates: 0,
    failures: [],
    stopped: false,
  };
  for (const file of files) {
    if (stopped()) {
      result.stopped = true;
      break;
    }
    update({ ...result, current: file.name });
    try {
      const response = await upload(file);
      response.duplicate ? result.duplicates++ : result.imported++;
    } catch (error) {
      result.failures.push({
        name: file.name,
        message: error instanceof Error ? error.message : String(error),
      });
    }
    result.done++;
    update({ ...result, failures: [...result.failures], current: "" });
  }
  return result;
}
