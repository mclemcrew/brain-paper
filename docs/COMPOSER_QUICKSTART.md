# First session with Medium

Use an Apple Silicon Mac with at least 16 GB RAM. Close other memory-heavy applications. Follow the README for source startup, or open the app attached to the pilot release when available.

1. Create a collection. New collections select Stable Audio 3 Medium and the 16 GB workload. Existing collections retain their settings; check them before training.
2. Open Settings, review the model terms and install Medium (about 9.5 GB). A gated download may require Hugging Face access. No recordings need to be uploaded.
3. Import at least three distinct pieces through Add audio. Listen and save accurate descriptions. Review piece families so alternate takes and mixes stay together.
4. Reserve comparison families. Approve descriptions only after reviewing them; machine suggestions remain drafts. Optional section descriptions replace whole-piece descriptions for the covered regions.
5. Inspect the training preview, then run the 20-update short test. Preparation normalizes each recording once and encodes whole described regions, splitting regions above the profile limit. The trainer samples windows: 30 seconds for the 16 GB profile, 47 seconds for the 32 GB profile. These are not the old independently normalized 15-second excerpts.
6. Generate with Original model, then with the saved adapter using the same prompt, duration and seed. Listen for clarity, prompt fit, development and endings. A completed training job is not evidence of improved music.
7. Export a WAV and open it in your DAW. Restart the app and confirm your collection, checkpoint and takes remain available.

To continue training, choose a compatible checkpoint and set the target above its current step count. A target of 100 from step 20 adds 80 updates. Continuation restores weights and step count, but restarts the optimizer. Recipe or training-material changes require a new compatible run.

Do not select the final checkpoint merely because it trained longest. Medium was preferred in the composer's listening feedback; no universal best checkpoint has been established. Small remains available for lightweight diagnostics.

For a problem report, include the app/source version, chip, RAM, macOS version, model, workload, job ID, error text and steps to reproduce. Do not include access tokens. Report listening judgments separately from crashes or failed jobs.
