/* Listening test v2: 3AFC "which section was regenerated?" + "sounds like Ravi" rating.
 *
 * Loads trials.json (built by `python -m ravi.eval.make_listening_test`); serve the directory
 * over HTTP (e.g. `python -m http.server -d listening-test-v2 8000`), fetch() does not work
 * from file://. Responses are saved to localStorage after every step (key
 * lt2:{build_id}:{block}:{rater}) and downloaded as responses_{rater}.json.
 *
 * The answer key is base64 JSON in each trial's `key`. It is decoded only when downloading
 * (to fill `correct`; the scorer recomputes it from trials.json anyway) and in ?review=1.
 */
(function () {
  "use strict";

  const LETTERS = ["A", "B", "C"];
  const params = new URLSearchParams(location.search);
  const REVIEW = params.get("review") === "1";

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  let DOC = null;        // trials.json
  let state = null;      // {rater, block, build_id, started_at, responses: {trial_id: resp}}
  let trials = [];       // current block's trials
  let cur = 0;           // index into trials
  let phase = 1;         // 1 = 3AFC, 2 = rating
  let pending = null;    // working response for the current trial

  // ───────────────────────────────── utilities

  function nowIso() { return new Date().toISOString(); }

  function decodeKey(s) {
    try {
      const bytes = Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
      return JSON.parse(new TextDecoder().decode(bytes));
    } catch (e) {
      return null;
    }
  }

  function storageKey(block, rater) {
    return "lt2:" + DOC.build_id + ":" + block + ":" + rater.trim().toLowerCase();
  }

  function loadState(block, rater) {
    try {
      const raw = localStorage.getItem(storageKey(block, rater));
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  function saveState() {
    if (REVIEW || !state) return;
    try {
      localStorage.setItem(storageKey(state.block, state.rater), JSON.stringify(state));
      localStorage.setItem("lt2:last", JSON.stringify({ rater: state.rater, block: state.block }));
    } catch (e) {
      showError("Could not save to this browser's storage (" + e.name + "). Download your responses regularly.");
    }
  }

  function showError(msg) {
    const el = $("#error");
    el.textContent = msg;
    el.hidden = false;
  }

  function show(id) {
    for (const s of ["#start", "#trial", "#done"]) $(s).hidden = s !== id;
  }

  function isComplete(r) { return r && r.choice && r.rating; }

  function nDone() {
    return trials.filter((t) => isComplete(state.responses[t.id])).length;
  }

  function updateProgress() {
    const n = trials.length;
    const d = state ? nDone() : 0;
    $("#progress-fill").style.width = n ? (100 * d / n).toFixed(1) + "%" : "0";
    $("#progress-text").textContent = n ? d + " of " + n + " trials done" : "";
    $("#download").disabled = !state || REVIEW;
    $("#who").textContent = state ? state.rater + " · block " + state.block : "";
  }

  function stopAll() {
    for (const a of $$("audio")) a.pause();
  }

  function sanitize(name) {
    return name.trim().replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^_+|_+$/g, "") || "rater";
  }

  // ───────────────────────────────── start screen

  function initStart() {
    const sel = $("#block");
    for (const b of Object.keys(DOC.blocks)) {
      const o = document.createElement("option");
      o.value = b;
      o.textContent = b + " (" + DOC.blocks[b].trials.length + " trials)";
      sel.appendChild(o);
    }
    let last = null;
    try { last = JSON.parse(localStorage.getItem("lt2:last") || "null"); } catch (e) { last = null; }
    const rater = params.get("rater") || (last && last.rater) || "";
    const block = params.get("block") || (last && last.block) || "";
    $("#rater").value = rater;
    if (block && DOC.blocks[block]) sel.value = block;
    const refresh = () => {
      const r = $("#rater").value.trim();
      const s = r ? loadState(sel.value, r) : null;
      const n = DOC.blocks[sel.value].trials.length;
      if (REVIEW) {
        $("#resume-info").textContent = "Review mode: browse block " + sel.value + " with the answers shown.";
        $("#start-btn").textContent = "Review";
      } else if (s) {
        const d = DOC.blocks[sel.value].trials.filter((t) => isComplete(s.responses[t.id])).length;
        $("#resume-info").textContent = "Saved progress found: " + d + " of " + n + " trials done.";
        $("#start-btn").textContent = d >= n ? "View" : "Resume";
      } else {
        $("#resume-info").textContent = "";
        $("#start-btn").textContent = "Start";
      }
      $("#start-btn").disabled = !REVIEW && !r;
    };
    $("#rater").addEventListener("input", refresh);
    sel.addEventListener("change", refresh);
    $("#start-btn").addEventListener("click", begin);
    $("#rater").addEventListener("keydown", (e) => { if (e.key === "Enter") begin(); });
    refresh();
    show("#start");
  }

  function begin() {
    const rater = $("#rater").value.trim() || (REVIEW ? "review" : "");
    if (!rater) return;
    const block = $("#block").value;
    trials = DOC.blocks[block].trials;
    state = (!REVIEW && loadState(block, rater)) || {
      format: "ravi-listening-test-v2-state", rater: rater, block: block, build_id: DOC.build_id,
      started_at: nowIso(), responses: {},
    };
    saveState();
    if (REVIEW) {
      cur = 0;
    } else {
      cur = trials.findIndex((t) => !isComplete(state.responses[t.id]));
      if (cur < 0) { finish(); return; }
    }
    show("#trial");
    renderTrial();
  }

  // ───────────────────────────────── trial

  function renderTrial() {
    stopAll();
    const t = trials[cur];
    const prev = state.responses[t.id];
    pending = prev ? Object.assign({}, prev) : {
      trial_id: t.id, item: t.item, index: t.index, choice: null, rating: null, comment: "",
      plays: { A: 0, B: 0, C: 0, context: 0 }, shown_at: null, choice_at: null, rated_at: null,
      submitted_at: null,
    };
    if (!pending.shown_at) pending.shown_at = nowIso();
    pending.resumed = Boolean(prev);
    phase = REVIEW ? 1 : (pending.choice ? 2 : 1);

    $("#trial-title").textContent = "Trial " + (cur + 1) + " of " + trials.length;
    $("#trial-id").textContent = t.id;
    for (const l of LETTERS) {
      const a = $('audio[data-audio="' + l + '"]');
      a.src = t.excerpts[l];
      a.load();
      const card = $('.excerpt[data-letter="' + l + '"]');
      card.classList.remove("chosen", "playing", "answer");
      card.classList.toggle("played", pending.plays[l] > 0);
      card.querySelector(".answer-tag").hidden = true;
    }
    const ctx = $("#context-audio");
    ctx.src = t.context;
    ctx.preload = phase === 2 ? "auto" : "none";
    $("#comment").value = pending.comment || "";
    buildScale();
    setChoice(pending.choice, false);
    setPhase(phase);
    if (REVIEW) renderReview(t);
    $("#review-nav").hidden = !REVIEW;
    updateProgress();
  }

  function setPhase(p) {
    phase = p;
    $("#phase1").hidden = !(p === 1 || REVIEW);
    $("#phase2").hidden = !(p === 2 || REVIEW);
    for (const b of $$(".btn-choose")) b.disabled = p !== 1 && !REVIEW;
    if (p === 2) {
      $("#locked-choice").textContent = pending.choice || "";
      $("#context-audio").preload = "auto";
    }
    updateConfirm();
    updateNext();
  }

  function setChoice(l, user) {
    if (phase !== 1 && !REVIEW) return;
    pending.choice = l;
    for (const c of $$(".excerpt")) c.classList.toggle("chosen", c.dataset.letter === l);
    if (user && REVIEW) $("#locked-choice").textContent = l;
    updateConfirm();
  }

  function allPlayed() {
    return LETTERS.every((l) => pending.plays[l] > 0);
  }

  function updateConfirm() {
    const need = DOC.require_all_played !== false && !REVIEW;
    const missing = LETTERS.filter((l) => !(pending.plays[l] > 0));
    $("#confirm").disabled = !pending.choice || (need && missing.length > 0) || REVIEW;
    $("#play-hint").textContent = need && missing.length
      ? "Play " + missing.join(", ") + " before answering."
      : (pending.choice ? "" : "Choose A, B or C.");
  }

  function confirmChoice() {
    if (phase !== 1 || $("#confirm").disabled) return;
    stopAll();
    pending.choice_at = nowIso();
    commit();
    setPhase(2);
  }

  function buildScale() {
    const sc = DOC.rating || { question: "How much does this sound like Ravi's music?", min: 1, max: 5, labels: {} };
    $("#rating-question").textContent = sc.question;
    const box = $("#scale");
    box.innerHTML = "";
    for (let v = sc.min; v <= sc.max; v++) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "btn";
      b.setAttribute("role", "radio");
      b.setAttribute("aria-checked", String(pending.rating === v));
      b.dataset.value = String(v);
      b.innerHTML = '<span class="num">' + v + '</span><span class="lab"></span>';
      b.querySelector(".lab").textContent = (sc.labels && sc.labels[String(v)]) || "";
      b.addEventListener("click", () => setRating(v));
      box.appendChild(b);
    }
  }

  function setRating(v) {
    if (phase !== 2 && !REVIEW) return;
    pending.rating = v;
    pending.rated_at = nowIso();
    for (const b of $$("#scale button")) b.setAttribute("aria-checked", String(Number(b.dataset.value) === v));
    updateNext();
  }

  function updateNext() {
    $("#next").disabled = !pending || !pending.rating || REVIEW;
    $("#rating-hint").textContent = pending && pending.rating ? "" : "Pick a rating to continue.";
  }

  function nextTrial() {
    if (phase !== 2 || $("#next").disabled) return;
    stopAll();
    pending.comment = $("#comment").value;
    pending.submitted_at = nowIso();
    commit();
    const nxt = trials.findIndex((t, i) => i > cur && !isComplete(state.responses[t.id]));
    const any = nxt >= 0 ? nxt : trials.findIndex((t) => !isComplete(state.responses[t.id]));
    if (any < 0) { finish(); return; }
    cur = any;
    renderTrial();
    window.scrollTo({ top: 0 });
  }

  function commit() {
    if (REVIEW) return;
    const r = Object.assign({}, pending);
    delete r.resumed;
    state.responses[r.trial_id] = r;
    state.updated_at = nowIso();
    saveState();
    updateProgress();
  }

  function togglePlay(audio) {
    if (!audio.paused) { audio.pause(); return; }
    for (const a of $$("audio")) if (a !== audio) a.pause();
    try { audio.currentTime = 0; } catch (e) { /* not seekable yet */ }
    const p = audio.play();
    if (p && p.catch) p.catch((e) => showError("Playback failed for " + audio.src + ": " + e.message));
  }

  function wireAudio() {
    for (const l of LETTERS) {
      const a = $('audio[data-audio="' + l + '"]');
      const card = $('.excerpt[data-letter="' + l + '"]');
      a.addEventListener("play", () => {
        for (const o of $$("audio")) if (o !== a) o.pause();
        pending.plays[l] = (pending.plays[l] || 0) + 1;
        card.classList.add("played", "playing");
        updateConfirm();
      });
      a.addEventListener("pause", () => card.classList.remove("playing"));
      a.addEventListener("ended", () => card.classList.remove("playing"));
      a.addEventListener("error", () => showError("Could not load " + a.getAttribute("src") + "."));
      $('[data-play="' + l + '"]').addEventListener("click", () => togglePlay(a));
      $('[data-choose="' + l + '"]').addEventListener("click", () => setChoice(l, true));
    }
    const ctx = $("#context-audio");
    ctx.addEventListener("play", () => {
      for (const o of $$("audio")) if (o !== ctx) o.pause();
      pending.plays.context = (pending.plays.context || 0) + 1;
    });
    ctx.addEventListener("error", () => {
      if (ctx.getAttribute("src")) showError("Could not load " + ctx.getAttribute("src") + ".");
    });
    $("#play-context").addEventListener("click", () => togglePlay(ctx));
  }

  // ───────────────────────────────── review mode

  function renderReview(t) {
    const k = decodeKey(t.key);
    const box = $("#review-info");
    box.hidden = false;
    if (!k) { box.textContent = "Could not decode the key of " + t.id + "."; return; }
    const card = $('.excerpt[data-letter="' + k.answer + '"]');
    if (card) {
      card.classList.add("answer");
      card.querySelector(".answer-tag").hidden = false;
    }
    const rows = [
      ["Answer", k.answer], ["Condition", k.condition], ["Track", k.track_id],
      ["Section", k.section_index + " (" + k.label + ")"], ["Output", k.stem], ["Item", k.item],
      ["Anchor / core", k.anchor + " / " + k.core_s + " s"], ["Gain", String(k.gain)],
    ];
    for (const l of LETTERS) {
      const e = k.excerpts[l];
      rows.push(["Excerpt " + l, e.role + ": section " + e.section_index + " " + e.label + ", context "
        + e.start_s.toFixed(2) + "–" + e.end_s.toFixed(2) + " s"]);
    }
    box.innerHTML = "<table></table>";
    const tb = box.querySelector("table");
    for (const [a, b] of rows) {
      const tr = document.createElement("tr");
      const th = document.createElement("th");
      const td = document.createElement("td");
      th.textContent = a;
      td.textContent = b;
      tr.append(th, td);
      tb.appendChild(tr);
    }
  }

  // ───────────────────────────────── finish / download

  function finish() {
    stopAll();
    show("#done");
    $("#done-info").textContent = nDone() + " of " + trials.length + " trials answered. Rater "
      + state.rater + ", block " + state.block + ".";
    updateProgress();
  }

  function download() {
    if (!state) return;
    const byId = {};
    for (const t of trials) byId[t.id] = t;
    const responses = [];
    for (const t of trials) {
      const r = state.responses[t.id];
      if (!r) continue;
      const k = decodeKey(t.key);
      responses.push(Object.assign({}, r, {
        correct: r.choice && k ? r.choice === k.answer : null,
        complete: Boolean(isComplete(r)),
      }));
    }
    const out = {
      format: "ravi-listening-test-v2-responses",
      version: 1,
      build_id: DOC.build_id,
      rater: state.rater,
      block: state.block,
      started_at: state.started_at,
      downloaded_at: nowIso(),
      user_agent: navigator.userAgent,
      n_trials: trials.length,
      n_complete: nDone(),
      responses: responses,
    };
    const blob = new Blob([JSON.stringify(out, null, 1) + "\n"], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "responses_" + sanitize(state.rater) + ".json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // ───────────────────────────────── keyboard

  function onKey(e) {
    if ($("#trial").hidden) return;
    const tag = (e.target && e.target.tagName) || "";
    if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") {
      if (e.key === "Escape") e.target.blur();
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const k = e.key;
    if (k === "Escape") { stopAll(); return; }
    if (phase === 1 || REVIEW) {
      const i = ["1", "2", "3"].indexOf(k);
      if (i >= 0) {
        e.preventDefault();
        togglePlay($('audio[data-audio="' + LETTERS[i] + '"]'));
        return;
      }
      const c = LETTERS.indexOf(k.toUpperCase());
      if (c >= 0 && k.length === 1) { e.preventDefault(); setChoice(LETTERS[c], true); return; }
      if (k === "Enter" && phase === 1) { e.preventDefault(); confirmChoice(); return; }
      if (REVIEW && k === "ArrowRight") { gotoReview(cur + 1); return; }
      if (REVIEW && k === "ArrowLeft") { gotoReview(cur - 1); return; }
    }
    if (phase === 2 && !REVIEW) {
      if (k === " " || k === "p" || k === "P") { e.preventDefault(); togglePlay($("#context-audio")); return; }
      const v = Number(k);
      if (k.length === 1 && v >= 1 && v <= 5) { e.preventDefault(); setRating(v); return; }
      if (k === "Enter") { e.preventDefault(); nextTrial(); return; }
    }
  }

  function gotoReview(i) {
    if (i < 0 || i >= trials.length) return;
    cur = i;
    renderTrial();
  }

  // ───────────────────────────────── boot

  async function boot() {
    if (REVIEW) $("#review-banner").hidden = false;
    try {
      const res = await fetch("trials.json", { cache: "no-store" });
      if (!res.ok) throw new Error("HTTP " + res.status);
      DOC = await res.json();
    } catch (e) {
      showError("Could not load trials.json (" + e.message + "). Build it with "
        + "`python -m ravi.eval.make_listening_test` and serve this directory over HTTP, e.g. "
        + "`python -m http.server -d listening-test-v2 8000`.");
      return;
    }
    if (!DOC.blocks || !Object.keys(DOC.blocks).length) { showError("trials.json has no blocks."); return; }
    wireAudio();
    $("#confirm").addEventListener("click", confirmChoice);
    $("#next").addEventListener("click", nextTrial);
    $("#comment").addEventListener("input", () => {
      if (!pending || phase !== 2) return;
      pending.comment = $("#comment").value;
      commit();  // keeps a draft comment across reloads; the trial stays incomplete until rated
    });
    $("#download").addEventListener("click", download);
    $("#download-done").addEventListener("click", download);
    $("#prev-trial").addEventListener("click", () => gotoReview(cur - 1));
    $("#next-trial").addEventListener("click", () => gotoReview(cur + 1));
    document.addEventListener("keydown", onKey);
    initStart();
  }

  boot();
})();
