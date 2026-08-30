/* glassbox visualizer — architecture view.

   The page holds a config, asks the server to describe it, and renders the
   answer. It never computes a shape or a parameter count itself: those come
   from the same code the model is built from, so the picture cannot drift out
   of step with what would actually be constructed. */

const KINDS = ["embedding", "attention", "feedforward", "norm", "residual", "output"];

/* Only these may be sent back. The response also carries derived values such
   as d_head, and returning one of those as a setting is refused by the server. */
const SETTABLE = new Set([
  "vocab_size", "block_size", "d_model", "n_layers", "n_heads", "n_kv_heads",
  "norm", "activation", "pos_encoding", "bias", "dropout",
]);

/* A shape as well as a hue per kind, so the legend survives being printed in
   grey and the two embedding stages read as the same thing at a glance. */
const MARKERS = {
  embedding:   '<circle cx="7" cy="7" r="5.5"/>',
  attention:   '<rect x="1.7" y="1.7" width="10.6" height="10.6" transform="rotate(45 7 7)"/>',
  feedforward: '<path d="M7 1.4 L13 12 L1 12 Z"/>',
  norm:        '<rect x="1.8" y="1.8" width="10.4" height="10.4"/>',
  residual:    '<rect x="0.8" y="5.6" width="12.4" height="2.8"/>',
  output:      '<circle cx="7" cy="7" r="5.5"/>',
};
const marker = (kind, size = 14) =>
  `<svg class="marker" width="${size}" height="${size}" viewBox="0 0 14 14"
        data-kind="${kind}" fill="var(--kind)" aria-hidden="true">${
    MARKERS[kind] || ""}</svg>`;

const state = {
  config: null, base: null, diagram: null, previous: null,
  selected: null, open: false,
  playing: false,
  // id -> the row element, so the walkthrough can open stages in place instead
  // of re-rendering, which would restart every transition.
  rows: new Map(),
  // Which stages are branched out, by id. Kept across re-renders so flipping a
  // toggle does not close what the reader had opened.
  expanded: new Set(),
};

const $ = (s) => document.querySelector(s);
const fmt = (n) => n.toLocaleString("en-US");
const settable = (cfg) =>
  Object.fromEntries(Object.entries(cfg).filter(([k]) => SETTABLE.has(k)));

async function post(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

/* ------------------------------------------------------------- startup */

async function boot() {
  const meta = await (await fetch("/meta")).json();
  const model = meta.models.find((m) => m.name === meta.default) || meta.models[0];

  if (model) {
    state.base = settable(model.config);
    $("#model-name").textContent =
      `${model.label} · val ${model.val_loss}`;
  } else {
    // The architecture view is built to work with nothing trained, so an empty
    // registry is a valid state rather than an error.
    state.base = { vocab_size: 65, block_size: 128, d_model: 192, n_layers: 6,
                   n_heads: 6, n_kv_heads: 6, norm: "layernorm",
                   activation: "gelu", pos_encoding: "learned", bias: true };
    $("#model-name").textContent = "no checkpoint loaded";
  }
  state.config = { ...state.base };

  buildKvOptions();
  bindControls();
  $("#play").addEventListener("click", walkthrough);
  await refresh();
  measureBar();
}

/* Grouped query attention assigns query heads to key/value heads in equal
   groups, so the only valid choices are the divisors of the head count. */
function buildKvOptions() {
  const n = state.config.n_heads;
  const seg = $("#kv-seg");
  seg.innerHTML = "";
  for (let k = 1; k <= n; k++) {
    if (n % k) continue;
    const b = document.createElement("button");
    b.dataset.value = String(k);
    b.textContent = k === n ? `${k}·all` : String(k);
    seg.appendChild(b);
  }
}

function bindControls() {
  document.querySelectorAll(".control").forEach((control) => {
    const field = control.dataset.field;
    control.addEventListener("click", (e) => {
      const button = e.target.closest("button[data-value]");
      if (!button) return;
      const raw = button.dataset.value;
      state.config[field] = field === "n_kv_heads" ? Number(raw) : raw;
      refresh();
    });
  });
}

function syncControls() {
  document.querySelectorAll(".control").forEach((c) => {
    const value = String(state.config[c.dataset.field]);
    c.querySelectorAll("button").forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.value === value)));
  });
}

/* -------------------------------------------------------------- render */

async function refresh() {
  let data;
  try {
    data = await post("/architecture", { config: settable(state.config) });
  } catch (err) {
    $("#diagram").innerHTML = `<p class="note">could not describe that
      configuration: ${err.message}</p>`;
    return;
  }
  state.previous = state.diagram;
  state.diagram = data;
  state.config = { ...state.config, ...settable(data.config) };

  syncControls();
  renderTotals(data);
  renderFlow(data);
  $("#preset-note").textContent =
    ["norm", "activation", "pos_encoding", "n_kv_heads"]
      .every((k) => state.config[k] === state.base[k])
      ? "this is the configuration the loaded checkpoint was trained with"
      : "describing a configuration this checkpoint was not trained with — the "
        + "shapes and counts are real, the weights would have to be retrained";
}

/* Compared by id, so a stage that merely moved is not reported as changed.
   What matters is one appearing, leaving, or costing something different. */
function changedIds(next, prev) {
  if (!prev) return new Set();
  const before = new Map(prev.blocks.map((b) => [b.id, b]));
  const out = new Set();
  for (const b of next.blocks) {
    const was = before.get(b.id);
    if (!was || was.present !== b.present || was.params !== b.params
        || was.label !== b.label) out.add(b.id);
  }
  return out;
}

/* A stage is the card plus whatever branches off it to the right. Opening one
   grows sideways rather than pushing the rest of the model down the page. */
function stageEl(block, changed) {
  const open = state.expanded.has(block.id);
  const n = block.steps.length;

  const row = document.createElement("div");
  row.className = "stage" + (block.present ? "" : " absent-row");
  row.dataset.kind = block.kind;
  row.dataset.open = String(open);

  const card = document.createElement("div");
  card.className = "card glass" + (block.present ? "" : " absent")
                   + (changed.has(block.id) && block.present ? " changed" : "");
  card.dataset.kind = block.kind;
  card.dataset.id = block.id;
  card.setAttribute("aria-selected", String(state.selected === block.id));
  card.innerHTML = `
    <div class="card-top">
      ${marker(block.kind)}
      <div class="card-head">
        <div class="card-kind"></div>
        <div class="card-title"></div>
      </div>
      <div class="card-nums">
        <div class="card-shape"></div>
        ${block.params ? '<div class="card-params"></div>' : ""}
      </div>
    </div>
    ${n ? `<div class="card-more"><span class="pulse"></span>
             <span class="chev">▶</span><span>${n} steps</span></div>` : ""}`;
  card.querySelector(".card-kind").textContent = block.kind;
  card.querySelector(".card-title").textContent = block.label;
  card.querySelector(".card-shape").textContent =
    block.out_shape.length ? `(${block.out_shape.join(", ")})` : "";
  if (block.params) card.querySelector(".card-params").textContent = fmt(block.params);
  row.appendChild(card);

  if (n) {
    const rail = document.createElement("div");
    rail.className = "rail";
    const steps = document.createElement("div");
    steps.className = "steps";
    block.steps.forEach((text, i) => {
      if (i) {
        const hop = document.createElement("div");
        hop.className = "hop";
        steps.appendChild(hop);
      }
      const step = document.createElement("div");
      step.className = "step";
      step.innerHTML = `<div class="tile glass" data-kind="${block.kind}">${i + 1}</div><p></p>`;
      step.querySelector("p").textContent = text;
      steps.appendChild(step);
    });
    rail.innerHTML = '<div class="rail-stem"></div>';
    rail.appendChild(steps);
    row.appendChild(rail);
  }

  // One click both branches the stage and fills the panel below: the same
  // question at two depths, so two gestures would only add a step.
  card.addEventListener("click", () => {
    if (state.playing) stop();
    if (n) {
      const nowOpen = !state.expanded.has(block.id);
      nowOpen ? state.expanded.add(block.id) : state.expanded.delete(block.id);
      row.dataset.open = String(nowOpen);
    }
    selectBlock(block);
  });

  state.rows.set(block.id, row);
  return row;
}

function link() {
  const el = document.createElement("div");
  el.className = "link";
  return el;
}

function renderFlow(data) {
  const changed = changedIds(data, state.previous);
  const host = $("#diagram");
  host.innerHTML = "";
  state.rows.clear();

  const outer = data.blocks.filter((b) => !b.id.startsWith("block."));
  const inner = data.blocks.filter((b) => b.id.startsWith("block."));
  const split = outer.findIndex((b) => b.id === "residual_start");

  const run = (blocks, into) => blocks.forEach((b, i) => {
    if (i) into.appendChild(link());
    into.appendChild(stageEl(b, changed));
  });

  run(outer.slice(0, split + 1), host);
  host.appendChild(link());

  const stack = document.createElement("div");
  stack.className = "stack";
  stack.dataset.open = String(state.open);
  stack.innerHTML = `
    <button class="stack-head">
      <span class="chev">▶</span>
      <span class="stack-title">transformer block</span>
      <span class="badge">×${data.totals.n_layers}</span>
      <span class="stack-meta">${fmt(data.totals.per_layer)} each</span>
    </button>
    <div class="stack-shut">
      <div class="mini"></div>
      <p>repeated ${data.totals.n_layers} times, each writing twice into the
         residual stream</p>
    </div>
    <div class="stack-body"></div>`;

  // Shut, the block still shows its makeup as a row of markers, so the shape
  // of what is hidden stays visible.
  const mini = stack.querySelector(".mini");
  inner.filter((b) => b.present).forEach((b) => {
    mini.insertAdjacentHTML("beforeend", marker(b.kind, 13));
  });

  run(inner, stack.querySelector(".stack-body"));
  stack.querySelector(".stack-head").addEventListener("click", () => {
    state.open = !state.open;
    stack.dataset.open = String(state.open);
  });
  host.appendChild(stack);

  host.appendChild(link());
  run(outer.slice(split + 1), host);
}

function renderTotals(data) {
  $("#stat-params").textContent = fmt(data.totals.parameters);
  $("#stat-layer").textContent = fmt(data.totals.per_layer);

  const byKind = data.totals.by_kind;
  const total = Object.values(byKind).reduce((a, b) => a + b, 0) || 1;
  const host = $("#breakdown");
  host.innerHTML = "";
  KINDS.filter((k) => byKind[k]).forEach((kind) => {
    const el = document.createElement("div");
    el.className = "bd";
    el.innerHTML = `${marker(kind, 11)}<span>${kind}</span>
      <b>${((byKind[kind] / total) * 100).toFixed(0)}%</b>`;
    host.appendChild(el);
  });
}

/* -------------------------------------------------------------- detail */

function selectBlock(block) {
  state.selected = block.id;
  document.querySelectorAll(".card").forEach((c) =>
    c.setAttribute("aria-selected", String(c.dataset.id === block.id)));

  const detail = $("#detail");
  detail.style.setProperty("--kind", `var(--${block.kind})`);
  detail.innerHTML = `
    <div class="d-head">${marker(block.kind)}<span class="d-kind"></span></div>
    <div class="d-title"></div>
    <div class="d-grid">
      <div>
        <p class="d-body"></p>
        ${block.note ? '<p class="d-note"></p>' : ""}
      </div>
      <div class="d-facts">
        <div class="d-fact"><span>output</span><b>${
          block.out_shape.length ? `(${block.out_shape.join(", ")})` : "—"}</b></div>
        <div class="d-fact"><span>parameters</span><b>${
          block.params ? fmt(block.params) : "none"}</b></div>
        <div class="d-fact"><span>steps inside</span><b>${block.steps.length || "—"}</b></div>
        <div class="d-fact"><span>in this configuration</span><b>${
          block.present ? "present" : "absent"}</b></div>
        <div class="d-src"></div>
      </div>
    </div>`;
  detail.querySelector(".d-kind").textContent = block.kind;
  detail.querySelector(".d-title").textContent = block.label;
  detail.querySelector(".d-body").textContent = block.detail;
  if (block.note) detail.querySelector(".d-note").textContent = block.note;
  detail.querySelector(".d-src").textContent = block.source;
  scheduleSignal();
}

/* --------------------------------------------------------- walkthrough

   Unfolds the model one stage at a time, leaving each open so the whole thing
   is laid bare by the end, then folds it all back to where it started. Slow on
   purpose: the point is to be followed, not to finish. */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const STEP_MS = 850;
const FOLD_MS = 130;
const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

function setOpen(id, open) {
  const row = state.rows.get(id);
  if (!row) return;
  open ? state.expanded.add(id) : state.expanded.delete(id);
  row.dataset.open = String(open);
}

function stop() {
  state.playing = false;
  const btn = $("#play");
  btn.dataset.playing = "false";
  btn.textContent = "▶ walkthrough";
  document.querySelectorAll(".stage.lit").forEach((r) => r.classList.remove("lit"));
}

async function walkthrough() {
  if (state.playing) return stop();
  if (!state.diagram) return;

  state.playing = true;
  const btn = $("#play");
  btn.dataset.playing = "true";
  btn.textContent = "■ stop";

  // Where the reader had things before we took over.
  const restoreOpen = new Set(state.expanded);
  const restoreStack = state.open;
  const restoreSelected = state.selected;

  const blocks = state.diagram.blocks.filter((b) => b.present);
  const stack = document.querySelector(".stack");

  // Start from a clean slate so the unfolding reads as a sequence.
  blocks.forEach((b) => setOpen(b.id, false));
  if (stack) { state.open = false; stack.dataset.open = "false"; }
  await sleep(reduced ? 0 : 320);

  const visited = [];
  for (const block of blocks) {
    if (!state.playing) break;

    // The repeated block has to be opened before its stages can be reached.
    if (block.id.startsWith("block.") && stack && !state.open) {
      state.open = true;
      stack.dataset.open = "true";
      await sleep(reduced ? 0 : 420);
    }

    const row = state.rows.get(block.id);
    if (row) {
      document.querySelectorAll(".stage.lit").forEach((r) => r.classList.remove("lit"));
      row.classList.add("lit");
      row.scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "center" });
      setTimeout(scheduleSignal, reduced ? 0 : 380);
    }
    if (block.steps.length) setOpen(block.id, true);
    selectBlock(block);
    visited.push(block.id);

    await sleep(reduced ? 0 : STEP_MS);
  }

  if (state.playing) await sleep(reduced ? 0 : 1400);

  // Everything comes back to where it was.
  document.querySelectorAll(".stage.lit").forEach((r) => r.classList.remove("lit"));
  for (const id of visited.reverse()) {
    if (!state.playing) break;
    setOpen(id, restoreOpen.has(id));
    await sleep(reduced ? 0 : FOLD_MS);
  }
  if (stack) { state.open = restoreStack; stack.dataset.open = String(restoreStack); }
  restoreOpen.forEach((id) => setOpen(id, true));
  if (restoreSelected) {
    const block = state.diagram.blocks.find((b) => b.id === restoreSelected);
    if (block) selectBlock(block);
  }
  $("#diagram").scrollIntoView({ behavior: reduced ? "auto" : "smooth", block: "start" });
  stop();
}

/* ----------------------------------------------------------------- signal

   Draws the line from the panel to the stage being explained. Recomputed on
   scroll and resize because the panel is fixed while the diagram moves under
   it, so the two ends are in different frames of reference. */

function drawSignal() {
  const svg = $("#signal");
  const path = svg.querySelector("path");
  const card = state.selected
    && document.querySelector(`.card[data-id="${CSS.escape(state.selected)}"]`);
  const panel = $("#detail");

  // Only in the wide layout, where the panel sits beside the diagram rather
  // than above it. Checking the panel's position value was the earlier version
  // of this and broke silently the moment it changed from fixed to sticky.
  const beside = matchMedia("(min-width: 1380px)").matches;
  if (!card || !panel || !beside || !state.selected) {
    svg.classList.remove("on");
    return;
  }

  const main = $("main").getBoundingClientRect();
  const a = panel.getBoundingClientRect();
  const b = card.getBoundingClientRect();

  const x1 = a.right - main.left;
  const y1 = Math.min(Math.max(b.top + b.height / 2, a.top + 24), a.bottom - 24) - main.top;
  const x2 = b.left - main.left - 6;
  const y2 = b.top + b.height / 2 - main.top;

  // Off the top or bottom of the viewport there is nothing to point at.
  if (b.bottom < 70 || b.top > innerHeight - 10) {
    svg.classList.remove("on");
    return;
  }

  const mid = x1 + (x2 - x1) * 0.55;
  path.setAttribute("d", `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`);
  svg.style.setProperty("--kind", getComputedStyle(card).getPropertyValue("--kind"));
  svg.classList.add("on");
}

let queued = false;
function scheduleSignal() {
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => { queued = false; drawSignal(); });
}
addEventListener("scroll", scheduleSignal, { passive: true });
addEventListener("resize", scheduleSignal);

/* The bar wraps differently at different widths, so the panel is told where it
   actually ends rather than assuming a height. */
function measureBar() {
  const bar = document.querySelector(".bar");
  if (!bar) return;
  // The bar is sticky at the top, so the panel has to begin below its height —
  // not below where the bar happens to sit before the page is scrolled.
  document.documentElement.style.setProperty(
    "--bar-clear", `${Math.round(bar.getBoundingClientRect().height) + 22}px`);
  scheduleSignal();
}
addEventListener("resize", measureBar);
addEventListener("load", measureBar);

boot();
