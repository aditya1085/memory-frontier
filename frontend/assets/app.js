/* ============================================================
   Memory Frontier — frontend logic
   No build step, no framework: plain fetch() calls to the FastAPI
   backend. Falls back to precomputed_results.json (bundled) if the
   live backend is unreachable, so the site still demonstrates the
   real trade-off even if the API is asleep/cold-starting on Railway.
   ============================================================ */

// ---- CONFIGURE THIS in index.html if frontend/backend are deployed separately ----
// Empty string means "same origin" (correct when FastAPI serves this frontend directly).
const API_BASE_URL = (typeof window.MEMORY_FRONTIER_API === "string")
  ? window.MEMORY_FRONTIER_API
  : "http://localhost:8000";
// -----------------------------------------------------------------------------------

const COLORS = { kv: "#4FB3BF", syn: "#E3A857", truth: "#8FBC94", wrong: "#C1666B", grid: "#2A353C", ink: "#9AAAAF" };

let PRECOMPUTED = null;

async function loadPrecomputed() {
  if (PRECOMPUTED) return PRECOMPUTED;
  try {
    const res = await fetch("assets/benchmark_results.json");
    PRECOMPUTED = await res.json();
  } catch (e) {
    PRECOMPUTED = null;
  }
  return PRECOMPUTED;
}

async function apiPost(path, body) {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

async function apiGet(path) {
  const res = await fetch(`${API_BASE_URL}${path}`);
  if (!res.ok) throw new Error(`API error ${res.status}`);
  return res.json();
}

function fmtBytes(b) {
  if (b >= 1e6) return (b / 1e6).toFixed(2) + " MB";
  if (b >= 1e3) return (b / 1e3).toFixed(1) + " KB";
  return b + " B";
}
function fmtPct(x) { return (x * 100).toFixed(0) + "%"; }

/* ---------------- generic line/bar chart via canvas (no CDN dependency) ---------------- */
function drawChart(canvas, opts) {
  // opts: { series: [{label, color, points: [{x,y}]}], xLabel, yLabel, yMax, yFormat }
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr; canvas.height = h * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);

  const padL = 54, padR = 14, padT = 14, padB = 34;
  const plotW = w - padL - padR, plotH = h - padT - padB;

  const allX = opts.series.flatMap(s => s.points.map(p => p.x));
  const allY = opts.series.flatMap(s => s.points.map(p => p.y));
  const xMin = 0, xMax = Math.max(...allX);
  const yMin = 0, yMax = opts.yMax !== undefined ? opts.yMax : Math.max(...allY) * 1.1;

  const xScale = x => padL + (x - xMin) / (xMax - xMin || 1) * plotW;
  const yScale = y => padT + plotH - (y - yMin) / (yMax - yMin || 1) * plotH;

  // grid
  ctx.strokeStyle = COLORS.grid;
  ctx.lineWidth = 1;
  ctx.font = "11px 'IBM Plex Mono', monospace";
  ctx.fillStyle = COLORS.ink;
  const ySteps = 4;
  for (let i = 0; i <= ySteps; i++) {
    const yVal = yMin + (yMax - yMin) * (i / ySteps);
    const yPix = yScale(yVal);
    ctx.beginPath();
    ctx.moveTo(padL, yPix);
    ctx.lineTo(w - padR, yPix);
    ctx.stroke();
    const label = opts.yFormat ? opts.yFormat(yVal) : yVal.toFixed(1);
    ctx.fillText(label, 4, yPix + 4);
  }

  // x axis labels (use actual data x values, sparse)
  const xTicks = [...new Set(allX)].sort((a, b) => a - b);
  const tickStride = Math.max(1, Math.floor(xTicks.length / 6));
  xTicks.forEach((xv, i) => {
    if (i % tickStride !== 0 && i !== xTicks.length - 1) return;
    const xp = xScale(xv);
    ctx.fillText(xv >= 1000 ? (xv/1000) + "k" : xv, xp - 10, h - 10);
  });

  // series lines
  opts.series.forEach(s => {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.points.forEach((p, i) => {
      const xp = xScale(p.x), yp = yScale(p.y);
      if (i === 0) ctx.moveTo(xp, yp); else ctx.lineTo(xp, yp);
    });
    ctx.stroke();
    ctx.fillStyle = s.color;
    s.points.forEach(p => {
      ctx.beginPath();
      ctx.arc(xScale(p.x), yScale(p.y), 3, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}

/* ---------------- hero mini animation ---------------- */
function animateHero() {
  const slotContainer = document.getElementById("hero-slots");
  if (!slotContainer) return;
  let count = 0;
  setInterval(() => {
    count = (count % 40) + 1;
    slotContainer.innerHTML = "";
    for (let i = 0; i < count; i++) {
      const s = document.createElement("div");
      s.className = "slot";
      slotContainer.appendChild(s);
    }
    const label = document.getElementById("hero-slots-count");
    if (label) label.textContent = count + " facts stored";
  }, 220);
}

/* ---------------- Playground + Benchmark section ---------------- */
async function runPlayground() {
  const dim = parseInt(document.getElementById("pg-dim").value);
  const decay = parseFloat(document.getElementById("pg-decay").value);
  const lr = parseFloat(document.getElementById("pg-lr").value);
  const targetFacts = parseInt(document.getElementById("pg-target").value);
  const seqLengths = [50, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000];

  const statusEl = document.getElementById("pg-status");
  statusEl.textContent = "Running live benchmark on backend…";

  let results;
  let source = "live";
  try {
    const data = await apiPost("/benchmark/run", {
      sequence_lengths: seqLengths, dim, decay, lr, num_target_facts: targetFacts, seed: 42
    });
    results = data.results;
  } catch (e) {
    source = "precomputed";
    const pre = await loadPrecomputed();
    if (pre && pre.sweeps && pre.sweeps.default) {
      results = pre.sweeps.default.results;
      statusEl.textContent = "Backend unreachable — showing precomputed default-config results instead.";
    } else {
      statusEl.textContent = "Could not reach backend and no precomputed fallback found.";
      return;
    }
  }

  if (source === "live") statusEl.textContent = "Live result from backend (dim=" + dim + ", λ=" + decay + ", η=" + lr + ").";

  const accCanvas = document.getElementById("chart-accuracy");
  const memCanvas = document.getElementById("chart-memory");

  drawChart(accCanvas, {
    series: [
      { label: "KV Cache", color: COLORS.kv, points: results.map(r => ({ x: r.sequence_length, y: r.kv_accuracy })) },
      { label: "Synaptic", color: COLORS.syn, points: results.map(r => ({ x: r.sequence_length, y: r.synaptic_accuracy })) },
    ],
    yMax: 1.05, yFormat: v => (v * 100).toFixed(0) + "%"
  });

  drawChart(memCanvas, {
    series: [
      { label: "KV Cache", color: COLORS.kv, points: results.map(r => ({ x: r.sequence_length, y: r.kv_memory_bytes / 1000 })) },
      { label: "Synaptic", color: COLORS.syn, points: results.map(r => ({ x: r.sequence_length, y: r.synaptic_memory_bytes / 1000 })) },
    ],
    yFormat: v => v.toFixed(0) + "KB"
  });

  const last = results[results.length - 1];
  document.getElementById("mc-kv-acc").textContent = fmtPct(last.kv_accuracy);
  document.getElementById("mc-syn-acc").textContent = fmtPct(last.synaptic_accuracy);
  document.getElementById("mc-kv-mem").textContent = fmtBytes(last.kv_memory_bytes);
  document.getElementById("mc-syn-mem").textContent = fmtBytes(last.synaptic_memory_bytes);
}

/* ---------------- Recall Lab ---------------- */
async function runRecall() {
  const factsRaw = document.getElementById("recall-facts").value;
  const query = document.getElementById("recall-query").value.trim();
  const distractors = parseInt(document.getElementById("recall-distractors").value);
  const banner = document.getElementById("recall-banner");

  const facts = factsRaw.split("\n").map(f => f.trim()).filter(Boolean);
  if (!facts.length || !query) {
    banner.className = "result-banner bad";
    banner.textContent = "Add at least one fact and a query.";
    return;
  }

  banner.className = "result-banner";
  banner.textContent = "Running…";

  try {
    const data = await apiPost("/recall/test", { facts, query, num_distractors: distractors, dim: 64, decay: 0.99, lr: 1.0, seed: 42 });
    const kv = data.kv_result, syn = data.synaptic_result;
    const kvOk = facts.includes(kv.retrieved_fact);
    const synOk = facts.includes(syn.retrieved_fact) && syn.retrieved_fact === kv.retrieved_fact;
    banner.className = "result-banner " + (syn.retrieved_fact ? "ok" : "bad");
    banner.innerHTML = `
      <div class="legend" style="margin-bottom:8px;">
        <span><span class="dot kv"></span>KV Cache retrieved: <b>${escapeHtml(kv.retrieved_fact)}</b> (sim ${kv.similarity.toFixed(2)})</span>
      </div>
      <div class="legend">
        <span><span class="dot syn"></span>Synaptic retrieved: <b>${escapeHtml(syn.retrieved_fact)}</b> (sim ${syn.similarity.toFixed(2)})</span>
      </div>
      <div class="note">Sequence length: ${data.sequence_length} facts. ${data.note}</div>
    `;
  } catch (e) {
    banner.className = "result-banner bad";
    banner.textContent = "Backend unreachable. Deploy/start the FastAPI backend and set window.MEMORY_FRONTIER_API to its URL.";
  }
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

/* ---------------- Forgetting Lab ---------------- */
async function updateForgetting() {
  const decay = parseFloat(document.getElementById("fg-decay").value);
  document.getElementById("fg-decay-val").textContent = decay.toFixed(3);

  let curve;
  try {
    curve = await apiPost("/memory/retention", { decay, steps: 200 });
  } catch (e) {
    const t = Array.from({length: 200}, (_, i) => i);
    curve = { t, retention: t.map(i => Math.pow(decay, i)) };
  }
  const canvas = document.getElementById("chart-retention");
  drawChart(canvas, {
    series: [{ label: "retention", color: COLORS.syn, points: curve.t.map((t, i) => ({ x: t, y: curve.retention[i] })) }],
    yMax: 1.05, yFormat: v => (v * 100).toFixed(0) + "%"
  });

  // also show accuracy-vs-length for this decay using precomputed sweeps if close match, else live
  try {
    const data = await apiPost("/benchmark/run", {
      sequence_lengths: [50, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000],
      dim: 64, decay, lr: 1.0, num_target_facts: 20, seed: 42
    });
    const canvas2 = document.getElementById("chart-fg-accuracy");
    drawChart(canvas2, {
      series: [
        { label: "Synaptic accuracy", color: COLORS.syn, points: data.results.map(r => ({ x: r.sequence_length, y: r.synaptic_accuracy })) },
      ],
      yMax: 1.05, yFormat: v => (v * 100).toFixed(0) + "%"
    });
  } catch (e) { /* skip if backend unreachable; retention curve above still works offline */ }
}

/* ---------------- Judge Demo ---------------- */
async function runJudgeDemo() {
  const out = document.getElementById("demo-output");
  out.innerHTML = '<p class="loading-text">Running reproducible benchmark…</p>';
  try {
    const data = await apiGet("/experiment/demo");
    const rows = data.results.map(r => `
      <tr>
        <td class="num">${r.sequence_length}</td>
        <td class="num">${fmtPct(r.kv_accuracy)}</td>
        <td class="num">${fmtPct(r.synaptic_accuracy)}</td>
        <td class="num">${fmtBytes(r.kv_memory_bytes)}</td>
        <td class="num">${fmtBytes(r.synaptic_memory_bytes)}</td>
      </tr>`).join("");
    out.innerHTML = `
      <div class="claim-box" style="margin-bottom:18px;">
        <span class="tag">CLAIM TESTED</span>
        ${escapeHtml(data.claim)}
      </div>
      <table class="cmp">
        <thead><tr><th>Sequence length</th><th>KV accuracy</th><th>Synaptic accuracy</th><th>KV memory</th><th>Synaptic memory</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
  } catch (e) {
    const pre = await loadPrecomputed();
    if (pre) {
      out.innerHTML = '<p class="note">Backend unreachable — showing precomputed benchmark from experiments/benchmark.py instead (same code path, run ahead of time).</p>';
    } else {
      out.innerHTML = '<p class="note">Backend unreachable and no precomputed data bundled.</p>';
    }
  }
}

/* ---------------- init ---------------- */
window.addEventListener("DOMContentLoaded", () => {
  animateHero();

  document.getElementById("pg-run")?.addEventListener("click", runPlayground);
  ["pg-dim","pg-decay","pg-lr","pg-target"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.addEventListener("input", () => {
      const disp = document.getElementById(id + "-val");
      if (disp) disp.textContent = el.value;
    });
  });

  document.getElementById("recall-run")?.addEventListener("click", runRecall);

  const fgSlider = document.getElementById("fg-decay");
  if (fgSlider) {
    fgSlider.addEventListener("input", updateForgetting);
    updateForgetting();
  }

  document.getElementById("demo-run")?.addEventListener("click", runJudgeDemo);

  // initial playground run with defaults, best-effort
  if (document.getElementById("chart-accuracy")) runPlayground();
});
