/* img-iter-agent 打分台 — 前端 SPA (Vanilla JS, 无依赖)
 * 基于侧边栏导航的极简设计系统重构版。
 * 视图：总览 / 运行中 / loop 详情 / 人工排序 / Agent 设置
 */
"use strict";

const STATUS_LABEL = {
  finished: "已结束",
  running: "运行中",
  awaiting_review: "等审批",
  error: "错误",
  unknown: "未知",
  idle: "未开始",
};

const STATUS_ORDER = { awaiting_review: 0, running: 1, error: 2, finished: 3, unknown: 4, idle: 5 };

// 可一键续跑的状态：等审批 或 已结束（resume() 现已支持 finished，并能收养无 handle 的外部 loop）。
// running/error/unknown 不给续——外部进程可能正写，靠 status 门控避免并发写冲突。
const RESUMABLE = ["awaiting_review", "finished"];
// 一个 sample 下挑「最近一个可续跑 loop」（按 started_at 倒序）供 sample 卡片「继续跑」用。
const latestResumableLoop = (loops) => (loops || [])
  .filter((l) => RESUMABLE.includes(l.status))
  .sort((a, b) => String(b.started_at || "").localeCompare(String(a.started_at || "")))[0];

const DISTILL_LABEL = { idle: "未蒸馏", running: "蒸馏中", done: "完成", error: "失败", no_runs: "无 run" };

const NODE_LABEL = {
  generator: { name: "Generator", desc: "生成提示词 + 出图" },
  critic: { name: "Critic", desc: "多维度打分" },
  summarizer: { name: "Summarizer", desc: "归纳经验 + 写记录" },
  human_review: { name: "等审批", desc: "等待人工裁决" },
};

// 节点分段指示（替代写死的假进度条）：生成 → 打分 → 审批，按 current_node 标记进行到哪一步。
const NODE_SEGMENTS = [
  { key: "generator", name: "生成" },
  { key: "critic", name: "打分" },
  { key: "human_review", name: "审批" },
];

function nodeSegmentIndex(currentNode) {
  if (!currentNode) return -1;
  if (currentNode === "summarizer") return 1; // 归入「打分」组
  return NODE_SEGMENTS.findIndex((s) => s.key === currentNode);
}

function renderNodeSegments(currentNode) {
  const cur = nodeSegmentIndex(currentNode);
  return `<div class="node-segments">${NODE_SEGMENTS.map((s, i) => {
    const cls = cur < 0 ? "" : i < cur ? "done" : i === cur ? "active" : "";
    const mark = i < cur ? "✓ " : "";
    const arrow = i < NODE_SEGMENTS.length - 1 ? `<span class="node-seg-arrow">›</span>` : "";
    return `<span class="node-seg ${cls}">${mark}${esc(s.name)}</span>${arrow}`;
  }).join("")}</div>`;
}

// 路由缓存
window.__overviewCache = null;
window.__loopCache = { current: null };

// 紧凑模式（图片优先、文字折叠）—— 默认开启，除非用户显式关闭
window.__compactMode = localStorage.getItem("img-iter-compact") !== "0";

function toggleCompactMode() {
  window.__compactMode = !window.__compactMode;
  localStorage.setItem("img-iter-compact", window.__compactMode ? "1" : "0");
  document.body.classList.toggle("compact-mode", window.__compactMode);
  const dot = document.querySelector(".compact-dot");
  if (dot) dot.classList.toggle("on", window.__compactMode);
  router();
}

function initCompactMode() {
  document.body.classList.toggle("compact-mode", window.__compactMode);
  const dot = document.querySelector(".compact-dot");
  if (dot) dot.classList.toggle("on", window.__compactMode);
}

// ---- API helpers ----
async function api(path, opts = {}) {
  const r = await fetch("/api" + path, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) {
    const t = await r.text().catch(() => r.statusText);
    throw new Error(`${r.status} ${t}`);
  }
  return r.status === 204 ? null : r.json();
}

function imgURL(path, loop) {
  const q = new URLSearchParams({ path });
  if (loop) q.set("loop", loop);
  return "/api/static/img?" + q.toString();
}

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (n, d = 3) => (n == null ? "—" : Number(n).toFixed(d));

// ---- Markdown 渲染（安全子集：标题/加粗/列表/任务列表）----
function renderMarkdown(md) {
  if (!md) return "";
  const lines = md.replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let listItems = [];
  let inPara = false;

  function flushList() {
    if (!listItems.length) return;
    const allTask = listItems.every((i) => i.startsWith('<li class="task-item"'));
    const cls = allTask ? "md-list task-list" : "md-list";
    out.push(`<ul class="${cls}">${listItems.join("")}</ul>`);
    listItems = [];
  }

  function flushPara() {
    if (!inPara) return;
    out.push("</p>");
    inPara = false;
  }

  function inline(s) {
    return esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  }

  for (const raw of lines) {
    const line = raw.trimEnd();
    if (!line.trim()) {
      flushList();
      flushPara();
      continue;
    }

    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      flushList();
      flushPara();
      const level = h[1].length;
      out.push(`<h${level} class="md-h${level}">${inline(h[2])}</h${level}>`);
      continue;
    }

    const task = line.match(/^-\s+\[([ xX])\]\s+(.*)$/);
    if (task) {
      const checked = task[1].toLowerCase() === "x" ? " checked" : "";
      listItems.push(`<li class="task-item"><span class="task-checkbox${checked}"></span>${inline(task[2])}</li>`);
      continue;
    }

    const li = line.match(/^-\s+(.*)$/);
    if (li) {
      listItems.push(`<li>${inline(li[1])}</li>`);
      continue;
    }

    flushList();
    if (!inPara) {
      out.push('<p class="md-p">');
      inPara = true;
    } else {
      out.push("<br>");
    }
    out.push(inline(line));
  }

  flushList();
  flushPara();
  return out.join("");
}

// ---- Toast ----
function toast(message, type = "default") {
  const stack = document.getElementById("toast-stack");
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  stack.appendChild(el);
  setTimeout(() => {
    el.style.opacity = "0";
    el.style.transform = "translateX(20px)";
    el.style.transition = "opacity .2s ease, transform .2s ease";
    setTimeout(() => el.remove(), 220);
  }, 2600);
}

// ---- Modal ----
function openModal(html) {
  document.getElementById("modal-body").innerHTML = html;
  document.getElementById("modal").classList.remove("hidden");
}

function closeModal() {
  document.getElementById("modal").classList.add("hidden");
}

document.getElementById("modal").addEventListener("click", (e) => {
  if (e.target.dataset.close !== undefined) closeModal();
});

// ---- Lightbox ----
const lb = document.getElementById("lightbox");
const lbImg = document.getElementById("lightbox-img");
const lbStage = document.getElementById("lightbox-stage");

let lbState = { scale: 1, tx: 0, ty: 0, dragging: false, dragMoved: false, lastX: 0, lastY: 0 };

function applyLightboxTransform(noTransition = false) {
  lbImg.classList.toggle("no-transition", noTransition);
  lbImg.style.setProperty("--zoom", lbState.scale.toFixed(3));
  lbImg.style.setProperty("--pan-x", `${lbState.tx}px`);
  lbImg.style.setProperty("--pan-y", `${lbState.ty}px`);
}

function clampPan() {
  if (lbState.scale <= 1) {
    lbState.tx = 0;
    lbState.ty = 0;
    return;
  }
  const rect = lbImg.getBoundingClientRect();
  const stageRect = lbStage.getBoundingClientRect();
  const maxX = Math.max(0, (rect.width - stageRect.width) / 2);
  const maxY = Math.max(0, (rect.height - stageRect.height) / 2);
  lbState.tx = Math.max(-maxX, Math.min(maxX, lbState.tx));
  lbState.ty = Math.max(-maxY, Math.min(maxY, lbState.ty));
}

function fitLightbox() {
  lbState.scale = 1;
  lbState.tx = 0;
  lbState.ty = 0;
  applyLightboxTransform();
}

function zoomLightbox(delta, noTransition = false) {
  lbState.scale = Math.max(1, Math.min(5, lbState.scale + delta));
  clampPan();
  applyLightboxTransform(noTransition);
}

function openLightbox(src) {
  lbImg.src = src;
  lb.classList.remove("hidden");
  fitLightbox();
}

function closeLightbox() {
  lb.classList.add("hidden");
  lbImg.src = "";
  fitLightbox();
}

lb.addEventListener("click", (e) => {
  if (e.target.dataset.close !== undefined) closeLightbox();
});

document.getElementById("lb-zoom-in").addEventListener("click", (e) => { e.stopPropagation(); zoomLightbox(0.5); });
document.getElementById("lb-zoom-out").addEventListener("click", (e) => { e.stopPropagation(); zoomLightbox(-0.5); });
document.getElementById("lb-zoom-reset").addEventListener("click", (e) => { e.stopPropagation(); fitLightbox(); });
document.getElementById("lb-close").addEventListener("click", (e) => { e.stopPropagation(); closeLightbox(); });

lbImg.addEventListener("click", (e) => {
  e.stopPropagation();
  if (lbState.dragMoved) { lbState.dragMoved = false; return; }
  if (lbState.scale > 1) fitLightbox();
  else zoomLightbox(1);
});

lbStage.addEventListener("wheel", (e) => {
  e.preventDefault();
  const delta = e.deltaY > 0 ? -0.25 : 0.25;
  zoomLightbox(delta, true);
}, { passive: false });

lbStage.addEventListener("mousedown", (e) => {
  if (lbState.scale <= 1) return;
  lbState.dragging = true;
  lbState.dragMoved = false;
  lbState.lastX = e.clientX;
  lbState.lastY = e.clientY;
  lbStage.classList.add("dragging");
});

document.addEventListener("mousemove", (e) => {
  if (!lbState.dragging) return;
  const dx = e.clientX - lbState.lastX;
  const dy = e.clientY - lbState.lastY;
  if (Math.abs(dx) > 2 || Math.abs(dy) > 2) lbState.dragMoved = true;
  lbState.lastX = e.clientX;
  lbState.lastY = e.clientY;
  lbState.tx += dx;
  lbState.ty += dy;
  clampPan();
  applyLightboxTransform(true);
});

document.addEventListener("mouseup", () => {
  if (!lbState.dragging) return;
  lbState.dragging = false;
  lbStage.classList.remove("dragging");
});

document.addEventListener("keydown", (e) => {
  if (lb.classList.contains("hidden")) return;
  if (e.key === "Escape") closeLightbox();
  if (e.key === "+" || e.key === "=") zoomLightbox(0.5);
  if (e.key === "-") zoomLightbox(-0.5);
  if (e.key === "0") fitLightbox();
});

// 全局图片点击委托
document.addEventListener("click", (e) => {
  const img = e.target.closest("img[data-lb]");
  if (img) { e.stopPropagation(); openLightbox(img.dataset.lb); }
});

// ---- Navigation ----
function setNavActive(nav) {
  document.querySelectorAll(".nav-item").forEach((a) => a.classList.toggle("active", a.dataset.nav === nav));
}

function setTitle(title, actionsHTML = "") {
  document.getElementById("page-title").textContent = title;
  document.getElementById("topbar-actions").innerHTML = actionsHTML;
}

// ---- Loading ----
function skeleton(count = 3) {
  return Array.from({ length: count }, () => `<div class="skeleton" style="height:64px;margin-bottom:12px"></div>`).join("");
}

// ---- prompt diff ----
function diffPrompt(prev, cur) {
  const toks = (s) => (s || "").split(/(\s+)/);
  const a = toks(prev), b = toks(cur);
  const n = a.length, m = b.length;
  const dp = Array.from({ length: n + 1 }, () => new Int16Array(m + 1));
  for (let i = n - 1; i >= 0; i--)
    for (let j = m - 1; j >= 0; j--)
      dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] === b[j]) { out.push(`<span>${esc(b[j])}</span>`); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push(`<del>${esc(a[i])}</del>`); i++; }
    else { out.push(`<ins>${esc(b[j])}</ins>`); j++; }
  }
  while (i < n) out.push(`<del>${esc(a[i++])}</del>`);
  while (j < m) out.push(`<ins>${esc(b[j++])}</ins>`);
  return out.join("");
}

function togglePromptBox(idx) {
  const el = document.getElementById(`prompt-box-${idx}`);
  if (el) el.hidden = !el.hidden;
}

function showPromptDiff(idx) {
  const loop = window.__loopCache.current;
  if (!loop || idx <= 0) return;
  const prev = loop.traces[idx - 1], cur = loop.traces[idx];
  const box = document.getElementById(`prompt-box-${idx}`);
  const btn = document.querySelector(`button[data-pdiff="${idx}"]`);
  if (!box) return;
  if (box.dataset.diffing === "1") {
    box.dataset.diffing = "";
    box.innerHTML = `<pre>${esc(cur.prompt || "(空)")}</pre>`;
    if (btn) btn.textContent = `对比第${prev.round}轮`;
    return;
  }
  box.dataset.diffing = "1";
  box.hidden = false;
  box.innerHTML = `<pre class="diff">${diffPrompt(prev.prompt, cur.prompt)}</pre>`;
  if (btn) btn.textContent = "还原 prompt";
}

// ---- 全局错误处理 ----
function handleError(e, context = "操作失败") {
  console.error(e);
  toast(`${context}: ${e.message}`, "error");
}

// ============ 视图：总览 ============
async function viewOverview() {
  setNavActive("overview");
  setTitle("总览", `<button class="btn btn-primary" onclick="startNewLoop()">＋ 启动新 loop</button>`);
  const app = document.getElementById("app");
  app.innerHTML = skeleton(4);

  try {
    const data = await api("/overview");
    window.__overviewCache = data;
    updatePendingCount(data);

    if (!data.benches.length) {
      app.innerHTML = `
        <div class="empty">
          <div>还没有 benchmark</div>
          <div class="muted">先准备数据目录，或启动一个 loop 开始体验。</div>
          <button class="btn btn-primary" onclick="startNewLoop()">启动新 loop</button>
        </div>`;
      return;
    }

    if (window.__compactMode) {
      app.innerHTML = renderOverviewCompact(data);
    } else {
      app.innerHTML = renderOverviewFull(data);
    }
  } catch (e) {
    handleError(e, "加载总览失败");
    app.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

function renderSampleCard(bench, sample) {
  const pendingCls = sample.pending === 0 ? "badge-success" : "badge-pending";
  const target = sample.loops[0]?.thumbnail ? imgURL(sample.loops[0].thumbnail, sample.loops[0].loop_id) : "";
  const loopChips = sample.loops
    .sort((a, b) => STATUS_ORDER[a.status] - STATUS_ORDER[b.status])
    .map((l) => {
      const best = l.best_restoration != null ? ` · 最佳 ${fmt(l.best_restoration)}` : "";
      return `<span class="loop-chip" onclick="location.hash='#/loop/${esc(l.loop_id)}'">
        <span class="dot ${l.status}"></span>
        ${esc(l.loop_id.replace(/.*-\d{4}-/, ""))}${best}
      </span>`;
    }).join("");

  // sample 卡片「继续跑」= 续最近一个可续跑 loop（一题多 loop 时取 started_at 最新的）
  const cont = latestResumableLoop(sample.loops);
  const contBtn = cont
    ? `<button class="btn btn-ghost btn-sm" onclick="resumeLoop('${esc(cont.loop_id)}','continue')">继续跑</button>`
    : "";

  return `<div class="sample-card">
    ${target ? `<img class="sample-thumb" src="${target}" data-lb="${target}" alt="">` : '<div class="sample-thumb"></div>'}
    <div class="sample-info">
      <div class="sample-title">
        <strong>${esc(sample.sample_id)}</strong>
        ${sample.product ? `<span class="muted">${esc(sample.product)}</span>` : ""}
        <span class="badge ${pendingCls}">${sample.pending === 0 ? "已排序" : `待排序 ${sample.pending}`}</span>
      </div>
      <div class="sample-meta">${sample.n_traces} trace · ${sample.loops.length} loop${sample.category ? " · " + esc(sample.category) : ""}</div>
      <div class="loop-inline">${loopChips}</div>
    </div>
    <div class="sample-actions">
      <a class="btn btn-secondary btn-sm" href="#/scoring/${esc(bench.bench_id)}/${esc(sample.sample_id)}">人工排序</a>
      ${contBtn}
    </div>
  </div>`;
}

function updatePendingCount(data) {
  let pending = 0, running = 0;
  for (const bench of data.benches) {
    for (const sample of bench.samples) {
      pending += sample.pending;
      for (const l of sample.loops) {
        if (l.status === "running" || l.status === "awaiting_review" || l.status === "error") running++;
      }
    }
  }
  const pEl = document.getElementById("nav-pending-count");
  if (pEl) { pEl.textContent = pending; pEl.hidden = pending === 0; }
  const rEl = document.getElementById("nav-running-count");
  if (rEl) { rEl.textContent = running; rEl.hidden = running === 0; }
}

function renderOverviewFull(data) {
  let html = "";
  for (const bench of data.benches) {
    const active = bench.samples.filter((s) => s.loops.length > 0);
    const idle = bench.samples.filter((s) => s.loops.length === 0);
    const hasActive = active.length > 0;
    const expanded = hasActive ? "open" : "";
    const expCount = bench.general_experience_count || 0;
    const expBadge = expCount > 0 ? ` <span class="badge badge-success">${expCount}</span>` : "";

    html += `<details class="bench-section card" ${expanded}>
      <summary class="bench-header">
        <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
        <h2>${esc(bench.bench_id)} ${bench.description ? "· " + esc(bench.description) : ""}</h2>
        <span class="badge">${bench.samples.length} sample</span>
        <button class="btn btn-secondary btn-sm exp-entry" title="跨 loop 通用经验（可移植 SKILL.md）" onclick="event.stopPropagation(); location.hash='#/experience/${esc(bench.bench_id)}'">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="vertical-align:-2px;margin-right:4px"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>
          通用经验${expBadge}
        </button>
      </summary>
      <div class="bench-body">`;

    if (hasActive) {
      for (const sample of active) html += renderSampleCard(bench, sample);
    } else {
      html += `<div class="card-padded muted">该 benchmark 还没有运行过的 loop。</div>`;
    }

    if (idle.length) {
      html += `<div class="idle-bank">
        <summary class="muted">可选题库（${idle.length} 道未运行）</summary>
        <div class="bank-list">
          ${idle.map((s) => `<button class="btn btn-ghost btn-sm" onclick="location.hash='#/scoring/${esc(bench.bench_id)}/${esc(s.sample_id)}'">${esc(s.sample_id)} ${s.product ? "· " + esc(s.product) : ""}</button>`).join("")}
        </div>
      </div>`;
    }

    html += `</div></details>`;
  }
  return html;
}

// 紧凑总览：按 sample 分组，每组内 loop 图卡横向滚动，图片优先
function renderOverviewCompact(data) {
  let html = "";
  for (const bench of data.benches) {
    const activeSamples = bench.samples.filter((s) => s.loops.length > 0);
    const idle = bench.samples.filter((s) => s.loops.length === 0);
    const expanded = activeSamples.length ? "open" : "";
    const expCount = bench.general_experience_count || 0;
    const expBadge = expCount > 0 ? ` <span class="badge badge-success">${expCount}</span>` : "";

    html += `<details class="bench-section card bench-compact" ${expanded}>
      <summary class="bench-header">
        <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
        <h2>${esc(bench.bench_id)}</h2>
        <span class="badge">${activeSamples.length} sample · ${activeSamples.reduce((n, s) => n + s.loops.length, 0)} loop</span>
        <button class="btn btn-ghost btn-sm exp-entry" title="跨 loop 通用经验（可移植 SKILL.md）" onclick="event.stopPropagation(); location.hash='#/experience/${esc(bench.bench_id)}'">通用经验${expBadge}</button>
      </summary>
      <div class="bench-body">`;

    if (activeSamples.length) {
      for (const sample of activeSamples) {
        html += `<div class="sample-group">
          <div class="sample-group-header">
            <span class="sample-group-id">${esc(sample.sample_id)}</span>
            <span class="sample-group-product">${sample.product ? esc(sample.product) : ""}</span>
            <span class="sample-group-meta">${sample.n_traces} trace · ${sample.loops.length} loop</span>
          </div>
          <div class="sample-group-grid">
            ${sample.loops.map((l) => renderLoopThumbCard(bench, sample, l)).join("")}
          </div>
        </div>`;
      }
    }

    if (idle.length) {
      html += `<div class="idle-bank">
        <summary class="muted">未运行 ${idle.length}</summary>
        <div class="bank-list">
          ${idle.map((s) => `<button class="btn btn-ghost btn-sm" onclick="location.hash='#/scoring/${esc(bench.bench_id)}/${esc(s.sample_id)}'">${esc(s.sample_id)}</button>`).join("")}
        </div>
      </div>`;
    }

    html += `</div></details>`;
  }
  return html;
}

function renderLoopThumbCard(bench, sample, loop) {
  const thumb = loop.thumbnail ? imgURL(loop.thumbnail, loop.loop_id) : "";
  const best = loop.best_restoration != null ? fmt(loop.best_restoration) : "—";
  const loopShort = esc(loop.loop_id.replace(/.*-\d{4}-/, ""));
  return `<div class="loop-thumb-card">
    <div class="loop-thumb-wrap" onclick="location.hash='#/loop/${esc(loop.loop_id)}'">
      ${thumb ? `<img class="loop-thumb" src="${thumb}" data-lb="${thumb}" alt="">` : '<div class="loop-thumb"></div>'}
      <span class="loop-thumb-score"><span class="dot ${loop.status}"></span>${best}</span>
    </div>
    <div class="loop-thumb-info">
      <div class="loop-thumb-title">
        <strong>${loopShort}</strong>
        <span class="muted">${esc(sample.sample_id)}${sample.product ? " · " + esc(sample.product) : ""}</span>
      </div>
      <div class="loop-thumb-meta">${loop.n_traces} trace · 最佳 ${best}</div>
      <div class="loop-thumb-actions">
        <a class="btn btn-secondary btn-sm" href="#/scoring/${esc(bench.bench_id)}/${esc(sample.sample_id)}">排序</a>
        ${RESUMABLE.includes(loop.status)
          ? `<button class="btn btn-ghost btn-sm" onclick="event.stopPropagation(); resumeLoop('${esc(loop.loop_id)}','continue')">继续跑</button>`
          : ""}
      </div>
    </div>
  </div>`;
}

// ============ 视图：运行中 ============
let _runningPollTimer = null;
let _runningLoaded = false;

async function viewRunning() {
  setNavActive("running");
  setTitle("运行中 / 待审批");
  if (_runningPollTimer) { clearInterval(_runningPollTimer); _runningPollTimer = null; }
  _runningLoaded = false;
  document.getElementById("app").innerHTML = skeleton(3);
  await loadRunning();
  // 后台 loop 状态会推进（running→awaiting_review→finished），定时刷新让页面如实反映。
  _runningPollTimer = setInterval(loadRunning, 3000);
}

async function loadRunning() {
  const app = document.getElementById("app");
  try {
    const data = await api("/overview");
    window.__overviewCache = data;
    updatePendingCount(data);  // 同步侧栏「运行中」计数

    const active = [];
    for (const bench of data.benches) {
      for (const sample of bench.samples) {
        for (const l of sample.loops) {
          if (l.status === "running" || l.status === "awaiting_review" || l.status === "error") {
            active.push({ bench, sample, loop: l });
          }
        }
      }
    }

    // running 的 loop 拉一次实时节点（Generator/Critic/Summarizer）+ 剩余轮数
    await Promise.all(
      active.filter((it) => it.loop.status === "running").map((it) =>
        api(`/loops/${encodeURIComponent(it.loop.loop_id)}/status`)
          .then((s) => { it.statusInfo = s; })
          .catch(() => {})
      )
    );

    _runningLoaded = true;
    app.innerHTML = active.length
      ? renderRunningCards(active)
      : `<div class="empty"><div>没有运行中或待审批的 loop</div><div class="muted">所有 loop 都已结束，或尚未启动。</div></div>`;
  } catch (e) {
    handleError(e, "加载运行中列表失败");
    if (!_runningLoaded) {
      app.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
    }
    // 已有内容时轮询瞬时失败不清空，下一拍自愈
  }
}

function renderRunningCards(active) {
  let html = `<div class="row gap-3">`;
  for (const { bench, sample, loop, statusInfo } of active) {
    let body = "";
    if (loop.status === "running") {
      const s = statusInfo;
      const node = s && s.current_node ? (NODE_LABEL[s.current_node] || { name: s.current_node, desc: "" }) : null;
      const nodeLine = node ? `<strong>${esc(node.name)}</strong> <span class="muted">${esc(node.desc)}</span>` : "执行中…";
      const remain = s && s.rounds_remaining > 0 ? ` · 剩 ${s.rounds_remaining} 轮` : "";
      const rnd = s && s.round != null ? `第 ${s.round} 轮 · ` : "";
      body = `${renderNodeSegments(s.current_node)}
        <div class="muted mt-2" style="font-size:12px">${rnd}${nodeLine}${remain}</div>
        <div class="mt-2"><a class="btn btn-secondary btn-sm" href="#/loop/${esc(loop.loop_id)}">查看详情</a></div>`;
    } else if (loop.status === "awaiting_review") {
      body = `<button class="btn btn-primary btn-sm" onclick="location.hash='#/loop/${esc(loop.loop_id)}'">去审批</button>`;
    } else if (loop.status === "error") {
      body = `<button class="btn btn-danger btn-sm" onclick="location.hash='#/loop/${esc(loop.loop_id)}'">查看错误</button>`;
    }
    html += `<div class="card card-padded" style="flex:1 1 320px;min-width:280px">
      <div class="row" style="justify-content:space-between">
        <span class="status-pill"><span class="dot ${loop.status}"></span>${esc(STATUS_LABEL[loop.status] || loop.status)}</span>
        <span class="muted mono" style="font-size:12px">${esc(loop.loop_id)}</span>
      </div>
      <h3 class="mt-3">${esc(sample.sample_id)} ${sample.product ? "· " + esc(sample.product) : ""}</h3>
      <div class="muted" style="font-size:12px">${esc(bench.bench_id)} · ${esc(loop.model)}</div>
      <div class="mt-3">${body}</div>
    </div>`;
  }
  return html + `</div>`;
}

// ============ 视图：loop 详情 ============
let _loopPollTimer = null;
let _distillPollTimer = null;
let _activityLastTotal = 0; // events.jsonl 行号游标（当前 loop；切 loop 时重置）
let _activityLLMCount = 0; // 当前 loop 的「思考」LLM 调用计数

// 渲染单个工具事件为一行活动（限量：参数默认折叠进 <details>，只露工具名 + 一行摘要）
function renderActivityTool(ev) {
  const mark = ev.status === "running"
    ? `<span class="act-spin">···</span>`
    : ev.status === "error"
    ? `<span class="act-err">✗</span>`
    : `<span class="act-ok">✓</span>`;
  const name = ev.tool
    ? `<span class="badge badge-ghost mono act-tool">${esc(ev.tool)}</span>`
    : `<span class="muted">工具</span>`;
  let text = "";
  if (ev.status === "done" && ev.result) text = esc(ev.result);
  else if (ev.status === "running" && ev.args) {
    text = esc(typeof ev.args === "string" ? ev.args : JSON.stringify(ev.args));
  } else if (ev.status === "error" && ev.result) text = esc(ev.result);
  const dur = ev.duration_ms != null ? `<span class="act-dur">${ev.duration_ms}ms</span>` : "";
  const hasArgs = ev.args && typeof ev.args === "object" && Object.keys(ev.args).length;
  const detail = (ev.status !== "running" && hasArgs)
    ? `<details class="act-detail"><summary>参数</summary><pre>${esc(JSON.stringify(ev.args, null, 2))}</pre></details>`
    : "";
  return `<div class="activity-item ${ev.status}">${mark}${name}<span class="act-text">${text}</span>${dur}${detail}</div>`;
}

// 增量拉取活动流事件并 append（LLM 调用折成计数，工具调用渲染为行）。复用 _loopPollTimer 那一拍。
async function pollLoopActivity(loopId) {
  const itemsEl = document.getElementById("act-items");
  if (!itemsEl) return; // 不在 loop 详情页 / 非 running（无面板）
  try {
    const data = await api(`/loops/${loopId}/events?since=${_activityLastTotal}`);
    const events = data && data.events ? data.events : [];
    if (!events.length) return;
    let llmDelta = 0;
    let html = "";
    for (const ev of events) {
      if (ev.type === "tool") html += renderActivityTool(ev);
      else if (ev.type === "llm" && ev.status === "done") llmDelta += 1;
    }
    if (html) itemsEl.insertAdjacentHTML("beforeend", html);
    if (llmDelta) {
      _activityLLMCount += llmDelta;
      const c = document.getElementById("act-llm-count");
      if (c) c.textContent = `思考 ${_activityLLMCount} 次`;
    }
    _activityLastTotal = data.total;
  } catch (_) { /* 瞬时失败静默，下一拍自愈 */ }
}

async function viewLoop(loopId) {
  if (_loopPollTimer) { clearInterval(_loopPollTimer); _loopPollTimer = null; }
  _activityLastTotal = 0;
  _activityLLMCount = 0;
  setNavActive("");
  setTitle("loop 详情");
  const app = document.getElementById("app");
  app.innerHTML = skeleton(4);

  try {
    const loop = await api(`/loops/${loopId}`);
    window.__loopCache.current = loop;
    const st = loop.status;

    setTitle(loopId, `<a class="btn btn-secondary btn-sm" href="#/scoring/${esc(loop.bench_id)}/${esc(loop.sample_id)}">人工排序</a>`);

    const html = window.__compactMode
      ? renderLoopCompact(loop, loopId)
      : renderLoopFull(loop, loopId);

    app.innerHTML = html;

    if (window.__compactMode) {
      initLoopCompact(loop);
    }

    // 人工提示词面板（异步填充）
    renderHintsPanel(loopId, loop.status);
    // Generator 本 loop 记忆面板（系统托管，可查看/编辑/清空）
    renderMemoryPanel(loopId);

    // 活动流首拉（running 时立即填充；后续靠 _loopPollTimer 增量）
    if (st === "running") pollLoopActivity(loopId);

    // prompt 事件委托（完整视图）
    app.addEventListener("click", (e) => {
      const t = e.target;
      if (t.dataset.ptog !== undefined) { togglePromptBox(+t.dataset.ptog); return; }
      if (t.dataset.pdiff !== undefined) { showPromptDiff(+t.dataset.pdiff); return; }
    });

    // 轮询
    if (st === "running" || st === "awaiting_review") {
      _loopPollTimer = setInterval(async () => {
        try {
          const s = await api(`/loops/${loopId}/status`);
          if (s.phase === "error" || s.phase === "finished") {
            clearInterval(_loopPollTimer); _loopPollTimer = null;
            viewLoop(loopId);
            return;
          }
          const nodeEl = document.getElementById("cur-node");
          if (nodeEl && s.current_node) {
            const n = NODE_LABEL[s.current_node] || { name: s.current_node, desc: "" };
            const extra = s.rounds_remaining > 0 ? ` · 剩 ${s.rounds_remaining} 轮` : "";
            nodeEl.innerHTML = `<strong>${esc(n.name)}</strong> ${esc(n.desc)}${extra}`;
          }
          const segEl = document.getElementById("node-segments");
          if (segEl) segEl.innerHTML = renderNodeSegments(s.current_node);
          await pollLoopActivity(loopId);
        } catch (_) {}
      }, 2000);
    }
  } catch (e) {
    handleError(e, "加载 loop 详情失败");
    app.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

// ============ loop 详情：完整视图 ============
function renderLoopFull(loop, loopId) {
  const bestIdx = loop.traces.length
    ? loop.traces.reduce((best, t, i) => ((t.verdict?.restoration ?? -1) > (loop.traces[best].verdict?.restoration ?? -1) ? i : best), 0)
    : -1;

  const controls = [];
  if (loop.status === "running") {
    controls.push(`<button class="btn btn-danger" onclick="resumeLoop('${esc(loopId)}','stop')">停止 loop</button>`);
  } else if (loop.status === "awaiting_review") {
    controls.push(`<button class="btn btn-primary" onclick="resumeLoop('${esc(loopId)}','continue')">继续跑</button>`);
    controls.push(`<button class="btn btn-danger" onclick="resumeLoop('${esc(loopId)}','stop')">停止并采用</button>`);
  } else if (loop.status === "finished") {
    controls.push(`<button class="btn btn-primary" onclick="resumeLoop('${esc(loopId)}','continue')">继续跑</button>`);
  }
  if (loop.status !== "running") {
    controls.push(`<button class="btn btn-danger" onclick="deleteLoop('${esc(loopId)}')">删除 loop</button>`);
  }

  let review = "";
  if (loop.status === "awaiting_review" && loop.interrupt_payload) {
    const p = loop.interrupt_payload;
    review = `<div class="review-card">
      <div class="review-info">
        <div class="review-title">等待人工审批</div>
        <div class="review-meta">${p.message || ""}${p.next_round ? ` · 下一轮 #${p.next_round}` : ""}</div>
      </div>
      <div class="review-actions">${controls.join("")}</div>
    </div>`;
  }

  let errorAlert = "";
  if (loop.status === "error" && loop.last_error) {
    errorAlert = `<div class="alert">
      <div class="alert-title">loop 出错</div>
      <pre>${esc(loop.last_error)}</pre>
    </div>`;
  }

  let targetCard = "";
  if (loop.target_image || loop.target_md) {
    targetCard = `<div class="card loop-target-card">
      <div class="loop-target-body card-padded">
        ${loop.target_image ? `<img src="${imgURL(loop.target_image)}" data-lb="${imgURL(loop.target_image)}" alt="target">` : ""}
        ${loop.target_md ? `<div class="target-desc">${renderMarkdown(loop.target_md)}</div>` : ""}
      </div>
    </div>`;
  }

  const statusBody = [];
  statusBody.push(`<span class="status-pill"><span class="dot ${loop.status}"></span>${esc(STATUS_LABEL[loop.status] || loop.status)}</span>`);
  statusBody.push(`<span class="muted mono">${esc(loop.model)}</span>`);
  if (loop.status === "running") {
    statusBody.push(`<div id="node-segments" class="node-segments"></div>`);
    statusBody.push(`<span class="muted" id="cur-node" style="font-size:12px"><strong>运行中</strong></span>`);
  }

  // Agent 活动流面板（仅 running 时显示；pollLoopActivity 增量填充）
  const activityPanel = loop.status === "running"
    ? `<div class="activity-stream card mt-3"><div class="activity-head"><strong>Agent 活动</strong> <span class="muted" id="act-llm-count">思考 0 次</span></div><div id="act-items" class="act-items"></div></div>`
    : "";

  let html = `
    <div class="breadcrumb mb-3"><a href="#/">总览</a><span>/</span><span>${esc(loop.bench_id)}</span><span>/</span><span>${esc(loop.sample_id)}</span></div>
    <div class="page-header">
      <div class="title-block">
        <div class="muted" style="font-size:12px">${esc(loopId)}</div>
        <h1>${esc(loop.sample_id)} ${loop.product ? `<span class="muted">${esc(loop.product)}</span>` : ""}</h1>
      </div>
      <div class="control-actions">${controls.join("")}</div>
    </div>
    <div class="status-bar">${statusBody.join("")}</div>
    ${activityPanel}
    ${errorAlert}
    ${review}
    ${targetCard}
    <div class="timeline">`;

  for (let i = 0; i < loop.traces.length; i++) {
    const t = loop.traces[i];
    const isBest = i === bestIdx && loop.traces.length > 1;
    const score = t.verdict?.restoration;
    const scoreCls = isBest ? "best" : "";
    const img = t.output_image_refs[0];
    html += `
      <div class="timeline-item ${isBest ? "best" : ""}">
        <div class="timeline-marker">${t.round}</div>
        <div class="timeline-body">
          <div class="trace-header">
            <span class="round">第 ${t.round} 轮</span>
            ${t.ts ? `<span class="muted" style="font-size:12px">${esc(t.ts)}</span>` : ""}
            <span class="score ${scoreCls}">还原度 ${fmt(score)}</span>
          </div>
          <div class="trace-content">
            ${img ? `<img class="trace-img" src="${imgURL(img, t.loop_id)}" data-lb="${imgURL(img, t.loop_id)}" alt="">` : '<div class="trace-img"></div>'}
            <div class="trace-detail">
              <div class="trace-params">
                ${t.test_variable ? `<div class="param"><span class="pk">变量</span><span class="pv">${esc(t.test_variable)}</span></div>` : ""}
                ${t.baseline_ref ? `<div class="param"><span class="pk">基线</span><span class="pv">${esc(t.baseline_ref)}</span></div>` : ""}
                ${t.gen_mode ? `<div class="param"><span class="pk">模式</span><span class="pv">${esc(t.gen_mode)}</span></div>` : ""}
                ${t.size ? `<div class="param"><span class="pk">尺寸</span><span class="pv">${esc(t.size)}</span></div>` : ""}
                ${t.human_rank != null ? `<div class="param"><span class="pk">人工排序</span><span class="pv">${fmt(t.human_rank)}</span></div>` : ""}
              </div>
              ${t.verdict ? renderDimList(t.verdict.dimensions) : ""}
              ${t.delta_note ? `<div class="dim-raw" style="margin-top:8px">改动：${esc(t.delta_note)}</div>` : ""}
              <div class="prompt-tools">
                <button class="btn btn-ghost btn-sm" data-ptog="${i}">查看 prompt</button>
                ${i > 0 ? `<button class="btn btn-ghost btn-sm" data-pdiff="${i}">对比第${loop.traces[i - 1].round}轮</button>` : ""}
                ${loop.status !== "running" ? `<button class="btn btn-danger btn-sm" onclick="deleteAttempt('${esc(loopId)}','${esc(t.trace_id)}',${t.round})">删除该轮</button>` : ""}
              </div>
              <div class="prompt-box" id="prompt-box-${i}" hidden><pre>${esc(t.prompt || "(空)")}</pre></div>
            </div>
          </div>
        </div>
      </div>`;
  }

  html += `</div>`;

  if (loop.conclusions && loop.conclusions.length) {
    html += `<h2 class="mt-4">经验结论</h2>${renderConclusions(loop.conclusions)}`;
  }

  html += `<div id="hints-panel" class="mt-4"></div>`;
  html += `<div id="memory-panel" class="mt-4"></div>`;
  return html;
}

// 紧凑 loop 详情：大图优先，文字折叠在详情里
function renderLoopCompact(loop, loopId) {
  const bestIdx = loop.traces.length
    ? loop.traces.reduce((best, t, i) => ((t.verdict?.restoration ?? -1) > (loop.traces[best].verdict?.restoration ?? -1) ? i : best), 0)
    : 0;

  const controls = [];
  if (loop.status === "running") {
    controls.push(`<button class="btn btn-danger" onclick="resumeLoop('${esc(loopId)}','stop')">停止 loop</button>`);
  } else if (loop.status === "awaiting_review") {
    controls.push(`<button class="btn btn-primary" onclick="resumeLoop('${esc(loopId)}','continue')">继续跑</button>`);
    controls.push(`<button class="btn btn-danger" onclick="resumeLoop('${esc(loopId)}','stop')">停止并采用</button>`);
  } else if (loop.status === "finished") {
    controls.push(`<button class="btn btn-primary" onclick="resumeLoop('${esc(loopId)}','continue')">继续跑</button>`);
  }
  if (loop.status !== "running") {
    controls.push(`<button class="btn btn-danger" onclick="deleteLoop('${esc(loopId)}')">删除 loop</button>`);
  }

  let errorAlert = "";
  if (loop.status === "error" && loop.last_error) {
    errorAlert = `<div class="alert">
      <div class="alert-title">loop 出错</div>
      <pre>${esc(loop.last_error)}</pre>
    </div>`;
  }

  const statusLine = `<span class="status-pill"><span class="dot ${loop.status}"></span>${esc(STATUS_LABEL[loop.status] || loop.status)}</span>
    <span class="muted">${esc(loop.sample_id)}${loop.product ? " · " + esc(loop.product) : ""}</span>
    <span class="muted mono">${esc(loop.model)}</span>`;

  const strip = loop.traces.map((t, i) => {
    const img = t.output_image_refs[0];
    const score = t.verdict?.restoration;
    const isBest = i === bestIdx && loop.traces.length > 1;
    return `<div class="trace-item ${i === bestIdx ? "selected current" : ""}" data-trace-idx="${i}" tabindex="0" role="button" aria-label="查看第 ${t.round} 轮">
      ${img ? `<img class="trace-thumb" src="${imgURL(img, t.loop_id)}" alt="">` : '<div class="trace-thumb"></div>'}
      <span class="trace-idx">#${t.round}</span>
      ${score != null ? `<span class="trace-score">${fmt(score)}</span>` : ""}
      ${isBest ? `<span class="trace-tag best">最佳</span>` : ""}
    </div>`;
  }).join("");

  const compareOptions = loop.traces.map((t, i) => `<option value="${i}">#${t.round}</option>`).join("");

  return `
    <div class="breadcrumb mb-3"><a href="#/">总览</a><span>/</span><span>${esc(loop.bench_id)}</span><span>/</span><span>${esc(loop.sample_id)}</span></div>
    <div class="compact-layout">
      <div class="status-bar compact-status">${statusLine}<div class="control-actions">${controls.join("")}</div></div>
      ${errorAlert}

      <div class="loop-hero card">
        <div class="hero-img-wrap">
          <img id="compact-hero-img" src="" data-lb="" alt="">
          <span class="hero-badge" id="compact-hero-badge"></span>
        </div>
        <div class="hero-meta">
          <div>
            <div class="hero-title">当前选中</div>
            <div class="hero-sub" id="compact-hero-sub"></div>
          </div>
          <a class="btn btn-secondary btn-sm" href="#/scoring/${esc(loop.bench_id)}/${esc(loop.sample_id)}">去排序</a>
          ${loop.status !== "running" ? `<button class="btn btn-danger btn-sm" onclick="deleteCompactCurrentRound('${esc(loopId)}')">删除当前轮</button>` : ""}
        </div>
      </div>

      <div class="card bench-section">
        <div class="bench-header" style="cursor:default">
          <h2>迭代历史</h2>
          <span class="badge badge-primary">${loop.traces.length} 轮</span>
        </div>
        <div class="trace-strip no-scrollbar" id="compact-strip">${strip}</div>
      </div>

      <details class="card bench-section">
        <summary class="bench-header">
          <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
          <h2>Critic 评分</h2>
          <span class="eval-score" id="compact-eval-score">—</span>
        </summary>
        <div id="compact-eval-body"></div>
      </details>

      <details class="card bench-section">
        <summary class="bench-header">
          <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
          <h2>Prompt & 对比</h2>
        </summary>
        <div id="compact-prompt-body">
          <div class="log-list"><pre id="compact-prompt-text"></pre></div>
          <div class="compare-bar">
            <select id="compact-base" class="select-sm">${compareOptions}</select>
            <span class="muted">vs</span>
            <select id="compact-target" class="select-sm">${compareOptions}</select>
          </div>
          <div class="diff-view" id="compact-diff-view"></div>
          <div class="diff-summary" id="compact-diff-summary"></div>
        </div>
      </details>

      ${loop.conclusions && loop.conclusions.length ? `<details class="card bench-section">
        <summary class="bench-header">
          <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
          <h2>经验结论</h2>
          <span class="badge">${loop.conclusions.length}</span>
        </summary>
        ${renderConclusions(loop.conclusions)}
      </details>` : ""}

      <div id="hints-panel"></div>
      <div id="memory-panel" class="mt-4"></div>
    </div>`;
}

function initLoopCompact(loop) {
  const strip = document.getElementById("compact-strip");
  if (!strip) return;
  const baseSel = document.getElementById("compact-base");
  const targetSel = document.getElementById("compact-target");

  const bestIdx = loop.traces.length
    ? loop.traces.reduce((best, t, i) => ((t.verdict?.restoration ?? -1) > (loop.traces[best].verdict?.restoration ?? -1) ? i : best), 0)
    : 0;

  const diffPrompt = (prev, cur) => {
    const toks = (s) => (s || "").split(/(\s+)/);
    const a = toks(prev), b = toks(cur);
    const n = a.length, m = b.length;
    const dp = Array.from({ length: n + 1 }, () => new Int16Array(m + 1));
    for (let i = n - 1; i >= 0; i--)
      for (let j = m - 1; j >= 0; j--)
        dp[i][j] = a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    const out = [];
    let i = 0, j = 0;
    while (i < n && j < m) {
      if (a[i] === b[j]) { out.push(`<span>${esc(b[j])}</span>`); i++; j++; }
      else if (dp[i + 1][j] >= dp[i][j + 1]) { out.push(`<del>${esc(a[i])}</del>`); i++; }
      else { out.push(`<ins>${esc(b[j])}</ins>`); j++; }
    }
    while (i < n) out.push(`<del>${esc(a[i++])}</del>`);
    while (j < m) out.push(`<ins>${esc(b[j++])}</ins>`);
    return out.join("");
  };

  const renderCompactDiff = () => {
    const base = loop.traces[+baseSel.value];
    const target = loop.traces[+targetSel.value];
    if (!base || !target) return;
    const view = document.getElementById("compact-diff-view");
    const summary = document.getElementById("compact-diff-summary");
    if (base.trace_id === target.trace_id) {
      if (view) view.innerHTML = "";
      if (summary) summary.textContent = "选择不同轮次进行对比";
      return;
    }
    const ops = diffPrompt(base.prompt || "", target.prompt || "");
    if (view) view.innerHTML = ops || "<span class=\"muted\">prompt 相同</span>";
    if (summary) summary.textContent = `${base.prompt ? base.prompt.length : 0} → ${target.prompt ? target.prompt.length : 0} 字符`;
  };

  const selectTrace = (idx) => {
    const t = loop.traces[idx];
    if (!t) return;
    strip.querySelectorAll(".trace-item").forEach((el) => el.classList.toggle("selected", +el.dataset.traceIdx === idx));
    const img = t.output_image_refs[0];
    const heroImg = document.getElementById("compact-hero-img");
    if (heroImg) {
      heroImg.src = img ? imgURL(img, t.loop_id) : "";
      heroImg.dataset.lb = img ? imgURL(img, t.loop_id) : "";
    }
    const best = loop.traces.reduce((best, tr) => ((tr.verdict?.restoration ?? -1) > (best.verdict?.restoration ?? -1) ? tr : best), loop.traces[0]);
    const isBest = t.trace_id === best.trace_id;
    const badge = document.getElementById("compact-hero-badge");
    if (badge) badge.textContent = `#${t.round}${isBest ? " 最佳" : ""} · ${fmt(t.verdict?.restoration)}`;
    const sub = document.getElementById("compact-hero-sub");
    if (sub) sub.textContent = `${esc(loop.model)} · ${t.ts || ""}`;
    const evalScore = document.getElementById("compact-eval-score");
    if (evalScore) evalScore.textContent = fmt(t.verdict?.restoration);
    const evalBody = document.getElementById("compact-eval-body");
    if (evalBody) evalBody.innerHTML = t.verdict ? renderDimList(t.verdict.dimensions) : '<div class="muted" style="padding:0 16px 16px">暂无评分</div>';
    const promptText = document.getElementById("compact-prompt-text");
    if (promptText) promptText.textContent = t.prompt || "(空)";
    if (baseSel) baseSel.value = String(Math.max(0, idx - 1));
    if (targetSel) targetSel.value = String(idx);
    renderCompactDiff();
  };

  strip.addEventListener("click", (e) => {
    const item = e.target.closest(".trace-item");
    if (!item) return;
    selectTrace(+item.dataset.traceIdx);
  });
  strip.addEventListener("keydown", (e) => {
    const item = e.target.closest(".trace-item");
    if (!item || (e.key !== "Enter" && e.key !== " ")) return;
    e.preventDefault();
    selectTrace(+item.dataset.traceIdx);
  });
  if (baseSel) baseSel.addEventListener("change", renderCompactDiff);
  if (targetSel) targetSel.addEventListener("change", renderCompactDiff);

  selectTrace(bestIdx);
}

function renderDimList(dimensions) {
  // 二分维度逐项渲染：✓/✗ + id + reason（含通过项，Critic 对每项都给理由）。
  // 连续维度无 items，回落到 raw（一句理由）。
  const itemRow = (it) => {
    const mark = it.passed ? "✓" : "✗";
    const cls = it.passed ? "pass" : "fail";
    return `<div class="cj cj-${cls}">
      <span class="cj-mark cj-${cls}">${mark}</span>
      <span class="cj-id">${esc(it.id)}</span>
      ${it.reason ? `<span class="cj-reason">${esc(it.reason)}</span>` : ""}
    </div>`;
  };
  return `<div class="dim-list">
    ${dimensions.map((d) => `
      <div class="dim">
        <span>${esc(d.dim)} <span class="muted">(${esc(d.scoring_type)})</span></span>
        <span class="v">${fmt(d.value)}</span>
      </div>
      ${(d.items && d.items.length)
        ? d.items.map(itemRow).join("")
        : (d.raw ? `<div class="dim-raw">${esc(d.raw)}</div>` : "")}
    `).join("")}
  </div>`;
}

function renderConclusions(conclusions) {
  const eff = conclusions.filter((c) => c.status === "verified_effective");
  const inef = conclusions.filter((c) => c.status === "ineffective");
  const pend = conclusions.filter((c) => c.status === "pending");
  const renderC = (c) => {
    const ev = c.critic_evidence;
    const evStr = ev ? `<div class="c-evidence">${ev.verdict_delta}<br><span class="muted">Critic: ${ev.before.reason || "—"} → ${ev.after.reason || "—"}</span></div>` : "";
    return `<div class="conclusion-item">
      <div class="c-dim">[${esc(c.dim)}] ${esc(c.finding || "")}</div>
      <div class="c-change">改动：${esc(c.change)}</div>
      ${c.lesson ? `<div class="c-lesson">${esc(c.lesson)}</div>` : ""}
      ${evStr}
      <div class="muted" style="font-size:11px">提出 r${c.created_round}${c.verified_round ? ` · 验证 r${c.verified_round}` : ""}</div>
    </div>`;
  };
  const group = (title, list, cls) => list.length ? `<div class="conclusion-group"><div class="conclusion-group-h ${cls}">${title}（${list.length}）</div>${list.map(renderC).join("")}</div>` : "";
  return group("已验证有效", eff, "eff") + group("验证无效", inef, "inef") + group("待验证", pend, "pend");
}

async function resumeLoop(loopId, decision) {
  if (decision === "stop" && !confirm("确定停止这个 loop？")) return;
  try {
    await api(`/loops/${loopId}/resume`, { method: "POST", body: JSON.stringify({ decision }) });
    toast(decision === "stop" ? "已停止 loop" : "已继续 loop", "success");
    viewLoop(loopId);
  } catch (e) { handleError(e, "操作失败"); }
}

// ---- 删除 loop / 删除单轮 attempt ----
async function deleteLoop(loopId) {
  if (!confirm(`确定删除 loop ${loopId}？\n该 loop 的所有轮次与 loop 内经验（conclusions.json 等）都会删除；跨 loop 蒸馏 skill 包保留。`)) return;
  try {
    await api(`/loops/${loopId}`, { method: "DELETE" });
    toast(`已删除 loop ${loopId}`, "success");
    location.hash = "#/";
  } catch (e) { handleError(e, "删除失败"); }
}

async function deleteAttempt(loopId, traceId, round) {
  if (!confirm(`删除第 ${round ?? "?"} 轮？该轮的图、prompt、评分都会移除。`)) return;
  try {
    await api(`/loops/${loopId}/attempts/${encodeURIComponent(traceId)}`, { method: "DELETE" });
    toast(`已删除第 ${round ?? "?"} 轮`, "success");
    viewLoop(loopId);
  } catch (e) { handleError(e, "删除失败"); }
}

// 紧凑模式：删当前选中轮（从 DOM 的 selected .trace-item 读 idx → 映射到 trace_id）
function deleteCompactCurrentRound(loopId) {
  const sel = document.querySelector("#compact-strip .trace-item.selected");
  if (!sel) { toast("未选中轮次", "warn"); return; }
  const idx = +sel.dataset.traceIdx;
  const loop = window.__loopCache.current || {};
  const t = (loop.traces || [])[idx];
  if (!t || !t.trace_id) return;
  deleteAttempt(loopId, t.trace_id, t.round);
}

// ============ 人工提示词面板（loop 详情页，运行中可增删） ============
// status 用来门控底部「继续跑（本轮生效）」按钮——只在可续跑状态显示。
async function renderHintsPanel(loopId, status) {
  const el = document.getElementById("hints-panel");
  if (!el) return;
  let hints = [];
  try {
    hints = ((await api(`/loops/${loopId}/hints`)).hints) || [];
  } catch (_) { return; }
  const renderHint = (h) => `<div class="hint-row">
    <span class="badge">${esc(h.agent)}</span>
    <span class="hint-text">${esc(h.text)}</span>
    <button class="btn btn-ghost btn-sm" onclick="removeLoopHint('${esc(loopId)}','${esc(h.id)}')">×</button>
  </div>`;
  // 按作用域分组：持久（sample，该考题所有 loop 共享）/ 临时（loop，仅当前 loop）。
  // 详情页本就 per-loop，分组让「临时提示词归属本 loop」一目了然。
  const persist = hints.filter((h) => h.scope === "sample");
  const temp = hints.filter((h) => h.scope !== "sample");
  const group = (title, subtitle, list) => `<div class="hint-group mt-2">
    <div class="muted" style="font-size:11px">${esc(title)}${subtitle ? ` · ${esc(subtitle)}` : ""}</div>
    <div class="hint-list">${list.length ? list.map(renderHint).join("") : '<div class="muted" style="font-size:12px">无</div>'}</div>
  </div>`;
  const editor = `<div class="nl-hint-editor mt-2">
    <div class="row" style="gap:8px;align-items:flex-start">
      <select id="hp-agent" class="hint-sel">
        <option value="critic">critic · 评判</option>
        <option value="generator">generator · 生图</option>
      </select>
      <select id="hp-scope" class="hint-sel">
        <option value="loop">临时（仅本 loop）</option>
        <option value="sample">持久（该考题所有 loop）</option>
      </select>
    </div>
    <textarea id="hp-text" class="mt-1" rows="2" placeholder="追加一条提示词…"></textarea>
    <button class="btn btn-primary btn-sm mt-1" onclick="addLoopHint('${esc(loopId)}')">＋ 添加</button>
  </div>`;
  // 「继续跑」与提示词注入绑定：加完提示词直接续跑，下一轮即生效。仅可续跑状态显示。
  const contBtn = RESUMABLE.includes(status)
    ? `<button class="btn btn-primary mt-2" onclick="resumeLoop('${esc(loopId)}','continue')">继续跑（本轮生效）</button>`
    : `<div class="muted" style="font-size:11px;margin-top:8px">提示词已记录，下一轮续跑时生效。</div>`;
  el.innerHTML = `<div class="card card-padded hints-card">
    <div class="hints-head">
      <h3 style="margin:0">人工提示词</h3>
      <span class="muted" style="font-size:11px">给 generator/critic 追加。临时=仅本 loop；持久=该考题所有 loop。</span>
    </div>
    ${group("持久", "该考题所有 loop", persist)}
    ${group("临时", `仅本 loop ${esc(loopId.replace(/.*-\d{4}-/, ""))}`, temp)}
    ${editor}
    ${contBtn}
  </div>`;
}

async function addLoopHint(loopId) {
  const agent = document.getElementById("hp-agent").value;
  const scope = document.getElementById("hp-scope").value;
  const text = document.getElementById("hp-text").value.trim();
  if (!text) return;
  try {
    await api(`/loops/${loopId}/hints`, { method: "POST", body: JSON.stringify({ agent, text, scope }) });
    toast("已添加提示词", "success");
    renderHintsPanel(loopId, window.__loopCache.current?.status);
  } catch (e) { handleError(e, "添加失败"); }
}

async function removeLoopHint(loopId, hintId) {
  try {
    await api(`/loops/${loopId}/hints/${hintId}`, { method: "DELETE" });
    toast("已删除", "success");
    renderHintsPanel(loopId, window.__loopCache.current?.status);
  } catch (e) { handleError(e, "删除失败"); }
}

// ============ Generator 本 loop 记忆面板（系统托管，按 loop 隔离；查看/编辑/清空） ============
// 记录 generator 每轮的动作（model/参数杠杆）+ Critic 结果，供下一轮选更合适的模型。
// 系统每轮自动追加；此处可人工编辑（下一轮注入即生效）或清空重来。
async function renderMemoryPanel(loopId) {
  const el = document.getElementById("memory-panel");
  if (!el) return;
  let content = "";
  try {
    content = ((await api(`/loops/${loopId}/memory`)).content) || "";
  } catch (_) { return; }
  el.innerHTML = `<details class="card card-padded hints-card">
    <summary class="bench-header">
      <svg class="chevron" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
      <h3 style="margin:0">Generator 本 loop 记忆</h3>
      <span class="muted" style="font-size:11px">系统每轮追加动作+结果；可编辑/清空</span>
    </summary>
    <textarea id="mem-text" class="mt-2" rows="10" style="width:100%;font-family:monospace;font-size:12px" placeholder="（尚无记忆——首轮 generator 出图后系统自动追加）">${esc(content)}</textarea>
    <div class="mt-1" style="display:flex;gap:8px">
      <button class="btn btn-primary btn-sm" onclick="saveMemoryPanel('${esc(loopId)}')">保存</button>
      <button class="btn btn-danger btn-sm" onclick="clearMemoryPanel('${esc(loopId)}')">清空</button>
    </div>
  </details>`;
}

async function saveMemoryPanel(loopId) {
  const ta = document.getElementById("mem-text");
  if (!ta) return;
  try {
    await api(`/loops/${loopId}/memory`, { method: "PUT", body: JSON.stringify({ content: ta.value }) });
    toast("记忆已保存，下一轮注入生效", "success");
  } catch (e) { handleError(e, "保存失败"); }
}

async function clearMemoryPanel(loopId) {
  if (!confirm("清空本 loop 的 generator 记忆？（系统托管文件重建为空，下一轮重新累积）")) return;
  try {
    await api(`/loops/${loopId}/memory`, { method: "DELETE" });
    toast("已清空", "success");
    renderMemoryPanel(loopId);
  } catch (e) { handleError(e, "清空失败"); }
}

// ============ 视图：人工排序 ============
let _rankState = null;
let _rankHistory = [];

async function viewScoring(benchId, sampleId) {
  setNavActive("scoring");
  setTitle("人工排序");
  const app = document.getElementById("app");
  app.innerHTML = skeleton(4);

  try {
    const overview = await api("/overview");
    let bench, sample;
    if (benchId && sampleId) {
      bench = overview.benches.find((b) => b.bench_id === benchId);
      sample = bench?.samples.find((s) => s.sample_id === sampleId);
    }

    if (!sample) {
      app.innerHTML = renderScoringLanding(overview);
      return;
    }

    const traces = [];
    for (const l of sample.loops) {
      const loop = await api(`/loops/${l.loop_id}`);
      for (const t of loop.traces) traces.push({ ...t, loop_id: l.loop_id, loop_short: l.loop_id.replace(/.*-\d{4}-/, "") });
    }
    traces.sort((a, b) => (b.verdict?.restoration ?? -1) - (a.verdict?.restoration ?? -1));

    const saved = await api(`/scoring/${benchId}/${sampleId}/ranks`);
    const savedMap = {};
    for (const r of saved.ranks) savedMap[r.trace_id] = r.rank;

    _rankState = { benchId, sampleId, traces, originalOrder: traces.slice() };
    _rankHistory = [];

    const target = sample.loops[0] ? (await api(`/loops/${sample.loops[0].loop_id}`)).target_image : null;
    const targetMd = sample.loops[0] ? (await api(`/loops/${sample.loops[0].loop_id}`)).target_md : null;

    app.innerHTML = `
      <div class="breadcrumb mb-3"><a href="#/">总览</a><span>/</span><span>${esc(benchId)}</span><span>/</span><span>${esc(sampleId)}</span></div>
      <div class="scoring-layout">
        <aside class="scoring-sidebar">
          <div class="card card-padded target-card">
            <h3>目标 sample</h3>
            ${target ? `<img src="${imgURL(target)}" data-lb="${imgURL(target)}" alt="target">` : '<div class="trace-img"></div>'}
            <div class="target-desc">${targetMd ? renderMarkdown(targetMd) : "暂无目标描述"}</div>
          </div>
        </aside>
        <section>
          <div class="scoring-toolbar">
            <h2 style="margin:0">拖拽排序（上=好）</h2>
            <label class="checkbox-row"><input type="checkbox" id="blind" onchange="renderRankList()"> 盲排</label>
            <button class="btn btn-ghost btn-sm" onclick="resetRanks()">重置为 AI 顺序</button>
            <button class="btn btn-ghost btn-sm" onclick="undoRank()" id="undo-btn" disabled>撤销</button>
            <button class="btn btn-primary" onclick="submitRanks()">提交排序</button>
          </div>
          <ul class="rank-list" id="rank-list"></ul>
          <div id="calib-result" class="calib-box" style="display:none"></div>
        </section>
      </div>`;
    renderRankList();
    loadCalibStatus(benchId, sampleId);
  } catch (e) {
    handleError(e, "加载排序页失败");
    app.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

function renderScoringLanding(overview) {
  let html = `<div class="empty"><div>选择要排序的 sample</div><div class="muted">从下方列表进入，或从总览点击 sample 的「人工排序」。</div></div>`;
  html += `<div class="row gap-3">`;
  for (const bench of overview.benches) {
    for (const sample of bench.samples) {
      if (!sample.loops.length) continue;
      const pending = sample.pending > 0 ? `<span class="badge badge-pending">待排序 ${sample.pending}</span>` : `<span class="badge badge-success">已排序</span>`;
      html += `<div class="card card-padded" style="flex:1 1 260px;min-width:220px;cursor:pointer" onclick="location.hash='#/scoring/${esc(bench.bench_id)}/${esc(sample.sample_id)}'">
        <div class="row" style="justify-content:space-between"><strong>${esc(sample.sample_id)}</strong>${pending}</div>
        <div class="muted" style="font-size:12px;margin-top:4px">${esc(bench.bench_id)}${sample.product ? " · " + esc(sample.product) : ""}</div>
        <div class="muted" style="font-size:12px">${sample.n_traces} trace · ${sample.loops.length} loop</div>
      </div>`;
    }
  }
  html += `</div>`;
  return html;
}

function renderRankList() {
  const ul = document.getElementById("rank-list");
  if (!ul || !_rankState) return;
  const blind = document.getElementById("blind")?.checked;
  ul.innerHTML = _rankState.traces
    .map((t, i) => {
      const img = t.output_image_refs[0];
      return `<li class="rank-item" draggable="true" data-idx="${i}" tabindex="0" aria-grabbed="false">
        <span class="grip">⠿</span>
        <span class="rank-num">#${i + 1}</span>
        ${img ? `<img class="rank-thumb" src="${imgURL(img, t.loop_id)}" data-lb="${imgURL(img, t.loop_id)}" alt="">` : '<div class="rank-thumb"></div>'}
        <span class="rank-meta">
          <span class="src">${esc(t.loop_short)} · 第${t.round}轮</span>
          ${!blind && t.verdict ? `<br>AI 还原度 ${fmt(t.verdict.restoration)}` : ""}
          ${t.human_rank != null ? `<br>历史人工排序: ${t.human_rank}` : ""}
        </span>
      </li>`;
    })
    .join("");
  bindDnD(ul);
}

function bindDnD(ul) {
  let dragIdx = null;
  ul.querySelectorAll(".rank-item").forEach((li) => {
    li.addEventListener("dragstart", (e) => {
      dragIdx = +li.dataset.idx;
      li.classList.add("dragging");
      li.setAttribute("aria-grabbed", "true");
      e.dataTransfer.effectAllowed = "move";
    });
    li.addEventListener("dragend", () => {
      li.classList.remove("dragging");
      li.setAttribute("aria-grabbed", "false");
      ul.querySelectorAll(".drag-over").forEach((x) => x.classList.remove("drag-over"));
    });
    li.addEventListener("dragover", (e) => { e.preventDefault(); li.classList.add("drag-over"); });
    li.addEventListener("dragleave", () => li.classList.remove("drag-over"));
    li.addEventListener("drop", (e) => {
      e.preventDefault();
      const dropIdx = +li.dataset.idx;
      li.classList.remove("drag-over");
      if (dragIdx === null || dragIdx === dropIdx) return;
      moveRank(dragIdx, dropIdx);
      dragIdx = null;
    });
    li.addEventListener("keydown", (e) => {
      const idx = +li.dataset.idx;
      if (e.key === "ArrowUp" && idx > 0) { e.preventDefault(); moveRank(idx, idx - 1); }
      if (e.key === "ArrowDown" && idx < _rankState.traces.length - 1) { e.preventDefault(); moveRank(idx, idx + 1); }
    });
  });
}

function moveRank(from, to) {
  _rankHistory.push(_rankState.traces.map((t) => t.trace_id));
  const arr = _rankState.traces;
  const [moved] = arr.splice(from, 1);
  arr.splice(to, 0, moved);
  renderRankList();
  document.getElementById("undo-btn").disabled = false;
}

function undoRank() {
  if (!_rankHistory.length) return;
  const prevIds = _rankHistory.pop();
  const map = new Map(_rankState.traces.map((t) => [t.trace_id, t]));
  _rankState.traces = prevIds.map((id) => map.get(id));
  renderRankList();
  document.getElementById("undo-btn").disabled = _rankHistory.length === 0;
}

function resetRanks() {
  _rankHistory.push(_rankState.traces.map((t) => t.trace_id));
  _rankState.traces = _rankState.originalOrder.slice();
  renderRankList();
  document.getElementById("undo-btn").disabled = false;
}

async function submitRanks() {
  const { benchId, sampleId, traces } = _rankState;
  const n = traces.length;
  const ranks = traces.map((t, i) => ({ trace_id: t.trace_id, rank: n - i }));
  try {
    await api(`/scoring/${benchId}/${sampleId}/ranks`, {
      method: "POST", body: JSON.stringify({ ranks, note: "web 排序" }),
    });
    toast("排序已提交，开始校准", "success");
    document.getElementById("calib-result").style.display = "block";
    document.getElementById("calib-result").innerHTML = '<div class="muted">校准中…</div>';
    pollCalib(benchId, sampleId);
  } catch (e) { handleError(e, "提交排序失败"); }
}

async function pollCalib(benchId, sampleId) {
  for (let i = 0; i < 20; i++) {
    await new Promise((r) => setTimeout(r, 600));
    const s = await api(`/scoring/${benchId}/${sampleId}/calibration`);
    if (s.state === "done" || s.state === "insufficient" || s.state === "error") { renderCalib(s); return; }
  }
  renderCalib(await api(`/scoring/${benchId}/${sampleId}/calibration`));
}

async function loadCalibStatus(benchId, sampleId) {
  const s = await api(`/scoring/${benchId}/${sampleId}/calibration`);
  if (s.state !== "idle") {
    document.getElementById("calib-result").style.display = "block";
    renderCalib(s);
  }
}

function renderCalib(s) {
  const box = document.getElementById("calib-result");
  if (!box) return;
  if (s.state === "insufficient") { box.innerHTML = `<div class="muted">${esc(s.message || "数据不足")}</div>`; return; }
  if (s.state === "error") { box.innerHTML = `<div class="alert"><div class="alert-title">校准出错</div></div>`; return; }
  if (s.state === "running") { box.innerHTML = '<div class="muted">校准中…</div>'; return; }
  if (!s.weights) { box.innerHTML = '<div class="muted">无结果</div>'; return; }
  const bars = Object.entries(s.weights)
    .sort((a, b) => b[1] - a[1])
    .map(([dim, w]) => {
      const prior = s.prior_weights?.[dim] ?? 0;
      const delta = w - prior;
      const sign = delta > 0.001 ? "▲" : delta < -0.001 ? "▼" : "·";
      return `<div class="weight-bar">
        <span class="dim-name">${esc(dim)}</span>
        <span class="bar"><span class="fill ${prior ? "" : "prior"}" style="width:${(w * 100).toFixed(0)}%"></span></span>
        <span class="v">${fmt(w)} <span class="muted">${sign}</span></span>
      </div>`;
    }).join("");
  box.innerHTML = `<h2>校准结果</h2>
    <div class="muted mb-3">排序吻合度 <strong>${(s.pairwise_accuracy * 100).toFixed(0)}%</strong> · ${s.n_pairs} 对 · ${s.n_traces} trace</div>
    <div>${bars}</div>
    <div class="muted mt-3" style="font-size:12px">下一轮 Critic 会应用这套权重。<button class="btn btn-ghost btn-sm" onclick="recalib()">重新校准</button></div>`;
}

async function recalib() {
  const { benchId, sampleId } = _rankState;
  await api(`/scoring/${benchId}/${sampleId}/calibrate`, { method: "POST" });
  document.getElementById("calib-result").innerHTML = '<div class="muted">重新校准中…</div>';
  pollCalib(benchId, sampleId);
}

// ============ 视图：Agent 设置 ============
let _cfgAgent = "generator";
let _cfgBench = null;        // 当前选中的 benchmark（generator 的技能随它切换）
let _cfgBenches = [];        // 可选 benchmark 列表（data/benchmarks/）

async function viewConfig() {
  setNavActive("config");
  setTitle("Agent 设置");
  const app = document.getElementById("app");
  app.innerHTML = skeleton(2);

  try {
    const data = await api("/config");
    _cfgBenches = data.benches || [];
    if (!_cfgBench && _cfgBenches.length) _cfgBench = _cfgBenches[0];
    app.innerHTML = `<div class="config-layout">
      <nav class="config-tabs" aria-label="Agent tabs">
        ${["generator", "critic", "distiller"].map((a) => `<button class="config-tab ${a === _cfgAgent ? "active" : ""}" onclick="switchCfg('${a}')">${a}</button>`).join("")}
      </nav>
      <div class="card card-padded config-form" id="cfg-form"></div>
    </div>`;
    await loadCfgForm();
  } catch (e) { handleError(e, "加载配置失败"); }
}

function switchCfg(a) { _cfgAgent = a; viewConfig(); }
function switchCfgBench(b) { _cfgBench = b; loadCfgForm(); }

async function loadCfgForm() {
  const form = document.getElementById("cfg-form");
  if (!form) return;
  form.innerHTML = skeleton(1);
  try {
    const q = _cfgBench ? `?bench=${encodeURIComponent(_cfgBench)}` : "";
    const a = await api(`/config/${_cfgAgent}${q}`);
    renderCfgForm(a);
  } catch (e) { handleError(e, "加载配置失败"); }
}

function renderCfgForm(a) {
  // benchmark 下拉：generator 的技能 = per-bench 蒸馏 skill_package，随它切换。
  const benchOpts = (_cfgBenches || []).map(
    (b) => `<option value="${esc(b)}" ${b === _cfgBench ? "selected" : ""}>${esc(b)}</option>`,
  ).join("");
  const benchSel = (a.agent === "generator" && benchOpts)
    ? `<label>benchmark（generator 技能随它切换；未蒸馏则裸跑）</label>
       <select id="cfg-bench" onchange="switchCfgBench(this.value)">${benchOpts}</select>`
    : "";
  // —— 只读派生区：工具 / loop 内职责 / 技能 / 其他参数（后端 get_agent_config 返回，POST 不写）——
  const toolsHtml = (a.tools && a.tools.length)
    ? `<div class="chip-row">${a.tools.map((t) => `<span class="cfg-chip">${esc(t)}</span>`).join("")}</div>`
    : `<span class="muted">无（直接调用 LLM，不走工具）</span>`;
  // loop 节点内兼任的非 LLM-tool 职责（如 critic 兼任的经验总结 summarize）。与 tools 分列，避免误当成 LLM 工具。
  const dutiesHtml = (a.duties && a.duties.length)
    ? `<ul class="cfg-skills">${a.duties.map(
        (d) => `<li><strong>${esc(d.name)}</strong>${d.desc ? ` — <span class="muted">${esc(d.desc)}</span>` : ""}</li>`,
      ).join("")}</ul>`
    : `<span class="muted">无</span>`;
  let skillsHtml;
  if (a.agent === "critic") {
    skillsHtml = `<span class="muted">critic 不使用技能（靠 user message 注入的 rubric/checklist 判分）</span>`;
  } else if (a.skills && a.skills.length) {
    skillsHtml = `<ul class="cfg-skills">${a.skills.map(
      (s) => `<li><strong>${esc(s.name)}</strong>${s.summary ? ` — <span class="muted">${esc(s.summary)}</span>` : ""}<br><span class="muted" style="font-size:11px">来源: ${esc(s.source || "")}</span></li>`,
    ).join("")}</ul>`;
  } else if (a.agent === "generator") {
    skillsHtml = `<span class="muted">该 benchmark 尚未蒸馏，generator 裸跑（无技能）</span>`;
  } else {
    skillsHtml = `<span class="muted">无</span>`;
  }
  const params = a.params || {};
  const paramKeys = Object.keys(params);
  const paramsHtml = paramKeys.length
    ? `<div class="trace-params">${paramKeys
        .map((k) => `<div class="param"><span class="pk">${esc(k)}</span><span class="pv">${esc(String(params[k]))}</span></div>`)
        .join("")}</div>`
    : `<span class="muted">无</span>`;

  // distiller 的「系统提示词」= skill-author authoring 阶段 prompt（编写技能包）；
  // model 走 summarizer_model（.env 默认），保存后下次蒸馏生效。三个 agent 均可编辑。
  const promptHint = a.agent === "distiller"
    ? ' <span class="muted" style="font-size:11px">（技能包编写 skill-author 的 prompt；保存后下次蒸馏生效）</span>'
    : "";

  document.getElementById("cfg-form").innerHTML = `
    <h2>${esc(_cfgAgent)} 配置</h2>
    ${benchSel}
    <label>模型 id${a.agent === "distiller" ? ' <span class="muted" style="font-size:11px">（蒸馏器 LLM，保存后下次蒸馏生效）</span>' : ""}</label>
    <input type="text" id="cfg-model" value="${esc(a.model)}">
    <label>系统提示词${promptHint}</label>
    <textarea id="cfg-prompt">${esc(a.system_prompt)}</textarea>
    <div class="row mt-3">
      <button class="btn btn-primary" onclick="saveCfg()">保存</button>
      <button class="btn btn-ghost" onclick="resetCfg()">恢复默认</button>
      <span id="cfg-msg" class="muted"></span>
    </div>

    <section class="cfg-readonly">
      <h2>可用工具</h2>
      ${toolsHtml}
      <h2>loop 内职责 <span class="muted" style="font-size:11px">（节点方法，非 LLM 工具）</span></h2>
      ${dutiesHtml}
      <h2>技能 (skills)</h2>
      ${skillsHtml}
      <h2>其他参数</h2>
      ${paramsHtml}
    </section>`;
}

async function saveCfg() {
  const model = document.getElementById("cfg-model").value;
  const prompt = document.getElementById("cfg-prompt").value;
  try {
    const q = _cfgBench ? `?bench=${encodeURIComponent(_cfgBench)}` : "";
    await api(`/config/${_cfgAgent}${q}`, { method: "POST", body: JSON.stringify({ system_prompt: prompt, model }) });
    toast("配置已保存", "success");
    document.getElementById("cfg-msg").textContent = "已保存 ✓";
    setTimeout(() => { const m = document.getElementById("cfg-msg"); if (m) m.textContent = ""; }, 2000);
  } catch (e) { handleError(e, "保存失败"); }
}

async function resetCfg() {
  if (!confirm("确定恢复为代码默认配置？")) return;
  try {
    const q = _cfgBench ? `?bench=${encodeURIComponent(_cfgBench)}` : "";
    await api(`/config/${_cfgAgent}${q}/reset`, { method: "POST" });
    toast("已恢复默认", "success");
    loadCfgForm();
  } catch (e) { handleError(e, "恢复失败"); }
}

// ============ 启动新 loop ============
let _newLoopData = null;
let _newLoopModels = null;
let _newLoopHints = [];  // 启动表单暂存的人工提示词（提交时随 body 发送）

async function startNewLoop(preBench, preSample) {
  _newLoopHints = [];
  if (!_newLoopData) _newLoopData = await api("/overview");
  if (!_newLoopModels) {
    try { _newLoopModels = await api("/models"); } catch (_) { _newLoopModels = { image_models: [], agent_models: [] }; }
  }
  if (!_newLoopData.benches.length) { toast("还没有 benchmark", "warn"); return; }

  const firstBench = preBench ? _newLoopData.benches.find((b) => b.bench_id === preBench) : _newLoopData.benches[0];
  const benchId = firstBench ? firstBench.bench_id : _newLoopData.benches[0].bench_id;
  const sampleId = preSample || (firstBench?.samples[0]?.sample_id || "");

  openModal(renderNewLoopForm(benchId, sampleId));
  onNewLoopBenchChange(sampleId);
  updateNewLoopPreview();
}

function renderNewLoopForm(benchId, sampleId) {
  const benches = _newLoopData.benches;
  const models = _newLoopModels || { image_models: [], agent_models: [] };
  const imageOpts = [`<option value="">默认（settings）</option>`, ...models.image_models.map((m) => `<option value="${esc(m.model_id)}">${esc(m.label)} · ${esc(m.model_id)}</option>`)].join("");
  const agentInfo = (models.agent_models || []).map((m) => `${esc(m.label)}=<code>${esc(m.model_id)}</code>`).join(" · ") || "（未配置）";
  const roundsOpts = [1, 2, 3, 4, 5, 6, 7, 8].map((n) => `<option value="${n}"${n === 4 ? " selected" : ""}>${n} 轮</option>`).join("");

  return `<h1>启动新 loop</h1>
    <div class="muted">对一个 sample 用固定模型跑一条自迭代闭环。</div>
    <div class="newloop-form mt-3">
      <label>benchmark</label>
      <select id="nl-bench" onchange="onNewLoopBenchChange()">${benches.map((b) => `<option value="${esc(b.bench_id)}"${b.bench_id === benchId ? " selected" : ""}>${esc(b.bench_id)}</option>`).join("")}</select>
      <label>sample</label>
      <select id="nl-sample" onchange="updateNewLoopPreview()"></select>
      <label>生图模型</label>
      <select id="nl-model" onchange="updateNewLoopPreview()">${imageOpts}</select>
      <label>轮数</label>
      <select id="nl-rounds" onchange="updateNewLoopPreview()">${roundsOpts}</select>
      <div class="muted" style="font-size:11px">选 1 = 首轮跑到等审批就停；选 &gt;1 = 后台自动连跑。</div>
      <label>备注</label>
      <input type="text" id="nl-note" placeholder="可选">
      <details class="nl-hints" style="margin-top:8px">
        <summary style="cursor:pointer">人工提示词（可选）<span class="muted" style="font-size:11px">给 generator/critic 追加要求；运行中可继续改</span></summary>
        <div class="nl-hint-editor mt-2">
          <div class="row" style="gap:8px;align-items:flex-start">
            <select id="nl-hint-agent" class="hint-sel">
              <option value="critic">critic · 评判</option>
              <option value="generator">generator · 生图</option>
            </select>
            <select id="nl-hint-scope" class="hint-sel">
              <option value="loop">临时（仅本 loop）</option>
              <option value="sample">持久（该考题所有 loop）</option>
            </select>
          </div>
          <textarea id="nl-hint-text" class="mt-1" rows="2" placeholder="如：画面中手的元素应是连续扁平的线条画，禁止出现断开或有结构的具象造型"></textarea>
          <button class="btn btn-ghost btn-sm mt-1" onclick="addNewLoopHint()">＋ 添加</button>
        </div>
        <div id="nl-hint-list" class="mt-2"></div>
      </details>
      <div class="muted" style="font-size:11px">Agent LLM（全局，不随 loop 改）：${agentInfo}</div>
      <div class="newloop-preview" id="nl-preview"></div>
      <div class="row mt-3">
        <button class="btn btn-primary" id="nl-submit" onclick="submitNewLoop()">启动 loop</button>
        <button class="btn btn-ghost" data-close>取消</button>
      </div>
    </div>`;
}

function onNewLoopBenchChange(preselectSample) {
  const benchId = document.getElementById("nl-bench").value;
  const bench = _newLoopData.benches.find((b) => b.bench_id === benchId);
  const sel = document.getElementById("nl-sample");
  sel.innerHTML = (bench?.samples || [])
    .map((s) => `<option value="${esc(s.sample_id)}"${s.sample_id === preselectSample ? " selected" : ""}>${esc(s.sample_id)} ${s.product ? "· " + esc(s.product) : ""}</option>`)
    .join("");
  updateNewLoopPreview();
}

function updateNewLoopPreview() {
  const benchId = document.getElementById("nl-bench").value;
  const sampleId = document.getElementById("nl-sample").value;
  const model = document.getElementById("nl-model").value.trim() || "（settings 默认）";
  const rounds = parseInt(document.getElementById("nl-rounds").value, 10) || 1;
  const bench = _newLoopData.benches.find((b) => b.bench_id === benchId);
  const sample = bench?.samples.find((s) => s.sample_id === sampleId);
  const product = sample?.product ? `（${esc(sample.product)}）` : "";

  // modal 只负责 web loop（一题一条：固定 loop_id=<bench>-<sample>）。存在则续跑，不存在则新建。
  // 续跑特定 loop（含外部 -tag loop）请用卡片/详情页的「继续跑」一键 resume，不走本 modal。
  const webLoopId = `${benchId}-${sampleId}`;
  const exists = (sample?.loops || []).some((l) => l.loop_id === webLoopId);
  const action = exists
    ? `续跑 web loop <code>${esc(webLoopId)}</code>（已存在）`
    : `新建 web loop <code>${esc(webLoopId)}</code>`;
  const roundsDesc = rounds > 1
    ? `将自动连跑 <strong>${rounds}</strong> 轮，跑满后停在审批节点。`
    : `首轮回跑到人工审批节点停下。`;

  document.getElementById("nl-preview").innerHTML =
    `将对 <code>${esc(sampleId)}</code>${product} 用 <code>${esc(model)}</code> ${action}<br>` + roundsDesc;
  document.getElementById("nl-submit").textContent = "启动 loop";
}

async function submitNewLoop() {
  const benchId = document.getElementById("nl-bench").value;
  const sampleId = document.getElementById("nl-sample").value;
  const model = document.getElementById("nl-model").value.trim() || undefined;
  const rounds = parseInt(document.getElementById("nl-rounds").value, 10) || undefined;
  const note = document.getElementById("nl-note").value.trim() || undefined;
  const hints = _newLoopHints.map(({ agent, scope, text }) => ({ agent, scope, text }));
  const btn = document.getElementById("nl-submit");
  btn.disabled = true; btn.textContent = "启动中…";
  try {
    const r = await api(`/loops`, {
      method: "POST",
      body: JSON.stringify({ bench_id: benchId, sample_id: sampleId, model, note, rounds, hints: hints.length ? hints : undefined }),
    });
    _newLoopHints = [];
    closeModal();
    toast("loop 已启动", "success");
    location.hash = `#/loop/${r.loop_id}`;
  } catch (e) {
    btn.disabled = false; btn.textContent = "启动 loop";
    handleError(e, "启动失败");
  }
}

function addNewLoopHint() {
  const agent = document.getElementById("nl-hint-agent").value;
  const scope = document.getElementById("nl-hint-scope").value;
  const text = document.getElementById("nl-hint-text").value.trim();
  if (!text) return;
  _newLoopHints.push({ agent, scope, text });
  document.getElementById("nl-hint-text").value = "";
  renderNewLoopHintList();
}

function removeNewLoopHint(idx) {
  _newLoopHints.splice(idx, 1);
  renderNewLoopHintList();
}

function renderNewLoopHintList() {
  const el = document.getElementById("nl-hint-list");
  if (!el) return;
  el.innerHTML = _newLoopHints.map((h, i) => `<div class="hint-row">
    <span class="badge ${h.scope === "sample" ? "badge-primary" : "badge-ghost"}">${h.scope === "sample" ? "持久" : "临时"}</span>
    <span class="badge">${esc(h.agent)}</span>
    <span class="hint-text">${esc(h.text)}</span>
    <button class="btn btn-ghost btn-sm" onclick="removeNewLoopHint(${i})">×</button>
  </div>`).join("");
}

// ============ 视图：通用经验（跨 loop 蒸馏） ============
async function viewExperience(benchId) {
  setNavActive("");
  setTitle("通用经验", `<a class="btn btn-ghost btn-sm" href="#/">← 总览</a>`);
  const app = document.getElementById("app");
  app.innerHTML = skeleton(2);
  await renderExperience(benchId);
}

async function renderExperience(benchId) {
  const app = document.getElementById("app");
  try {
    const exp = await api(`/experience/${encodeURIComponent(benchId)}`);
    window.__expBench = benchId;
    window.__expCache = exp;  // 供 editLesson 取当前值填表
    const count = (exp.lessons || []).length;
    const lessonsHtml = count
      ? exp.lessons.map((l) => renderLessonCard(l, benchId)).join("")
      : `<div class="empty"><div>尚无通用经验</div><div class="muted">点「重新蒸馏」跨 loop 归纳生成（需该 benchmark 已有 run）。</div></div>`;
    const metaBits = [];
    if (exp.scene || exp.bench_description) metaBits.push(esc(exp.scene || exp.bench_description));
    if (exp.dimensions && exp.dimensions.length) metaBits.push("维度：" + exp.dimensions.map(esc).join(" · "));
    const srcLine = exp.updated_at
      ? `<div class="muted exp-meta">更新 ${esc(exp.updated_at)}${(exp.source_runs || []).length ? " · 来源 " + exp.source_runs.map(esc).join(", ") : ""}</div>`
      : "";

    app.innerHTML = `
      <div class="breadcrumb mb-3"><a href="#/">总览</a><span>/</span><span>${esc(benchId)}</span><span>/</span><span>通用经验</span></div>
      <div class="exp-toolbar">
        <h2 style="margin:0">通用经验 · ${esc(benchId)}</h2>
        <div class="exp-actions">
          <button class="btn btn-secondary btn-sm" title="规范技能包 zip（SKILL.md + references + assets）" onclick="exportSkillPackage('${esc(benchId)}')">导出技能包(.zip)</button>
          <button class="btn btn-ghost btn-sm" onclick="exportSkillMd('${esc(benchId)}')">SKILL.md</button>
          <button class="btn btn-ghost btn-sm" onclick="copySkillMd('${esc(benchId)}')">复制</button>
          <button class="btn btn-primary btn-sm" id="distill-btn" onclick="triggerDistill('${esc(benchId)}')">重新蒸馏</button>
        </div>
      </div>
      <div id="distill-status" class="distill-box" style="display:none"></div>
      ${metaBits.length ? `<div class="muted mb-2">${metaBits.join("　·　")}</div>` : ""}
      ${exp.summary ? `<div class="card card-padded mb-3"><strong>总览</strong><div class="mt-1">${esc(exp.summary)}</div>${srcLine}</div>` : srcLine}
      <div class="exp-lessons">${lessonsHtml}</div>`;
  } catch (e) {
    handleError(e, "加载经验失败");
    app.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

const LESSON_STATUS_LABEL = { active: "active", refuted: "已证伪", superseded: "已修订", archived: "已归档" };
const WHEN_LABEL = { construction: "首轮", fix: "修复", always: "通用" };

function renderLessonCard(l, benchId) {
  const conf = l.confidence != null ? Math.round((l.confidence || 0) * 100) : null;
  const confBar = conf != null
    ? `<span class="conf"><span class="conf-bar"><span class="conf-fill" style="width:${conf}%"></span></span><span class="conf-pct">${conf}%</span></span>`
    : "";
  const dos = (l.dos || []).map((d) => `<li class="dos">✓ ${esc(d)}</li>`).join("");
  const donts = (l.donts || []).map((d) => `<li class="donts">✗ ${esc(d)}</li>`).join("");
  const evd = (l.evidence || []).length ? `<div class="muted lesson-evidence">证据：${l.evidence.map(esc).join("；")}</div>` : "";
  const isActive = l.status === "active";
  const statusBadge = isActive ? "" : `<span class="badge lesson-status">${LESSON_STATUS_LABEL[l.status] || l.status}</span>`;
  const retireReason = (!isActive && l.retire_reason) ? `<div class="muted lesson-evidence">理由：${esc(l.retire_reason)}</div>` : "";
  const catTag = l.category ? `<span class="badge lesson-cat">${esc(l.category)}</span>` : "";
  const apTag = l.applies_when ? `<span class="lesson-when">${WHEN_LABEL[l.applies_when] || l.applies_when}</span>` : "";
  const bid = `'${esc(benchId)}'`, lid = `'${esc(l.id)}'`;
  const actions = `<div class="lesson-actions">
      <button class="btn btn-ghost btn-sm" onclick="editLesson(${bid},${lid})">✏️ 编辑</button>
      ${isActive ? `<button class="btn btn-ghost btn-sm" onclick="refuteLesson(${bid},${lid})">🚫 标无效</button>
      <button class="btn btn-ghost btn-sm" onclick="archiveLesson(${bid},${lid})">🗑 归档</button>` : ""}
    </div>`;
  return `<div class="card card-padded lesson-card ${isActive ? "" : "lesson-retired"}">
    <div class="lesson-head">
      <span class="badge">${esc(l.dim)}</span>
      ${catTag}
      <span class="lesson-insight">${esc(l.insight)}</span>
      ${apTag}
      ${confBar}
      ${statusBadge}
    </div>
    ${(dos || donts) ? `<ul class="lesson-list">${dos}${donts}</ul>` : ""}
    ${evd}${retireReason}
    ${actions}
  </div>`;
}

function editLesson(benchId, id) {
  const l = (window.__expCache?.lessons || []).find((x) => x.id === id);
  if (!l) { toast("找不到该经验", "error"); return; }
  openModal(`
    <div class="modal-card">
      <h3>编辑经验（保存后重新启用为 active）</h3>
      <div class="form-row"><label>insight（一句话规律）</label><textarea id="le-insight" rows="2">${esc(l.insight)}</textarea></div>
      <div class="form-row"><label>dos（每行一条）</label><textarea id="le-dos" rows="3">${esc((l.dos || []).join("\n"))}</textarea></div>
      <div class="form-row"><label>donts（每行一条）</label><textarea id="le-donts" rows="3">${esc((l.donts || []).join("\n"))}</textarea></div>
      <div class="form-row"><label>category（粗类）</label><input id="le-category" value="${esc(l.category || "")}"></div>
      <div class="form-row"><label>applies_when</label>
        <select id="le-when">${["always", "construction", "fix"].map((o) => `<option value="${o}" ${l.applies_when === o ? "selected" : ""}>${WHEN_LABEL[o]}</option>`).join("")}</select>
      </div>
      <div class="form-row"><label>confidence (0-1)</label><input id="le-conf" type="number" step="0.05" min="0" max="1" value="${l.confidence ?? 0.5}"></div>
      <div class="form-actions">
        <button class="btn btn-ghost" onclick="closeModal()">取消</button>
        <button class="btn btn-primary" onclick="submitLessonEdit('${esc(benchId)}','${esc(id)}')">保存</button>
      </div>
    </div>`);
}

async function submitLessonEdit(benchId, id) {
  const split = (v) => v.split("\n").map((s) => s.trim()).filter(Boolean);
  const body = {
    insight: document.getElementById("le-insight").value.trim(),
    dos: split(document.getElementById("le-dos").value),
    donts: split(document.getElementById("le-donts").value),
    category: document.getElementById("le-category").value.trim(),
    applies_when: document.getElementById("le-when").value,
    confidence: parseFloat(document.getElementById("le-conf").value),
  };
  try {
    await api(`/experience/${encodeURIComponent(benchId)}/lessons/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(body) });
    closeModal(); toast("已保存", "success"); renderExperience(benchId);
  } catch (e) { handleError(e, "保存失败"); }
}

async function refuteLesson(benchId, id) {
  const reason = window.prompt("标无效理由（该经验将不再被消费，翻新时勿复生）：");
  if (reason === null) return;
  try {
    await api(`/experience/${encodeURIComponent(benchId)}/lessons/${encodeURIComponent(id)}/refute`, { method: "POST", body: JSON.stringify({ reason }) });
    toast("已标无效", "success"); renderExperience(benchId);
  } catch (e) { handleError(e, "操作失败"); }
}

async function archiveLesson(benchId, id) {
  if (!confirm("归档这条经验？将不再被消费。")) return;
  try {
    await api(`/experience/${encodeURIComponent(benchId)}/lessons/${encodeURIComponent(id)}`, { method: "DELETE" });
    toast("已归档", "success"); renderExperience(benchId);
  } catch (e) { handleError(e, "操作失败"); }
}

async function triggerDistill(benchId) {
  const btn = document.getElementById("distill-btn");
  if (btn) { btn.disabled = true; btn.textContent = "蒸馏中…"; }
  try {
    await api(`/experience/${encodeURIComponent(benchId)}/distill`, { method: "POST" });
    pollDistillStatus(benchId);
  } catch (e) {
    if (btn) { btn.disabled = false; btn.textContent = "重新蒸馏"; }
    handleError(e, "触发蒸馏失败");
  }
}

function pollDistillStatus(benchId) {
  if (_distillPollTimer) clearInterval(_distillPollTimer);
  const terminal = (s) => s === "done" || s === "error" || s === "no_runs";
  const tick = async () => {
    try {
      const st = await api(`/experience/${encodeURIComponent(benchId)}/distill/status`);
      const box = document.getElementById("distill-status");
      if (box) {
        box.style.display = "block";
        box.className = "distill-box " + ((st.state === "error" || st.state === "no_runs") ? "distill-err" : "distill-run");
        box.innerHTML = `<strong>${DISTILL_LABEL[st.state] || st.state}</strong>${st.message ? " · " + esc(st.message) : ""}${st.error ? " · " + esc(st.error) : ""}`;
      }
      if (terminal(st.state)) {
        if (_distillPollTimer) { clearInterval(_distillPollTimer); _distillPollTimer = null; }
        const btn = document.getElementById("distill-btn");
        if (btn) { btn.disabled = false; btn.textContent = "重新蒸馏"; }
        if (st.state === "done") { toast("蒸馏完成", "success"); renderExperience(benchId); }
      }
    } catch (e) { /* 瞬时错误忽略，继续轮询 */ }
  };
  tick();
  _distillPollTimer = setInterval(tick, 2000);
}

async function exportSkillMd(benchId) {
  window.open(`/api/experience/${encodeURIComponent(benchId)}/skill.md`, "_blank");
}

function exportSkillPackage(benchId) {
  // 下载规范技能包 zip（目录 zip：SKILL.md + references/ + assets/），可装到别的 agent 工具
  window.location.href = `/api/experience/${encodeURIComponent(benchId)}/skill.zip`;
}

async function copySkillMd(benchId) {
  try {
    const r = await fetch(`/api/experience/${encodeURIComponent(benchId)}/skill.md`);
    if (!r.ok) throw new Error(await r.text().catch(() => r.statusText));
    const txt = await r.text();
    await navigator.clipboard.writeText(txt);
    toast("SKILL.md 已复制", "success");
  } catch (e) { handleError(e, "复制失败（可能尚无 SKILL.md，先蒸馏）"); }
}

// ============ 视图：Benchmark 管理 ============
async function viewBenchmarks() {
  setNavActive("benchmarks");
  setTitle("Benchmark 管理", `<button class="btn btn-primary" onclick="openCreateBenchForm()">＋ 新增 benchmark</button>`);
  const app = document.getElementById("app");
  app.innerHTML = skeleton(3);
  try {
    const data = await api("/benchmarks");
    const benches = data.benches || [];
    if (!benches.length) {
      app.innerHTML = `<div class="empty"><div>还没有 benchmark</div><div class="muted">新建一个 benchmark，或在数据目录准备后刷新。</div><button class="btn btn-primary mt-2" onclick="openCreateBenchForm()">＋ 新增 benchmark</button></div>`;
      return;
    }
    app.innerHTML = `<div class="row gap-3">${benches.map(renderBenchCard).join("")}</div>`;
  } catch (e) {
    handleError(e, "加载 benchmark 列表失败");
    app.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

function renderBenchCard(b) {
  const skill = b.has_distill_skill ? `<span class="badge badge-success">已蒸馏 skill</span>` : "";
  return `<div class="card card-padded" style="flex:1 1 320px;min-width:300px">
    <div class="row" style="justify-content:space-between;align-items:flex-start">
      <h3 style="margin:0">${esc(b.bench_id)}</h3>${skill}
    </div>
    <div class="muted" style="font-size:12px;margin-top:4px">${esc(b.description || b.scene || "")}</div>
    <div class="mt-2">
      <span class="badge">${b.n_samples} sample</span>
      <span class="badge">${b.n_dims} 维度</span>
      <span class="badge">${b.n_loops} loop</span>
      ${b.task_type ? `<span class="badge badge-ghost">${esc(b.task_type)}</span>` : ""}
    </div>
    <div class="mt-2">
      <a class="btn btn-secondary btn-sm" href="#/benchmarks/${encodeURIComponent(b.bench_id)}">查看结构 / 消费者</a>
      <a class="btn btn-ghost btn-sm" href="#/">去总览</a>
    </div>
  </div>`;
}

async function viewBenchmarkDetail(benchId) {
  setNavActive("benchmarks");
  setTitle("Benchmark 详情", `<a class="btn btn-ghost btn-sm" href="#/benchmarks">← 返回</a>`);
  const app = document.getElementById("app");
  app.innerHTML = skeleton(3);
  try {
    const b = await api(`/benchmarks/${encodeURIComponent(benchId)}`);
    window.__benchCache = b;
    app.innerHTML = renderBenchmarkDetail(b);
  } catch (e) {
    handleError(e, "加载 benchmark 详情失败");
    app.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

function renderBenchmarkDetail(b) {
  const dims = b.dimensions || [];
  const samples = b.samples || [];
  const task = b.task || {};

  const dimRows = dims.map((d) => `<tr>
    <td><strong>${esc(d.dim)}</strong>${d.ref_needed ? ' <span class="ref-dot" title="需参考图">●</span>' : ""}</td>
    <td>${esc(d.desc || "")}</td>
    <td>${fmt(d.weight_init, 2)}</td>
    <td>${esc(d.scoring_type)}</td>
    <td>${d.scoring_type === "binary" ? d.n_check_items : "—"}</td>
  </tr>`).join("");

  const sampleRows = samples.map((s) => `<tr>
    <td><strong>${esc(s.sample_id)}</strong></td>
    <td>${esc(s.product || "—")}</td>
    <td>${esc(s.category || "—")}</td>
    <td>${s.has_target ? "✓" : `<span class="muted">缺</span>`}</td>
    <td>${s.n_loops}</td>
    <td><button class="btn btn-danger btn-sm" onclick="deleteSample('${esc(b.bench_id)}','${esc(s.sample_id)}')">删除</button></td>
  </tr>`).join("");

  const tree = (b.file_tree || []).map((f) =>
    `<div class="tree-row ${f.type}">${f.type === "dir" ? "📁" : "📄"} ${esc(f.path)}</div>`,
  ).join("");

  const consumers = (b.consumers || []).map((c) => `<div class="consume-item">
    <div class="consume-name">${esc(c.name)} <span class="badge badge-ghost">${esc(c.signal || "")}</span></div>
    <div class="muted" style="font-size:12px">${esc(c.desc || "")}</div>
  </div>`).join("");

  return `
    <div class="breadcrumb mb-3"><a href="#/benchmarks">Benchmark 管理</a><span>/</span><span>${esc(b.bench_id)}</span></div>
    <div class="page-header"><div class="title-block">
      <h1>${esc(b.bench_id)}</h1>
      <div class="muted" style="font-size:12px">${esc(b.scene || "")}${b.description ? " · " + esc(b.description) : ""}</div>
    </div></div>

    <div class="card card-padded mt-2">
      <h2>基本信息</h2>
      <div class="trace-params">
        <div class="param"><span class="pk">场景</span><span class="pv">${esc(b.scene || "—")}</span></div>
        <div class="param"><span class="pk">评分方法</span><span class="pv">${esc(b.scoring_method || "—")}</span></div>
        <div class="param"><span class="pk">任务类型</span><span class="pv">${esc(task.type || "—")}</span></div>
        <div class="param"><span class="pk">视图</span><span class="pv">${esc((task.views || []).join(", ") || "—")}</span></div>
        <div class="param"><span class="pk">对比型维度</span><span class="pv">${esc((b.comparative_dims || []).join(", ") || "—")}</span></div>
      </div>
    </div>

    <div class="card card-padded mt-3">
      <h2>评分维度（${dims.length}）</h2>
      <table class="bench-table"><thead><tr><th>维度</th><th>说明</th><th>权重</th><th>类型</th><th>check 项</th></tr></thead>
        <tbody>${dimRows || `<tr><td colspan="5" class="muted">无</td></tr>`}</tbody></table>
      <div class="muted" style="font-size:11px;margin-top:6px">● = 需 target 参考图（对比型维度）</div>
    </div>

    <div class="card card-padded mt-3">
      <h2>题目 samples（${samples.length}）</h2>
      <table class="bench-table"><thead><tr><th>sample</th><th>产品</th><th>品类</th><th>target</th><th>loop</th><th></th></tr></thead>
        <tbody>${sampleRows || `<tr><td colspan="6" class="muted">无</td></tr>`}</tbody></table>
    </div>

    <div class="card card-padded mt-3">
      <h2>内容被谁消费</h2>
      <div class="consume-list">${consumers}</div>
    </div>

    <div class="card card-padded mt-3">
      <h2>目录结构</h2>
      <div class="file-tree">${tree || '<div class="muted">空</div>'}</div>
    </div>`;
}

async function deleteSample(benchId, sampleId) {
  if (!confirm(`确定删除 sample ${sampleId}？\n该题目的所有 loop、loop 内经验、人工提示词与人工排序都会删除；跨 loop 蒸馏 skill 包保留。`)) return;
  try {
    await api(`/benchmarks/${encodeURIComponent(benchId)}/samples/${encodeURIComponent(sampleId)}`, { method: "DELETE" });
    toast(`已删除 sample ${sampleId}`, "success");
    viewBenchmarkDetail(benchId);
  } catch (e) { handleError(e, "删除失败"); }
}

// ---- 新增 benchmark 表单（modal）----
const _FURNITURE_DIMS = [
  { dim: "consistency", desc: "三视图跨张一致性：同产品/同色/几何比例一致", weight_init: 0.25, ref_needed: true, scoring_type: "binary",
    check_items: "C1 三视图是同一产品\nC2 三视图颜色一致\nC3 侧视与正视几何尺寸一致(不预设具体形态)\nC4 各视图高度比例一致\nC5 每个视图完整呈现整个产品", rubric_ref: "" },
  { dim: "product_structure", desc: "产品结构：部件数/位置/形态正确", weight_init: 0.22, ref_needed: true, scoring_type: "binary",
    check_items: "S1 整体结构忠实还原参考图\nS2 无部件穿模/重叠\nS3 无部件缺失/多余\nS4 整体造型轮廓忠实还原参考图", rubric_ref: "" },
  { dim: "material_texture", desc: "材质还原度(对照参考)，连续分", weight_init: 0.18, ref_needed: true, scoring_type: "continuous", check_items: "", rubric_ref: "rubric.md#材质纹理" },
  { dim: "color_accuracy", desc: "颜色准确度(对照参考)，连续分", weight_init: 0.13, ref_needed: true, scoring_type: "continuous", check_items: "", rubric_ref: "rubric.md#颜色一致" },
  { dim: "artifact_defect", desc: "无瑕疵：无变形/失真/模糊/伪影/悬浮", weight_init: 0.12, ref_needed: false, scoring_type: "binary",
    check_items: "A1 直线无弯曲\nA2 对称结构对称\nA3 无模糊/拼接痕\nA4 接地有阴影(不悬浮)", rubric_ref: "" },
  { dim: "commercial_focus", desc: "商业可用：主体突出/白底干净/构图合规", weight_init: 0.10, ref_needed: false, scoring_type: "binary",
    check_items: "B1 主体居中突出\nB2 背景纯白干净\nB3 留白构图符合平台规范", rubric_ref: "" },
];
let _benchDims = [];
let _benchSamples = [];

function openCreateBenchForm() {
  _benchDims = JSON.parse(JSON.stringify(_FURNITURE_DIMS));
  _benchSamples = [{ sample_id: "s001", product: "", category: "" }];
  openModal(renderCreateBenchForm());
  renderDimEditor();
  renderSampleEditor();
}

function renderCreateBenchForm() {
  return `<h1>新增 benchmark</h1>
    <div class="muted">填写元数据、评分维度、题目，并上传每个题目的 target 参考图。生成 manifest.json + rubric.md + content_spec 脚手架。</div>
    <div class="newloop-form mt-3">
      <label>bench_id <span class="muted" style="font-size:11px">（目录名，仅字母数字 _ . -）</span></label>
      <input type="text" id="cb-bench" placeholder="如 furniture_product_whitebg">
      <label>场景描述</label>
      <input type="text" id="cb-scene" placeholder="如 白底产品三视图（产品还原度导向）">
      <label>说明</label>
      <textarea id="cb-desc" rows="2" placeholder="本 benchmark 聚焦的还原度难点（可选）"></textarea>
      <div class="row" style="gap:8px">
        <div style="flex:1"><label>任务预设</label>
          <select id="cb-task" onchange="onTaskPresetChange()">
            <option value="three_view_whitebg_single_image">三视图白底</option>
            <option value="style_transfer">风格迁移</option>
            <option value="custom">自定义</option>
          </select></div>
        <div style="flex:1"><label>评分方法</label>
          <select id="cb-scoring">
            <option value="hybrid_with_rank_calibration">混合评分 + 排序校准</option>
            <option value="binary">纯二分</option>
            <option value="continuous">纯连续</option>
          </select></div>
      </div>
      <label>视图（逗号分隔）</label>
      <input type="text" id="cb-views" value="front,side,perspective">

      <div class="cb-section-title mt-3">评分维度</div>
      <div id="cb-dims"></div>
      <button class="btn btn-ghost btn-sm" onclick="addBenchDim()">＋ 添加维度</button>

      <div class="cb-section-title mt-3">题目 samples</div>
      <div id="cb-samples"></div>
      <button class="btn btn-ghost btn-sm" onclick="addBenchSample()">＋ 添加题目</button>

      <div class="row mt-3">
        <button class="btn btn-primary" id="cb-submit" onclick="submitCreateBench()">创建 benchmark</button>
        <button class="btn btn-ghost" data-close>取消</button>
      </div>
    </div>`;
}

function onTaskPresetChange() {
  const t = document.getElementById("cb-task").value;
  const v = document.getElementById("cb-views");
  if (t === "three_view_whitebg_single_image") v.value = "front,side,perspective";
  else if (t === "style_transfer") v.value = "";
}

function syncDimsFromDom() {
  document.querySelectorAll("#cb-dims .dim-editor-row").forEach((row, i) => {
    if (!_benchDims[i]) return;
    row.querySelectorAll(".cb-dim").forEach((inp) => {
      const k = inp.dataset.k;
      let val = inp.value;
      if (k === "weight_init") val = parseFloat(val) || 0;
      if (k === "ref_needed") val = (val === "true");
      _benchDims[i][k] = val;
    });
  });
}

function addBenchDim() {
  syncDimsFromDom();
  _benchDims.push({ dim: "", desc: "", weight_init: 0, ref_needed: false, scoring_type: "binary", check_items: "", rubric_ref: "" });
  renderDimEditor();
}

function removeBenchDim(i) {
  syncDimsFromDom();
  _benchDims.splice(i, 1);
  renderDimEditor();
}

function renderDimEditor() {
  syncDimsFromDom();
  const el = document.getElementById("cb-dims");
  if (!el) return;
  el.innerHTML = _benchDims.map((d, i) => `<div class="dim-editor-row">
    <div class="row" style="gap:6px;align-items:flex-end">
      <div style="flex:2"><label>维度名</label><input class="cb-dim" data-i="${i}" data-k="dim" value="${esc(d.dim)}"></div>
      <div style="flex:1"><label>权重</label><input class="cb-dim" data-i="${i}" data-k="weight_init" type="number" step="0.01" min="0" max="1" value="${d.weight_init}"></div>
      <div style="flex:1"><label>类型</label><select class="cb-dim" data-i="${i}" data-k="scoring_type" onchange="renderDimEditor()"><option value="binary"${d.scoring_type === "binary" ? " selected" : ""}>二分</option><option value="continuous"${d.scoring_type === "continuous" ? " selected" : ""}>连续</option></select></div>
      <div style="flex:1"><label>需参考图</label><select class="cb-dim" data-i="${i}" data-k="ref_needed"><option value="true"${d.ref_needed ? " selected" : ""}>是</option><option value="false"${!d.ref_needed ? " selected" : ""}>否</option></select></div>
      <button class="btn btn-ghost btn-sm" onclick="removeBenchDim(${i})">×</button>
    </div>
    <label>说明</label><input class="cb-dim" data-i="${i}" data-k="desc" value="${esc(d.desc || "")}">
    <label>check_items <span class="muted" style="font-size:11px">（二分维度；每行一项，如 "C1 ..."）</span></label>
    <textarea class="cb-dim" data-i="${i}" data-k="check_items" rows="3">${esc(d.check_items || "")}</textarea>
    <label>rubric_ref <span class="muted" style="font-size:11px">（连续维度）</span></label>
    <input class="cb-dim" data-i="${i}" data-k="rubric_ref" value="${esc(d.rubric_ref || "")}">
  </div>`).join("");
}

function syncSamplesFromDom() {
  document.querySelectorAll("#cb-samples .sample-editor-row").forEach((row, i) => {
    if (!_benchSamples[i]) return;
    row.querySelectorAll(".cb-sample").forEach((inp) => {
      if (_benchSamples[i]) _benchSamples[i][inp.dataset.k] = inp.value;
    });
  });
}

function addBenchSample() {
  syncSamplesFromDom();
  _benchSamples.push({ sample_id: "", product: "", category: "" });
  renderSampleEditor();
}

function removeBenchSample(i) {
  syncSamplesFromDom();
  _benchSamples.splice(i, 1);
  renderSampleEditor();
}

function renderSampleEditor() {
  syncSamplesFromDom();
  const el = document.getElementById("cb-samples");
  if (!el) return;
  el.innerHTML = _benchSamples.map((s, i) => `<div class="sample-editor-row">
    <div class="row" style="gap:6px;align-items:flex-end">
      <div style="flex:1"><label>sample_id</label><input class="cb-sample" data-i="${i}" data-k="sample_id" value="${esc(s.sample_id)}"></div>
      <div style="flex:2"><label>产品</label><input class="cb-sample" data-i="${i}" data-k="product" value="${esc(s.product || "")}"></div>
      <div style="flex:1"><label>品类</label><input class="cb-sample" data-i="${i}" data-k="category" value="${esc(s.category || "")}"></div>
      <button class="btn btn-ghost btn-sm" onclick="removeBenchSample(${i})">×</button>
    </div>
    <label>target 参考图 <span class="muted" style="font-size:11px">（对比型维度评判锚；可选，支持 jpg/png）</span></label>
    <input type="file" accept="image/*" class="cb-target">
  </div>`).join("");
}

async function submitCreateBench() {
  syncDimsFromDom();
  syncSamplesFromDom();
  const benchId = document.getElementById("cb-bench").value.trim();
  const scene = document.getElementById("cb-scene").value.trim();
  const description = document.getElementById("cb-desc").value.trim();
  const scoringMethod = document.getElementById("cb-scoring").value;
  const taskType = document.getElementById("cb-task").value;
  const views = document.getElementById("cb-views").value.trim();
  if (!benchId) { toast("请填 bench_id", "warn"); return; }
  if (!_benchDims.length || !_benchSamples.length) { toast("至少需要一个维度和一道题目", "warn"); return; }
  const sids = _benchSamples.map((s) => s.sample_id.trim()).filter(Boolean);
  if (sids.length !== _benchSamples.length || new Set(sids).size !== sids.length) {
    toast("每个题目需有唯一非空 sample_id", "warn"); return;
  }

  const dimensions = _benchDims.map((d) => ({
    dim: (d.dim || "").trim(),
    desc: (d.desc || "").trim() || null,
    weight_init: d.weight_init,
    ref_needed: !!d.ref_needed,
    scoring_type: d.scoring_type,
    check_items: d.scoring_type === "binary"
      ? (d.check_items || "").split("\n").map((x) => x.trim()).filter(Boolean) : null,
    rubric_ref: d.scoring_type === "continuous" ? ((d.rubric_ref || "").trim() || null) : null,
  }));

  const fd = new FormData();
  fd.append("bench_id", benchId);
  fd.append("scene", scene);
  fd.append("description", description);
  fd.append("scoring_method", scoringMethod);
  fd.append("task_type", taskType);
  fd.append("views", views);
  fd.append("dimensions", JSON.stringify(dimensions));
  fd.append("samples", JSON.stringify(_benchSamples.map((s) => ({
    sample_id: s.sample_id.trim(), product: (s.product || "").trim() || null, category: (s.category || "").trim() || null,
  }))));

  // 配对每个 sample 的 target 图（按行内 sample_id，避免编辑后错位）
  document.querySelectorAll("#cb-samples .sample-editor-row").forEach((row) => {
    const sidIn = row.querySelector('.cb-sample[data-k="sample_id"]');
    const fileIn = row.querySelector(".cb-target");
    const sid = (sidIn?.value || "").trim();
    if (sid && fileIn?.files?.[0]) fd.append(`target_${sid}`, fileIn.files[0]);
  });

  const btn = document.getElementById("cb-submit");
  btn.disabled = true; btn.textContent = "创建中…";
  try {
    const r = await fetch("/api/benchmarks", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`${r.status} ${await r.text().catch(() => r.statusText)}`);
    const data = await r.json();
    closeModal();
    toast("benchmark 已创建", "success");
    location.hash = `#/benchmarks/${encodeURIComponent(data.bench_id)}`;
  } catch (e) {
    btn.disabled = false; btn.textContent = "创建 benchmark";
    handleError(e, "创建失败");
  }
}

// ============ 路由 ============
async function router() {
  const h = location.hash.slice(1) || "/";
  closeModal(); closeLightbox();
  if (_loopPollTimer) { clearInterval(_loopPollTimer); _loopPollTimer = null; }
  if (_distillPollTimer) { clearInterval(_distillPollTimer); _distillPollTimer = null; }
  if (_runningPollTimer) { clearInterval(_runningPollTimer); _runningPollTimer = null; }

  try {
    if (h === "/" || h === "") await viewOverview();
    else if (h === "/running") await viewRunning();
    else if (h === "/scoring") await viewScoring();
    else if (h.startsWith("/scoring/")) {
      const [bench, sample] = h.slice(9).split("/");
      await viewScoring(decodeURIComponent(bench), decodeURIComponent(sample));
    } else if (h.startsWith("/loop/")) await viewLoop(decodeURIComponent(h.slice(6)));
    else if (h.startsWith("/experience/")) await viewExperience(decodeURIComponent(h.slice(12)));
    else if (h === "/benchmarks") await viewBenchmarks();
    else if (h.startsWith("/benchmarks/")) await viewBenchmarkDetail(decodeURIComponent(h.slice(12)));
    else if (h === "/config") await viewConfig();
    else document.getElementById("app").innerHTML = '<div class="empty">404</div>';
  } catch (e) {
    handleError(e, "页面加载失败");
    document.getElementById("app").innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

window.addEventListener("hashchange", router);
window.addEventListener("load", () => { initCompactMode(); router(); });
