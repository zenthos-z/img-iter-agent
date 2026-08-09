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

const DISTILL_LABEL = { idle: "未蒸馏", running: "蒸馏中", done: "完成", error: "失败", no_runs: "无 run" };

const NODE_LABEL = {
  generator: { name: "Generator", desc: "生成提示词 + 出图" },
  critic: { name: "Critic", desc: "多维度打分" },
  summarizer: { name: "Summarizer", desc: "归纳经验 + 写记录" },
  human_review: { name: "等审批", desc: "等待人工裁决" },
};

// 路由缓存
window.__overviewCache = null;
window.__loopCache = { current: null };

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
          <button class="btn btn-ghost btn-sm exp-entry" title="跨 loop 通用经验（可移植 SKILL.md）" onclick="event.stopPropagation(); location.hash='#/experience/${esc(bench.bench_id)}'">通用经验${expBadge}</button>
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
    app.innerHTML = html;
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
      <button class="btn btn-ghost btn-sm" onclick="startNewLoop('${esc(bench.bench_id)}','${esc(sample.sample_id)}')">继续跑</button>
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
      body = `<div class="progress-track"><div class="progress-fill" style="width:60%"></div></div>
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

async function viewLoop(loopId) {
  if (_loopPollTimer) { clearInterval(_loopPollTimer); _loopPollTimer = null; }
  setNavActive("");
  setTitle("loop 详情");
  const app = document.getElementById("app");
  app.innerHTML = skeleton(4);

  try {
    const loop = await api(`/loops/${loopId}`);
    window.__loopCache.current = loop;
    const st = loop.status;

    setTitle(loopId, `<a class="btn btn-secondary btn-sm" href="#/scoring/${esc(loop.bench_id)}/${esc(loop.sample_id)}">人工排序</a>`);

    let html = `<div class="page-header">
      <div class="title-block">
        <nav class="breadcrumb">
          <a href="#/">总览</a>
          <span>/</span>
          <a href="#/">${esc(loop.bench_id)}</a>
          <span>/</span>
          <span>${esc(loop.sample_id)}</span>
        </nav>
        <h1>${esc(loopId)}</h1>
        <div class="muted">${esc(loop.model)} ${loop.note ? "· " + esc(loop.note) : ""}</div>
      </div>
    </div>`;

    // 状态条
    html += `<div class="status-bar">
      <span class="status-pill"><span class="dot ${st}"></span>${esc(STATUS_LABEL[st] || st)}</span>
      ${loop.round != null ? `<span class="muted">当前轮: ${loop.round}</span>` : ""}
      ${loop.finished_at ? `<span class="muted">完成: ${esc(loop.finished_at)}</span>` : ""}
      <div class="control-actions">
        ${st === "awaiting_review" ? `
          <button class="btn btn-primary" onclick="resumeLoop('${esc(loopId)}','continue')">继续下一轮</button>
          <button class="btn btn-danger" onclick="resumeLoop('${esc(loopId)}','stop')">停止</button>` : ""}
        ${st === "running" ? `<span class="muted" id="cur-node">执行中…</span>` : ""}
      </div>
    </div>`;

    // 人工提示词面板（常驻：运行中可增删，下一轮生效）
    html += `<div id="hints-panel" class="mt-3"></div>`;

    // error
    if (st === "error" && loop.last_error) {
      const err = loop.last_error.split("\n").find((l) => /失败|Error|error|Forbidden|Timeout|4\d\d|5\d\d/.test(l)) || loop.last_error.split("\n")[0];
      html += `<div class="alert">
        <div class="alert-title">运行出错</div>
        <div>${esc(err.trim())}</div>
        <details><summary class="muted">完整错误</summary><pre>${esc(loop.last_error)}</pre></details>
      </div>`;
    }

    // awaiting_review
    if (st === "awaiting_review" && loop.interrupt_payload) {
      const ip = loop.interrupt_payload;
      html += `<div class="review-card">
        <div class="review-info">
          <div class="review-title">第 ${ip.round} 轮等待审批</div>
          <div class="review-meta">还原度: ${fmt(ip.restoration)}${ip.failed_items?.length ? " · 失败项: " + esc(ip.failed_items.join(", ")) : ""}</div>
        </div>
        <button class="btn btn-primary" onclick="resumeLoop('${esc(loopId)}','continue')">继续下一轮</button>
        <button class="btn btn-danger" onclick="resumeLoop('${esc(loopId)}','stop')">停止</button>
      </div>`;
    }

    // 目标 sample
    if (loop.target_image || loop.target_md) {
      html += `<div class="card card-padded loop-target-card">
        <h3>目标 sample · ${esc(loop.sample_id)}</h3>
        <div class="loop-target-body">
          ${loop.target_image ? `<img src="${imgURL(loop.target_image)}" data-lb="${imgURL(loop.target_image)}" alt="target">` : '<div class="trace-img"></div>'}
          <div class="target-desc">${loop.target_md ? renderMarkdown(loop.target_md) : "暂无目标描述"}</div>
        </div>
      </div>`;
    }

    // Trace 时间线
    html += `<h2>迭代轨迹（${loop.traces.length} trace）</h2>`;
    if (!loop.traces.length) {
      html += `<div class="card card-padded muted">该 loop 还没有 trace。</div>`;
    } else {
      const bestRestoration = Math.max(...loop.traces.filter((x) => x.verdict).map((x) => x.verdict.restoration));
      html += `<div class="timeline">`;
      for (let i = 0; i < loop.traces.length; i++) {
        const t = loop.traces[i];
        const img = t.output_image_refs[0];
        const isBest = t.verdict && t.verdict.restoration === bestRestoration;
        const prev = i > 0 ? loop.traces[i - 1] : null;
        const hasDiff = prev && prev.prompt && prev.prompt !== t.prompt;
        html += `<div class="timeline-item ${isBest ? "best" : ""}">
          <div class="timeline-marker">${t.round}</div>
          <div class="timeline-body">
            <div class="trace-header">
              <span class="round">第 ${t.round} 轮</span>
              <span class="muted">${esc(t.test_variable || "基线")}</span>
              ${isBest ? '<span class="badge badge-primary">最佳</span>' : ""}
              ${t.human_rank != null ? `<span class="badge">人工排序 #${t.human_rank}</span>` : ""}
              <span class="score ${isBest ? "best" : ""}">还原度 ${t.verdict ? fmt(t.verdict.restoration) : "—"}</span>
            </div>
            <div class="trace-content">
              <div>
                ${img ? `<img class="trace-img" src="${imgURL(img, loopId)}" data-lb="${imgURL(img, loopId)}" alt="">` : '<div class="trace-img"></div>'}
              </div>
              <div class="trace-detail">
                <div class="trace-params">
                  ${t.model ? `<div class="param"><span class="pk">模型</span><span class="pv">${esc(t.model)}</span></div>` : ""}
                  ${t.size ? `<div class="param"><span class="pk">尺寸</span><span class="pv">${esc(t.size)}</span></div>` : ""}
                  ${t.gen_mode ? `<div class="param"><span class="pk">模式</span><span class="pv">${esc(t.gen_mode)}</span></div>` : ""}
                  ${t.baseline_ref ? `<div class="param"><span class="pk">基线</span><span class="pv">${esc(t.baseline_ref)}</span></div>` : ""}
                  ${t.ts ? `<div class="param"><span class="pk">时间</span><span class="pv">${esc(t.ts)}</span></div>` : ""}
                </div>
                ${t.verdict ? renderDimList(t.verdict.dimensions) : ""}
                <div class="prompt-tools">
                  <button class="btn btn-ghost btn-sm" data-ptog="${i}">展开 prompt</button>
                  ${hasDiff ? `<button class="btn btn-ghost btn-sm" data-pdiff="${i}">对比第${prev.round}轮</button>` : ""}
                </div>
                <div class="prompt-box" id="prompt-box-${i}" hidden><pre>${esc(t.prompt || "(空)")}</pre></div>
              </div>
            </div>
          </div>
        </div>`;
      }
      html += `</div>`;
    }

    // 经验沉淀
    if (loop.conclusions && loop.conclusions.length) {
      html += `<details class="card card-padded mt-4" open>
        <summary style="cursor:pointer;font-weight:600">经验沉淀（${loop.conclusions.length} 条）</summary>
        <div class="mt-3">${renderConclusions(loop.conclusions)}</div>
      </details>`;
    }

    app.innerHTML = html;

    // 人工提示词面板（异步填充）
    renderHintsPanel(loopId);

    // prompt 事件委托
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
        } catch (_) {}
      }, 2000);
    }
  } catch (e) {
    handleError(e, "加载 loop 详情失败");
    app.innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
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

// ============ 人工提示词面板（loop 详情页，运行中可增删） ============
async function renderHintsPanel(loopId) {
  const el = document.getElementById("hints-panel");
  if (!el) return;
  let hints = [];
  try {
    hints = ((await api(`/loops/${loopId}/hints`)).hints) || [];
  } catch (_) { return; }
  const scopeBadge = (s) => s === "sample"
    ? `<span class="badge badge-primary">持久</span>`
    : `<span class="badge badge-ghost">临时</span>`;
  const renderHint = (h) => `<div class="hint-row">
    ${scopeBadge(h.scope)}
    <span class="badge">${esc(h.agent)}</span>
    <span class="hint-text">${esc(h.text)}</span>
    <button class="btn btn-ghost btn-sm" onclick="removeLoopHint('${esc(loopId)}','${esc(h.id)}')">×</button>
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
  el.innerHTML = `<div class="card card-padded hints-card">
    <div class="hints-head">
      <h3 style="margin:0">人工提示词</h3>
      <span class="muted" style="font-size:11px">给 generator/critic 追加；下一轮生效。临时=仅本 loop；持久=该考题所有 loop。</span>
    </div>
    <div class="hint-list mt-2">${hints.length ? hints.map(renderHint).join("") : '<div class="muted" style="font-size:12px">暂无提示词。</div>'}</div>
    ${editor}
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
    renderHintsPanel(loopId);
  } catch (e) { handleError(e, "添加失败"); }
}

async function removeLoopHint(loopId, hintId) {
  try {
    await api(`/loops/${loopId}/hints/${hintId}`, { method: "DELETE" });
    toast("已删除", "success");
    renderHintsPanel(loopId);
  } catch (e) { handleError(e, "删除失败"); }
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

async function viewConfig() {
  setNavActive("config");
  setTitle("Agent 设置");
  const app = document.getElementById("app");
  app.innerHTML = skeleton(2);

  try {
    const data = await api("/config");
    app.innerHTML = `<div class="config-layout">
      <nav class="config-tabs" aria-label="Agent tabs">
        ${["generator", "critic", "summarizer"].map((a) => `<button class="config-tab ${a === _cfgAgent ? "active" : ""}" onclick="switchCfg('${a}')">${a}</button>`).join("")}
      </nav>
      <div class="card card-padded config-form" id="cfg-form"></div>
    </div>`;
    renderCfgForm(data.agents);
  } catch (e) { handleError(e, "加载配置失败"); }
}

function switchCfg(a) { _cfgAgent = a; viewConfig(); }

function renderCfgForm(agents) {
  const a = agents.find((x) => x.agent === _cfgAgent) || agents[0];
  document.getElementById("cfg-form").innerHTML = `
    <h2>${esc(_cfgAgent)} 配置</h2>
    <label>模型 id</label>
    <input type="text" id="cfg-model" value="${esc(a.model)}">
    <label>系统提示词</label>
    <textarea id="cfg-prompt">${esc(a.system_prompt)}</textarea>
    <div class="row mt-3">
      <button class="btn btn-primary" onclick="saveCfg()">保存</button>
      <button class="btn btn-ghost" onclick="resetCfg()">恢复默认</button>
      <span id="cfg-msg" class="muted"></span>
    </div>`;
}

async function saveCfg() {
  const model = document.getElementById("cfg-model").value;
  const prompt = document.getElementById("cfg-prompt").value;
  try {
    await api(`/config/${_cfgAgent}`, { method: "POST", body: JSON.stringify({ system_prompt: prompt, model }) });
    toast("配置已保存", "success");
    document.getElementById("cfg-msg").textContent = "已保存 ✓";
    setTimeout(() => { const m = document.getElementById("cfg-msg"); if (m) m.textContent = ""; }, 2000);
  } catch (e) { handleError(e, "保存失败"); }
}

async function resetCfg() {
  if (!confirm("确定恢复为代码默认配置？")) return;
  try {
    await api(`/config/${_cfgAgent}/reset`, { method: "POST" });
    toast("已恢复默认", "success");
    viewConfig();
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
  const existing = sample?.loops?.[0];
  const product = sample?.product ? `（${esc(sample.product)}）` : "";

  let action, btnText;
  if (existing) {
    action = `继续 loop <code>${esc(existing.loop_id)}</code>（已有 ${existing.n_traces} 轮）`;
    btnText = "继续跑下一轮";
  } else {
    action = `新建 loop <code>${esc(benchId)}-${esc(sampleId)}</code>`;
    btnText = "启动 loop";
  }
  const roundsDesc = rounds > 1
    ? `将自动连跑 <strong>${rounds}</strong> 轮，跑满后停在审批节点。`
    : `首轮回跑到人工审批节点停下。`;

  document.getElementById("nl-preview").innerHTML =
    `将对 <code>${esc(sampleId)}</code>${product} 用 <code>${esc(model)}</code> ${action}<br>` + roundsDesc;
  document.getElementById("nl-submit").textContent = btnText;
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
          <button class="btn btn-secondary btn-sm" title="规范可安装技能包（SKILL.md + references + assets）" onclick="exportSkillPackage('${esc(benchId)}')">导出技能包(.skill)</button>
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
  // 下载规范 .skill 包（目录 zip：SKILL.md + references/ + assets/），可装到别的 agent 工具
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
    else if (h === "/config") await viewConfig();
    else document.getElementById("app").innerHTML = '<div class="empty">404</div>';
  } catch (e) {
    handleError(e, "页面加载失败");
    document.getElementById("app").innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}

window.addEventListener("hashchange", router);
window.addEventListener("load", router);
