/* img-iter-agent 打分台 — 前端 SPA (Vanilla JS, 无依赖)
 * 5 屏：①总览 ②loop详情 ③trace详情(modal) ④排序 ⑤Agent设置
 * hash 路由：#/  #/loop/:id  #/scoring/:bench/:sample  #/config
 */
"use strict";

const STATUS_LABEL = {
  finished: "已结束", running: "运行中", awaiting_review: "等审批",
  error: "错误", unknown: "未知", idle: "空闲",
};
const STATUS_ORDER = { awaiting_review: 0, running: 1, error: 2, finished: 3, unknown: 4, idle: 5 };
// graph 节点 → 中文说明（让用户看到当前是哪个 agent 在操作）
const NODE_LABEL = {
  generator: { name: "Generator", desc: "生成提示词 + 出图" },
  critic: { name: "Critic", desc: "多维度打分" },
  summarizer: { name: "Summarizer", desc: "归纳经验 + 写记录" },
  human_review: { name: "等审批", desc: "等待人工裁决" },
};

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

// ---- lightbox ----
function openLightbox(src) {
  document.getElementById("lightbox-img").src = src;
  document.getElementById("lightbox").classList.remove("hidden");
}
function closeLightbox() {
  document.getElementById("lightbox").classList.add("hidden");
}
document.getElementById("lightbox").addEventListener("click", (e) => {
  if (e.target.dataset.close !== undefined) closeLightbox();
});

// ---- trace 详情 modal ----
// 状态显式挂 window（顶层 let 在某些调用路径下闭包不可见，挂 window 最稳）
window.__loopCache = window.__loopCache || {};
let _curTraceIdx = 0;
function openTraceModal(idx) {
  const loop = window.__loopCache.current;
  if (!loop) return;
  _curTraceIdx = idx;
  const t = loop.traces[idx];
  const body = document.getElementById("trace-modal-body");
  body.innerHTML = traceDetailHTML(t, idx);
  document.getElementById("trace-modal").classList.remove("hidden");
}
function closeTraceModal() {
  document.getElementById("trace-modal").classList.add("hidden");
}
document.getElementById("trace-modal").addEventListener("click", (e) => {
  if (e.target.dataset.close !== undefined) closeTraceModal();
});

function traceDetailHTML(t, idx) {
  const loop = window.__loopCache.current;
  const v = t.verdict;
  const dims = v
    ? v.dimensions
        .map(
          (d) => `<div class="dim"><span>${esc(d.dim)} <span class="muted">(${d.scoring_type})</span></span><span class="v">${fmt(d.value)}</span></div>
        ${d.failed_items.map((f) => `<div class="failed">${esc(f.id)}: ${esc(f.reason)}</div>`).join("")}
        ${d.raw ? `<div class="failed">${esc(d.raw)}</div>` : ""}`
        )
        .join("")
    : '<div class="muted">无评分</div>';
  const img = t.output_image_refs[0];
  const target = loop?.target_image;
  // 本轮 prompt：默认折叠，可展开；若有上一轮则提供 diff 对比
  const prev = idx > 0 ? loop.traces[idx - 1] : null;
  const hasDiff = prev && prev.prompt && prev.prompt !== t.prompt;
  const promptBlock = `
    <div class="prompt-tools">
      <button class="ghost sm" data-prompt-toggle="${idx}">展开 / 折叠</button>
      ${hasDiff ? `<button class="ghost sm" data-prompt-diff="${idx}">与第${prev.round}轮对比</button>` : `<span class="muted sm">${idx === 0 ? "首轮基线" : "与上轮相同"}</span>`}
    </div>
    <div class="prompt-full" id="prompt-full-${idx}" hidden><pre>${esc(t.prompt || "(空)")}</pre></div>`;
  return `
    <h1>trace ${esc(t.trace_id)} · 第 ${t.round} 轮</h1>
    <div class="two-col">
      <div>
        ${img ? `<img src="${imgURL(img, loop.loop_id)}" data-lb="${imgURL(img, loop.loop_id)}">` : '<div class="muted">无图</div>'}
        <div class="muted" style="margin-top:6px">还原度: <span class="res">${fmt(t.verdict?.restoration)}</span></div>
      </div>
      <div>
        ${target ? `<img src="${imgURL(target)}" data-lb="${imgURL(target)}">` : '<div class="muted">无 target</div>'}
        ${loop?.target_md ? `<div class="prompt-snippet" style="max-height:6em">${esc(loop.target_md.slice(0, 400))}</div>` : ""}
      </div>
    </div>
    <h2 style="margin-top:16px">Critic 评分</h2>
    <div class="dim-list">${dims}</div>
    ${t.delta_note ? `<h2 style="margin-top:16px">本轮改动</h2><div class="prompt-full"><pre>${esc(t.delta_note)}</pre></div>` : ""}
    <h2 style="margin-top:12px">本轮 prompt</h2>
    ${promptBlock}
    <div class="row" style="margin-top:14px">
      <a class="btn secondary" href="#/scoring/${loop.bench_id}/${loop.sample_id}">给 ${esc(loop.sample_id)} 排序</a>
      <button class="ghost" data-close>关闭</button>
    </div>`;
}

// prompt 展开/折叠 + diff 对比
function togglePromptFull(idx) {
  const el = document.getElementById(`prompt-full-${idx}`);
  if (el) el.hidden = !el.hidden;
}

// 简易逐词 diff：把 cur 相对 prev 的差异标红(删除)/标绿(新增)，按空格/标点切词
function diffPrompt(prev, cur) {
  const toks = (s) => (s || "").split(/(\s+)/); // 保留空白 token
  const a = toks(prev), b = toks(cur);
  // LCS dp
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
  while (i < n) { out.push(`<del>${esc(a[i++])}</del>`); }
  while (j < m) { out.push(`<ins>${esc(b[j++])}</ins>`); }
  return out.join("");
}

function showPromptDiff(idx) {
  const loop = window.__loopCache.current;
  if (!loop || idx <= 0) return;
  const prev = loop.traces[idx - 1], cur = loop.traces[idx];
  const box = document.getElementById(`prompt-full-${idx}`);
  if (!box) return;
  const btn = document.querySelector(`button[data-prompt-diff="${idx}"]`);
  // 已在显示 diff → 还原为原始 prompt（保持展开）
  if (box.dataset.diffing === "1") {
    box.dataset.diffing = "";
    box.innerHTML = `<pre>${esc(cur.prompt || "(空)")}</pre>`;
    if (btn) btn.textContent = `与第${prev.round}轮对比`;
    return;
  }
  box.dataset.diffing = "1";
  box.hidden = false;
  box.innerHTML = `<pre class="diff">${diffPrompt(prev.prompt, cur.prompt)}</pre>`;
  if (btn) btn.textContent = "还原 prompt";
}

// 绑定 modal 内的 lightbox、关闭、prompt 展开/diff
document.getElementById("trace-modal-body").addEventListener("click", (e) => {
  const t = e.target;
  if (t.dataset.lb) { openLightbox(t.dataset.lb); return; }
  if (t.dataset.promptToggle !== undefined) { togglePromptFull(+t.dataset.promptToggle); return; }
  if (t.dataset.promptDiff !== undefined) { showPromptDiff(+t.dataset.promptDiff); return; }
  if (t.dataset.close !== undefined) closeTraceModal();
});

// ============ 屏① 总览 ============
async function viewOverview() {
  const data = await api("/overview");
  const app = document.getElementById("app");
  if (!data.benches.length) {
    app.innerHTML = '<div class="empty">还没有 benchmark。先放一个，或启动新 loop。</div><button onclick="startNewLoop()">启动新 loop</button>';
    return;
  }
  let html = '<h1>总览</h1><button class="secondary" onclick="startNewLoop()">＋ 启动新 loop</button>';
  // 渲染单个 sample 的行（有 loop 的用）
  const renderSampleRow = (bench, sample) => {
    const pendingCls = sample.pending === 0 ? "zero" : "pending";
    let h = `<div class="sample-row">
      <div class="sample-head">
        <span class="sample-title">${esc(sample.sample_id)} ${sample.product ? "· " + esc(sample.product) : ""}</span>
        <span class="badge">${sample.n_traces} trace · ${sample.loops.length} loop</span>
        <span class="badge ${pendingCls}">待打分 ${sample.pending}</span>
        <a class="btn" href="#/scoring/${esc(bench.bench_id)}/${esc(sample.sample_id)}">给 ${esc(sample.sample_id)} 排序 →</a>
      </div>
      <div class="loop-cards">`;
    for (const l of sample.loops) {
      const thumb = l.thumbnail ? imgURL(l.thumbnail, l.loop_id) : "";
      h += `<div class="loop-card" onclick="location.hash='#/loop/${esc(l.loop_id)}'">
        ${thumb ? `<img class="loop-thumb" src="${thumb}">` : '<div class="loop-thumb"></div>'}
        <div class="loop-meta">
          <span class="dot ${l.status}"></span>${esc(STATUS_LABEL[l.status] || l.status)} · ${l.n_traces}轮<br>
          ${l.best_restoration != null ? `还原度 <span class="res">${fmt(l.best_restoration)}</span>` : ""}
        </div>
      </div>`;
    }
    h += `</div></div>`;
    return h;
  };
  for (const bench of data.benches) {
    // 拆分：有 loop 的（已运行）vs 无 loop 的（未运行题库）
    const active = bench.samples.filter(s => s.loops.length > 0);
    const idle = bench.samples.filter(s => s.loops.length === 0);
    const hasActive = active.length > 0;
    html += `<h2>${esc(bench.bench_id)} ${bench.description ? "· " + esc(bench.description) : ""}</h2>`;
    if (hasActive) {
      for (const sample of active) html += renderSampleRow(bench, sample);
    } else {
      html += '<div class="muted">这个 benchmark 还没有任何 loop。</div>';
    }
    // 未运行的题折叠到底部「可选题库」
    if (idle.length) {
      html += `<details class="idle-bank" ${hasActive ? "" : "open"} style="margin-top:8px">
        <summary class="muted">可选题库（${idle.length} 道未运行的题）</summary>
        <div style="margin-top:6px;display:flex;flex-wrap:wrap;gap:6px">`;
      for (const s of idle) {
        html += `<button class="ghost sm" onclick="location.hash='#/scoring/${esc(bench.bench_id)}/${esc(s.sample_id)}'">${esc(s.sample_id)} ${s.product ? "· " + esc(s.product) : ""}</button>`;
      }
      html += `</div></details>`;
    }
  }
  app.innerHTML = html;
}

// ============ 屏② loop 详情 ============
let _loopPollTimer = null;
async function viewLoop(loopId) {
  if (_loopPollTimer) { clearInterval(_loopPollTimer); _loopPollTimer = null; }
  const loop = await api(`/loops/${loopId}`);
  window.__loopCache.current = loop; // 供 trace 详情 modal 按 idx 取数据
  const app = document.getElementById("app");
  const st = loop.status;
  const ctrl = loop.status === "awaiting_review" || loop.status === "running" || loop.status === "error";
  let html = `<div class="row" style="margin-bottom:8px"><a href="#/">← 总览</a></div>`;
  html += `<h1>loop ${esc(loopId)}</h1>
    <div class="muted">${esc(loop.bench_id)} · ${esc(loop.sample_id)} · ${esc(loop.model)} · ${esc(loop.note || "")}</div>`;

  // 状态条（监测 + 远程控制）
  html += `<div class="status-bar">
    <span class="dot ${st}"></span><strong>${esc(STATUS_LABEL[st] || st)}</strong>
    ${loop.round != null ? `<span class="muted">当前轮: ${loop.round}</span>` : ""}
    ${loop.finished_at ? `<span class="muted">完成: ${esc(loop.finished_at)}</span>` : ""}
    ${st === "awaiting_review" ? `
      <button onclick="resumeLoop('${esc(loopId)}','continue')">继续下一轮</button>
      <button class="danger" onclick="resumeLoop('${esc(loopId)}','stop')">停止</button>` : ""}
    ${st === "running" ? `<span class="muted" id="cur-node">执行中…</span>` : ""}
  </div>`;

  // error 时显示错误摘要（从冗长 traceback 提取首行有用信息）
  if (st === "error" && loop.last_error) {
    // 取 traceback 里第一行 "X失败: ..." 或首条 httpx 错误
    const err = loop.last_error.split("\n").find(l => /失败|Error|error|Forbidden|Timeout|4\d\d|5\d\d/.test(l))
      || loop.last_error.split("\n")[0];
    html += `<div class="card" style="margin-bottom:16px;border-color:#e55">
      <strong style="color:#c33">出错</strong>（已跑 ${loop.traces.length} 轮）<br>
      <span class="muted">${esc(err.trim())}</span>
      <details style="margin-top:6px"><summary class="muted">完整错误</summary>
        <pre style="max-height:200px;overflow:auto;font-size:11px;white-space:pre-wrap">${esc(loop.last_error)}</pre></details>
    </div>`;
  }

  // 等审批时显示 interrupt payload
  if (st === "awaiting_review" && loop.interrupt_payload) {
    const ip = loop.interrupt_payload;
    html += `<div class="card" style="margin-bottom:16px">
      <strong>等审批 · 第 ${ip.round} 轮结果</strong><br>
      <span class="muted">还原度: ${fmt(ip.restoration)}</span>
      ${ip.failed_items?.length ? `<br><span class="muted">失败项: ${esc(ip.failed_items.join(", "))}</span>` : ""}
    </div>`;
  }

  // trace 列
  html += `<h2>迭代轨迹（${loop.traces.length} trace）</h2><div class="trace-row" id="trace-row">`;
  for (let i = 0; i < loop.traces.length; i++) {
    const t = loop.traces[i];
    const img = t.output_image_refs[0];
    const best = loop.traces.length && t.verdict && t.verdict.restoration === Math.max(...loop.traces.filter(x=>x.verdict).map(x=>x.verdict.restoration));
    const prev = i > 0 ? loop.traces[i - 1] : null;
    const hasDiff = prev && prev.prompt && prev.prompt !== t.prompt;
    html += `<div class="trace-col" data-idx="${i}">
      ${img ? `<img src="${imgURL(img, loopId)}">` : '<div class="loop-thumb"></div>'}
      <div class="trace-line">
        第 ${t.round} 轮 · ${esc(t.test_variable || "基线")}
        ${t.verdict ? `<br>还原度 <span class="res ${best ? "best" : ""}">${fmt(t.verdict.restoration)}${best ? " ★" : ""}</span>` : ""}
        ${t.human_rank != null ? `<br>人工排序: <span class="res">${t.human_rank}</span>` : ""}
      </div>
      <div class="trace-params">
        ${t.model ? `<div class="param"><span class="pk">模型</span><span class="pv">${esc(t.model)}</span></div>` : ""}
        ${t.size ? `<div class="param"><span class="pk">尺寸</span><span class="pv">${esc(t.size)}</span></div>` : ""}
        ${t.gen_mode ? `<div class="param"><span class="pk">模式</span><span class="pv">${esc(t.gen_mode)}</span></div>` : ""}
        ${t.baseline_ref ? `<div class="param"><span class="pk">基线</span><span class="pv">${esc(t.baseline_ref)}</span></div>` : ""}
        ${t.ts ? `<div class="param"><span class="pk">时间</span><span class="pv">${esc(t.ts)}</span></div>` : ""}
      </div>
      <div class="prompt-tools">
        <button class="ghost sm" data-ptog="${i}">展开 prompt</button>
        ${hasDiff ? `<button class="ghost sm" data-pdiff="${i}">对比第${prev.round}轮</button>` : ""}
      </div>
      <div class="prompt-full" id="row-prompt-${i}" hidden><pre>${esc(t.prompt || "(空)")}</pre></div>
    </div>`;
  }
  html += `</div>`;

  // 经验知识库（结构化结论，按 status 分组）
  if (loop.conclusions && loop.conclusions.length) {
    const eff = loop.conclusions.filter(c => c.status === "verified_effective");
    const inef = loop.conclusions.filter(c => c.status === "ineffective");
    const pend = loop.conclusions.filter(c => c.status === "pending");
    const renderC = (c) => {
      const ev = c.critic_evidence;
      const evStr = ev ? `${ev.verdict_delta}<br><span class="muted">Critic: ${ev.before.reason || "—"} → ${ev.after.reason || "—"}</span>` : "";
      return `<div class="conclusion-item">
        <div class="c-dim">[${esc(c.dim)}] ${esc(c.finding || "")}</div>
        <div class="c-change">改动：${esc(c.change)}</div>
        ${c.lesson ? `<div class="c-lesson">${esc(c.lesson)}</div>` : ""}
        ${evStr ? `<div class="c-evidence">${evStr}</div>` : ""}
        <div class="muted" style="font-size:11px">提出 r${c.created_round}${c.verified_round ? ` · 验证 r${c.verified_round}` : ""}</div>
      </div>`;
    };
    const group = (title, list, cls) => list.length ? `<div class="c-group"><div class="c-group-h ${cls}">${title}（${list.length}）</div>${list.map(renderC).join("")}</div>` : "";
    html += `<details style="margin-top:20px" open><summary><strong>经验沉淀（${loop.conclusions.length} 条）</strong></summary>
      <div class="card" style="margin-top:10px">
        ${group("✓ 已验证有效", eff, "eff")}
        ${group("✗ 验证无效", inef, "inef")}
        ${group("◯ 待验证", pend, "pend")}
      </div></details>`;
  }
  // 排序入口
  html += `<div class="row" style="margin-top:16px"><a class="btn" href="#/scoring/${esc(loop.bench_id)}/${esc(loop.sample_id)}">给 ${esc(loop.sample_id)} 排序 →</a></div>`;

  app.innerHTML = html;
  // 事件委托：点 trace-col 打开详情 modal；prompt 按钮单独处理（不进 modal）
  const traceRow = document.getElementById("trace-row");
  if (traceRow) {
    traceRow.addEventListener("click", (e) => {
      const t = e.target;
      if (t.dataset.lb) { e.stopPropagation(); openLightbox(t.dataset.lb); return; }
      // prompt 展开/对比按钮（stopPropagation 已阻止冒泡到卡片）
      if (t.dataset.ptog !== undefined) { toggleRowPrompt(+t.dataset.ptog); return; }
      if (t.dataset.pdiff !== undefined) { showRowDiff(+t.dataset.pdiff); return; }
      const col = t.closest(".trace-col");
      if (col && col.dataset.idx != null) openTraceModal(+col.dataset.idx);
    });
  }

  // 运行中/等审批时轮询状态：实时显示当前 agent 节点；状态或轮次真正变化才整页刷新。
  // 注意：error/finished 是终态，不轮询（否则会无限 reload 把滚动位置冲回顶部）。
  if (st === "running" || st === "awaiting_review") {
    _loopPollTimer = setInterval(async () => {
      try {
        const s = await api(`/loops/${loopId}/status`);
        // 终态（error/finished）才整页刷新拉取最终数据；运行中只更新节点文本
        if (s.phase === "error" || s.phase === "finished") {
          clearInterval(_loopPollTimer); _loopPollTimer = null;
          location.reload();
          return;
        }
        // 实时更新当前节点（不刷整页，避免打断观察）
        const nodeEl = document.getElementById("cur-node");
        if (nodeEl && s.current_node) {
          const n = NODE_LABEL[s.current_node] || { name: s.current_node, desc: "" };
          const extra = s.rounds_remaining > 0 ? `（自动连跑剩 ${s.rounds_remaining} 轮）` : "";
          nodeEl.innerHTML = `执行中：<strong>${esc(n.name)}</strong> · ${esc(n.desc)}${extra}`;
        }
      } catch (_) {}
    }, 2000);
  }
}

// 列表内 prompt 展开/折叠
function toggleRowPrompt(idx) {
  const el = document.getElementById(`row-prompt-${idx}`);
  if (!el) return;
  el.hidden = !el.hidden;
}
// 列表内 prompt diff 对比（再次点击还原为原始 prompt）
function showRowDiff(idx) {
  const loop = window.__loopCache.current;
  if (!loop || idx <= 0) return;
  const cur = loop.traces[idx];
  const box = document.getElementById(`row-prompt-${idx}`);
  if (!box) return;
  const btn = document.querySelector(`button[data-pdiff="${idx}"]`);
  // 已在显示 diff → 还原为原始 prompt 并收起
  if (box.dataset.diffing === "1") {
    box.dataset.diffing = "";
    box.innerHTML = `<pre>${esc(cur.prompt || "(空)")}</pre>`;
    box.hidden = true;
    if (btn) btn.textContent = `对比第${loop.traces[idx - 1].round}轮`;
    return;
  }
  // 展开 diff
  const prev = loop.traces[idx - 1];
  box.dataset.diffing = "1";
  box.hidden = false;
  box.innerHTML = `<pre class="diff">${diffPrompt(prev.prompt, cur.prompt)}</pre>`;
  if (btn) btn.textContent = "还原 prompt";
}

async function resumeLoop(loopId, decision) {
  if (decision === "stop" && !confirm("确定停止这个 loop？")) return;
  try {
    await api(`/loops/${loopId}/resume`, { method: "POST", body: JSON.stringify({ decision }) });
    location.reload();
  } catch (e) {
    alert("操作失败: " + e.message);
  }
}

// ============ 屏④ 排序 ============
let _rankState = null;
async function viewScoring(benchId, sampleId) {
  const overview = await api("/overview");
  const bench = overview.benches.find(b => b.bench_id === benchId);
  const sample = bench?.samples.find(s => s.sample_id === sampleId);
  if (!sample) { document.getElementById("app").innerHTML = '<div class="empty">找不到该 sample</div>'; return; }

  // 收集该 sample 下所有 trace（跨 loop）
  const traces = [];
  for (const l of sample.loops) {
    const loop = await api(`/loops/${l.loop_id}`);
    for (const t of loop.traces) traces.push({ ...t, loop_id: l.loop_id, loop_short: l.loop_id.replace(/.*-\d{4}-/, "") });
  }
  // 按 AI 还原度降序作为初始顺序（用户再拖）
  traces.sort((a, b) => (b.verdict?.restoration ?? -1) - (a.verdict?.restoration ?? -1));

  // 已有的人工排序回显
  const saved = await api(`/scoring/${benchId}/${sampleId}/ranks`);
  const savedMap = {};
  for (const r of saved.ranks) savedMap[r.trace_id] = r.rank;

  _rankState = { benchId, sampleId, traces };
  const app = document.getElementById("app");
  app.innerHTML = `
    <div class="row" style="margin-bottom:8px"><a href="#/">← 总览</a></div>
    <h1>人工排序 · ${esc(sampleId)} ${sample.product ? "· " + esc(sample.product) : ""}</h1>
    <div class="muted">拖拽排序（上=好，下=差）。这是校准 Critic 权重的信号。缩略图可点击放大。</div>
    <label><input type="checkbox" id="blind" onchange="toggleBlind()"> 盲排（隐藏 AI 还原度，避免锚定偏差）</label>
    <ul class="rank-list" id="rank-list"></ul>
    <div class="row">
      <button onclick="submitRanks()">提交排序 → 触发自动校准</button>
    </div>
    <div id="calib-result" class="calib-box" style="display:none"></div>`;
  renderRankList();
  loadCalibStatus(benchId, sampleId);
}

function renderRankList() {
  const ul = document.getElementById("rank-list");
  const blind = document.getElementById("blind")?.checked;
  ul.innerHTML = _rankState.traces
    .map((t, i) => {
      const img = t.output_image_refs[0];
      return `<li class="rank-item" draggable="true" data-idx="${i}">
        <span class="grip">⠿</span>
        <span class="rank-num">#${i + 1}</span>
        ${img ? `<img class="rank-thumb" src="${imgURL(img, t.loop_id)}" data-lb="${imgURL(img, t.loop_id)}">` : '<div class="rank-thumb"></div>'}
        <span class="rank-meta">
          <span class="src">${esc(t.loop_short)} · 第${t.round}轮</span>
          ${!blind && t.verdict ? `<br>AI还原度 ${fmt(t.verdict.restoration)}` : ""}
          ${t.human_rank != null ? `<br>历史人工排序: ${t.human_rank}` : ""}
        </span>
      </li>`;
    })
    .join("");
  bindDnD(ul);
  ul.querySelectorAll("img[data-lb]").forEach(img => img.addEventListener("click", () => openLightbox(img.dataset.lb)));
}

function toggleBlind() { renderRankList(); }

function bindDnD(ul) {
  let dragIdx = null;
  ul.querySelectorAll(".rank-item").forEach((li) => {
    li.addEventListener("dragstart", () => { dragIdx = +li.dataset.idx; li.classList.add("dragging"); });
    li.addEventListener("dragend", () => { li.classList.remove("dragging"); ul.querySelectorAll(".drag-over").forEach(x => x.classList.remove("drag-over")); });
    li.addEventListener("dragover", (e) => { e.preventDefault(); li.classList.add("drag-over"); });
    li.addEventListener("dragleave", () => li.classList.remove("drag-over"));
    li.addEventListener("drop", (e) => {
      e.preventDefault();
      const dropIdx = +li.dataset.idx;
      if (dragIdx === null || dragIdx === dropIdx) return;
      const arr = _rankState.traces;
      const [moved] = arr.splice(dragIdx, 1);
      arr.splice(dropIdx, 0, moved);
      dragIdx = null;
      renderRankList();
    });
  });
}

async function submitRanks() {
  const { benchId, sampleId, traces } = _rankState;
  // rank 越大越好：列表上方(rank 1)应得最大 rank
  const n = traces.length;
  const ranks = traces.map((t, i) => ({ trace_id: t.trace_id, rank: n - i }));
  try {
    await api(`/scoring/${benchId}/${sampleId}/ranks`, {
      method: "POST", body: JSON.stringify({ ranks, note: "web 排序" }),
    });
    document.getElementById("calib-result").style.display = "block";
    document.getElementById("calib-result").innerHTML = '<div class="muted">校准中…</div>';
    pollCalib(benchId, sampleId);
  } catch (e) { alert("提交失败: " + e.message); }
}

async function pollCalib(benchId, sampleId) {
  for (let i = 0; i < 20; i++) {
    await new Promise(r => setTimeout(r, 600));
    const s = await api(`/scoring/${benchId}/${sampleId}/calibration`);
    if (s.state === "done" || s.state === "insufficient" || s.state === "error") {
      renderCalib(s);
      return;
    }
  }
  renderCalib(await api(`/scoring/${benchId}/${sampleId}/calibration`));
}

async function loadCalibStatus(benchId, sampleId) {
  const s = await api(`/scoring/${benchId}/${sampleId}/calibration`);
  if (s.state !== "idle") { document.getElementById("calib-result").style.display = "block"; renderCalib(s); }
}

function renderCalib(s) {
  const box = document.getElementById("calib-result");
  if (!box) return;
  if (s.state === "insufficient") { box.innerHTML = `<div class="muted">${esc(s.message || "数据不足")}</div>`; return; }
  if (s.state === "error") { box.innerHTML = `<div class="bad">校准出错</div>`; return; }
  if (s.state === "running") { box.innerHTML = '<div class="muted">校准中…</div>'; return; }
  if (!s.weights) { box.innerHTML = '<div class="muted">无结果</div>'; return; }
  const bars = Object.entries(s.weights)
    .sort((a, b) => b[1] - a[1])
    .map(([dim, w]) => {
      const prior = s.prior_weights?.[dim] ?? 0;
      const delta = w - prior;
      const sign = delta > 0.001 ? "▲" : delta < -0.001 ? "▼" : "·";
      return `<div class="weight-bar"><span style="width:120px">${esc(dim)}</span>
        <span class="bar"><span class="fill" style="width:${(w * 100).toFixed(0)}%"></span></span>
        <span class="v">${fmt(w)} <span class="muted">${sign}</span></span></div>`;
    }).join("");
  box.innerHTML = `<h2>校准结果（${s.state === "done" ? "已完成" : s.state}）</h2>
    <div class="muted">排序吻合度: <strong>${(s.pairwise_accuracy * 100).toFixed(0)}%</strong> · ${s.n_pairs} 对 · ${s.n_traces} trace</div>
    <div style="margin-top:8px">${bars}</div>
    <div class="hint">下一轮 Critic 会自动用上这套权重（sample 级）。<a href="#" onclick="recalib();return false">重新校准</a></div>`;
}

async function recalib() {
  const { benchId, sampleId } = _rankState;
  await api(`/scoring/${benchId}/${sampleId}/calibrate`, { method: "POST" });
  document.getElementById("calib-result").innerHTML = '<div class="muted">重新校准中…</div>';
  pollCalib(benchId, sampleId);
}

// ============ 屏⑤ Agent 设置 ============
let _cfgAgent = "generator";
async function viewConfig() {
  const data = await api("/config");
  const app = document.getElementById("app");
  app.innerHTML = `
    <div class="row" style="margin-bottom:8px"><a href="#/">← 总览</a></div>
    <h1>Agent 设置</h1>
    <div class="muted">只改系统提示词 + 模型 id。改完下个 loop 生效（工具/skill 后续单独接入）。</div>
    <div class="config-tabs" id="cfg-tabs">
      ${["generator", "critic", "summarizer"].map(a => `<button class="${a === _cfgAgent ? "active" : ""}" onclick="switchCfg('${a}')">${a}</button>`).join("")}
    </div>
    <div class="card" id="cfg-form"></div>`;
  renderCfgForm(data.agents);
}

function switchCfg(a) { _cfgAgent = a; viewConfig(); }

function renderCfgForm(agents) {
  const a = agents.find(x => x.agent === _cfgAgent) || agents[0];
  document.getElementById("cfg-form").innerHTML = `
    <label>模型 id</label>
    <input type="text" id="cfg-model" value="${esc(a.model)}" style="width:100%">
    <label>系统提示词</label>
    <textarea id="cfg-prompt">${esc(a.system_prompt)}</textarea>
    <div class="row" style="margin-top:10px">
      <button onclick="saveCfg()">保存</button>
      <button class="secondary" onclick="resetCfg()">恢复默认</button>
      <span id="cfg-msg" class="muted"></span>
    </div>`;
}

async function saveCfg() {
  const model = document.getElementById("cfg-model").value;
  const prompt = document.getElementById("cfg-prompt").value;
  try {
    await api(`/config/${_cfgAgent}`, { method: "POST", body: JSON.stringify({ system_prompt: prompt, model }) });
    document.getElementById("cfg-msg").textContent = "已保存 ✓";
    setTimeout(() => { const m = document.getElementById("cfg-msg"); if (m) m.textContent = ""; }, 2000);
  } catch (e) { alert("保存失败: " + e.message); }
}
async function resetCfg() {
  if (!confirm("恢复代码默认？")) return;
  await api(`/config/${_cfgAgent}/reset`, { method: "POST" });
  viewConfig();
}

// ============ 启动新 loop ============
let _newLoopData = null; // 缓存 overview 供表单用
let _newLoopModels = null; // 缓存 /api/models（可选生图模型）
async function startNewLoop() {
  if (!_newLoopData) _newLoopData = await api("/overview");
  if (!_newLoopModels) {
    try { _newLoopModels = await api("/models"); } catch (_) { _newLoopModels = { image_models: [], agent_models: [] }; }
  }
  if (!_newLoopData.benches.length) { alert("还没有 benchmark，无法启动"); return; }
  const body = document.getElementById("trace-modal-body");
  const firstBench = _newLoopData.benches[0];
  const firstSample = firstBench.samples[0];
  body.innerHTML = renderNewLoopForm(firstBench.bench_id, firstSample?.sample_id || "");
  document.getElementById("trace-modal").classList.remove("hidden");
  onNewLoopBenchChange(); // 初始化时填充 sample 下拉
  updateNewLoopPreview();
}

function renderNewLoopForm(benchId, sampleId) {
  const benches = _newLoopData.benches;
  const models = _newLoopModels || { image_models: [], agent_models: [] };
  // 生图模型下拉：默认项 + .env 已配置的非空模型
  const imageOpts = [
    `<option value="">默认（settings）</option>`,
    ...models.image_models.map(
      m => `<option value="${esc(m.model_id)}">${esc(m.label)} · ${esc(m.model_id)}</option>`
    ),
  ].join("");
  // agent LLM 作为只读参考（由全局 .env 控制，此处仅展示，不随 loop 改动）
  const agentInfo = (models.agent_models || [])
    .map(m => `${esc(m.label)}=<code>${esc(m.model_id)}</code>`).join(" · ") || "（未配置）";
  // 轮数选项：1=单轮跑首停审批；2~8=后台自动连跑
  const roundsOpts = [1, 2, 3, 4, 5, 6, 7, 8]
    .map(n => `<option value="${n}"${n === 4 ? " selected" : ""}>${n} 轮</option>`).join("");
  return `
    <h1>启动新 loop</h1>
    <div class="muted">对一个 sample 用固定模型跑一条自迭代闭环。</div>
    <div class="newloop-form" style="margin-top:14px">
      <label>benchmark</label>
      <select id="nl-bench" onchange="onNewLoopBenchChange()">
        ${benches.map(b => `<option value="${esc(b.bench_id)}">${esc(b.bench_id)}</option>`).join("")}
      </select>
      <label>sample</label>
      <select id="nl-sample" onchange="updateNewLoopPreview()"></select>
      <label>生图模型（决定本 loop 出图，留空用 settings 默认）</label>
      <select id="nl-model" onchange="updateNewLoopPreview()">${imageOpts}</select>
      <label>轮数</label>
      <select id="nl-rounds" onchange="updateNewLoopPreview()">${roundsOpts}</select>
      <div class="muted" style="font-size:11px">选 1 = 首轮跑到等审批就停；选 &gt;1 = 后台自动连跑，无需逐轮审批，跑满后自动结束。</div>
      <label>备注</label>
      <input type="text" id="nl-note" placeholder="可选">
      <div class="muted" style="font-size:11px">Agent LLM（全局，不随 loop 改）：${agentInfo}</div>
      <div class="preview" id="nl-preview"></div>
      <div class="row" style="margin-top:8px">
        <button id="nl-submit" onclick="submitNewLoop()">启动 loop</button>
        <button class="ghost" data-close>取消</button>
      </div>
    </div>`;
}

function onNewLoopBenchChange() {
  const benchId = document.getElementById("nl-bench").value;
  const bench = _newLoopData.benches.find(b => b.bench_id === benchId);
  const sel = document.getElementById("nl-sample");
  sel.innerHTML = (bench?.samples || [])
    .map(s => `<option value="${esc(s.sample_id)}">${esc(s.sample_id)} ${s.product ? "· " + esc(s.product) : ""}</option>`)
    .join("");
  updateNewLoopPreview();
}

function updateNewLoopPreview() {
  const benchId = document.getElementById("nl-bench").value;
  const sampleId = document.getElementById("nl-sample").value;
  const model = document.getElementById("nl-model").value.trim() || "（settings 默认）";
  const rounds = parseInt(document.getElementById("nl-rounds").value, 10) || 1;
  // 一题一 loop：检测该 sample 是否已有 loop
  const bench = _newLoopData.benches.find(b => b.bench_id === benchId);
  const sample = bench?.samples.find(s => s.sample_id === sampleId);
  const existing = sample?.loops?.[0]; // 一题只一条 loop
  let action, detail;
  if (existing) {
    action = `继续 loop <code>${esc(existing.loop_id)}</code>`;
    detail = `已有 ${existing.n_traces} 轮，将继续跑下一轮（在原有 loop 上叠加轮数）。`;
    document.getElementById("nl-submit").textContent = "继续跑下一轮";
  } else {
    action = `新建 loop <code>${esc(benchId)}-${esc(sampleId)}</code>`;
    document.getElementById("nl-submit").textContent = "启动 loop";
  }
  // 轮数说明
  const roundsDesc = rounds > 1
    ? `将自动连跑 <strong>${rounds}</strong> 轮（无需逐轮审批，跑满自动结束）。`
    : `首轮回跑到人工审批节点停下，可在 loop 详情里点「继续/停止」。`;
  const product = sample?.product ? `（${esc(sample.product)}）` : "";
  document.getElementById("nl-preview").innerHTML =
    `将对 <code>${esc(sampleId)}</code>${product} 用 <code>${esc(model)}</code> ${action}<br>` + roundsDesc;
}

async function submitNewLoop() {
  const benchId = document.getElementById("nl-bench").value;
  const sampleId = document.getElementById("nl-sample").value;
  const model = document.getElementById("nl-model").value.trim() || undefined;
  const rounds = parseInt(document.getElementById("nl-rounds").value, 10) || undefined;
  const note = document.getElementById("nl-note").value.trim() || undefined;
  const btn = document.getElementById("nl-submit");
  btn.disabled = true; btn.textContent = "启动中…";
  try {
    const r = await api(`/loops`, {
      method: "POST",
      body: JSON.stringify({ bench_id: benchId, sample_id: sampleId, model, note, rounds }),
    });
    closeTraceModal();
    location.hash = `#/loop/${r.loop_id}`; // 跳转到新 loop 详情
  } catch (e) {
    btn.disabled = false; btn.textContent = "启动 loop";
    alert("启动失败: " + e.message + "\n（可能未配置 .env 的 dmxapi key/model_id）");
  }
}

// ============ 路由 ============
async function router() {
  const h = location.hash.slice(1) || "/";
  closeTraceModal(); closeLightbox();
  if (_loopPollTimer) { clearInterval(_loopPollTimer); _loopPollTimer = null; }
  try {
    if (h === "/" || h === "") await viewOverview();
    else if (h.startsWith("/loop/")) await viewLoop(decodeURIComponent(h.slice(6)));
    else if (h.startsWith("/scoring/")) {
      const [bench, sample] = h.slice(9).split("/");
      await viewScoring(decodeURIComponent(bench), decodeURIComponent(sample));
    } else if (h === "/config") await viewConfig();
    else document.getElementById("app").innerHTML = '<div class="empty">404</div>';
  } catch (e) {
    document.getElementById("app").innerHTML = `<div class="empty">加载失败: ${esc(e.message)}</div>`;
  }
}
window.addEventListener("hashchange", router);
window.addEventListener("load", router);
