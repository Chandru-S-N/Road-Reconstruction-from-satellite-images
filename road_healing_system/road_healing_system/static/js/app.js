/* Road network healing - main controller. Plain JavaScript, no external libraries. */
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
const h = (v) => String(v ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const num = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "n/a" : (typeof v === "number" ? v.toFixed(d) : h(v));
const pct = (v) => (v === null || v === undefined) ? "n/a" : `${Math.round(v * 100)}%`;

const SLIDES = [
  { key: "original", title: "Original satellite image" },
  { key: "mask", title: "Generated road mask" },
  { key: "graph", title: "Road graph" },
  { key: "fragments", title: "Road fragments" },
  { key: "gaps", title: "Road-to-road candidate gaps" },
  { key: "review", title: "Gap review", gap: true },
  { key: "final", title: "Final road network" },
  { key: "compare", title: "Before and after" },
  { key: "map", title: "Map explorer" },
];
const IDX = Object.fromEntries(SLIDES.map((s, i) => [s.key, i]));
const FEATURES = ["A", "O", "W", "G", "N"];
const FCOLORS = { A: "#2f68b8", O: "#3f8a35", W: "#6650c2", G: "#1b8691", N: "#c96a16" };
const COVER = [["road_like", "#22d3ee"], ["vegetation", "#16a34a"], ["shadow", "#7c3aed"], ["water", "#2563eb"],
  ["textured", "#f97316"], ["unexplained", "#a3a3a3"]];
const COVER_LABEL = { en: { road_like: "road-like", vegetation: "vegetation-like", shadow: "shadow-like", water: "water-like",
  textured: "textured", unexplained: "other" }, ta: { road_like: "சாலை போன்றது", vegetation: "தாவரம் போன்றது",
  shadow: "நிழல் போன்றது", water: "நீர் போன்றது", textured: "அமைப்புடையது", unexplained: "பிற" } };
const POLARITY = { en: { supports: "Supports", against: "Against", neutral: "Neutral", uncertain: "Uncertain", context: "Context" },
  ta: { supports: "ஆதரவு", against: "எதிர்", neutral: "நடுநிலை", uncertain: "நிச்சயமற்றது", context: "சூழல்" } };
const REC_COLOR = { recommended: "#15803d", uncertain: "#b45309", not_recommended: "#c62828" };

const S = { busy: false, jobId: null, live: false, status: null, R: null, schema: null, slide: 0, gap: 0, voice: null,
  graphData: null, graphView: null, map: null, mapSel: null, cmpMode: "image", cmpView: "side", poll: null };

// --------------------------------------------------------------------------- //
// basics
// --------------------------------------------------------------------------- //
async function api(path, opts = {}) {
  let res;
  try { res = await fetch(path, opts); } catch { throw new Error("The server could not be reached. Is the app still running?"); }
  let data = null;
  try { data = await res.json(); } catch { /* not JSON */ }
  if (!res.ok) throw new Error((data && data.error) || `Request failed (${res.status}).`);
  return data;
}
let toastTimer;
function toast(msg, err = false) {
  const t = $("#toast"); t.textContent = msg; t.classList.toggle("error", err); t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { t.hidden = true; }, err ? 7000 : 3500);
}
function showErr(el, msg) { el.textContent = msg || ""; el.hidden = !msg; }
const file = (rel) => S.status && S.status.files ? S.status.files[rel] : null;
const L = () => S.voice.lang;
const T = (obj) => obj ? (obj[L()] || obj.en || "") : "";
function fig(rel, caption) {
  const url = file(rel);
  return `<figure class="fig">${url ? `<img src="${h(url)}" alt="${h(caption)}">` : `<div class="ph">${h(caption)}<br>not available</div>`}
    <figcaption>${h(caption)}</figcaption></figure>`;
}
function badge(v, text) { return `<span class="badge ${h(v)}">${h(text || String(v).replace(/_/g, " "))}</span>`; }
function table(cols, rows, opts = {}) {
  if (!rows || !rows.length) return `<p class="empty">${h(opts.empty || "Nothing to show.")}</p>`;
  return `<table><thead><tr>${cols.map(c => `<th>${h(c[1])}</th>`).join("")}</tr></thead><tbody>${rows.map((r, i) =>
    `<tr ${opts.attrs ? opts.attrs(r, i) : ""}>${cols.map(c => `<td>${c[2] ? c[2](r) : h(r[c[0]] ?? "")}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
}
const gaps = () => (S.R && S.R.gaps) || [];
const curGap = () => gaps()[S.gap];
function firstUndecided() { const i = gaps().findIndex(g => g.user_decision === "pending"); return i < 0 ? gaps().length : i; }

// --------------------------------------------------------------------------- //
// voice
// --------------------------------------------------------------------------- //
function setupVoice() {
  S.voice = new VoiceGuide((st) => {
    if (st.caption !== undefined) $("#captionText").textContent = st.caption;
    if (st.status !== undefined) {
      const el = $("#voiceStatus");
      el.textContent = st.warning ? `${st.status} — ${st.warning}` : st.status;
      el.classList.toggle("speaking", !!st.speaking);
      el.classList.toggle("warn", !!st.warning);
    }
  });
  const t = $("#voiceToggle");
  const sync = () => { t.setAttribute("aria-pressed", String(S.voice.enabled)); $("#voiceWaves").style.display = S.voice.enabled ? "" : "none"; };
  t.onclick = () => { S.voice.setEnabled(!S.voice.enabled); sync(); };
  $("#voiceLang").value = S.voice.lang;
  $("#voiceLang").onchange = (e) => { S.voice.setLang(e.target.value); if (S.R) { renderSlide(S.slide); speakSlide(); } };
  $("#voiceAuto").checked = S.voice.auto;
  $("#voiceAuto").onchange = (e) => S.voice.setAuto(e.target.checked);
  $("#voiceReplay").onclick = () => S.voice.replay();
  $("#voiceStop").onclick = () => S.voice.stop();
  sync();
}
function slideVoice(i) {
  const R = S.R || {}, v = R.voice || {}, g = curGap(), key = SLIDES[i].key;
  if (SLIDES[i].gap && !g) return { en: "There are no candidate gaps to review.", ta: "ஆய்வு செய்ய இடைவெளிகள் எதுவும் இல்லை." };
  if (key === "review") {
    if (g.user_decision !== "pending" && g.recon_explanation && g.recon_explanation.result) {
      const ex = g.recon_explanation;
      return { en: [ex.result.en, ...(ex.points || []).map(p => p.en)].join(" "), ta: [ex.result.ta, ...(ex.points || []).map(p => p.ta)].join(" ") };
    }
    return g.xai.voice.review;
  }
  return { original: v.original, mask: v.mask, graph: v.graph, fragments: v.fragments, gaps: v.gaps, final: v.final,
    compare: v.compare, map: v.map }[key];
}
function speakSlide(queue = false) { S.voice.announce(slideVoice(S.slide), { queue }); }

// --------------------------------------------------------------------------- //
// upload, samples, running
// --------------------------------------------------------------------------- //
function showScreen(which) {
  $("#startScreen").hidden = which !== "start";
  $("#progressScreen").hidden = which !== "progress";
  $("#carousel").hidden = which !== "carousel";
}
function bindUpload() {
  const dz = $("#dropzone"), input = $("#imageInput");
  const refresh = () => {
    const f = input.files[0];
    $("#imageName").textContent = f ? `${f.name} (${(f.size / 1024).toFixed(0)} kB)` : "";
    $("#uploadBtn").disabled = !f && !$("#maskInput").files[0];
    showErr($("#uploadError"), "");
  };
  input.onchange = refresh; $("#maskInput").onchange = refresh;
  ["dragenter", "dragover"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach(ev => dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz.addEventListener("drop", e => {
    const f = e.dataTransfer.files[0];
    if (!f) return;
    if (!/\.(png|jpe?g)$/i.test(f.name)) { showErr($("#uploadError"), `${f.name} is not a PNG or JPG file.`); return; }
    const dt = new DataTransfer(); dt.items.add(f); input.files = dt.files; refresh();
  });
  $("#uploadBtn").onclick = async () => {
    const fd = new FormData();
    if (input.files[0]) fd.append("image", input.files[0]);
    if ($("#maskInput").files[0]) fd.append("mask", $("#maskInput").files[0]);
    if ($("#refInput").files[0]) fd.append("reference", $("#refInput").files[0]);
    $("#uploadBtn").disabled = true;
    try { onJob(await api("/api/jobs", { method: "POST", body: fd })); await startRun(); }
    catch (err) { showErr($("#uploadError"), err.message); }
    finally { $("#uploadBtn").disabled = false; }
  };
}
async function loadSamples() {
  try {
    const d = await api("/api/samples");
    $("#samples").innerHTML = d.samples.map(s => `<button type="button" class="sample" data-name="${h(s.name)}" ${s.loadable ? "" : "disabled"} title="${h(s.description)}">
      ${s.thumbnail ? `<img src="${h(s.thumbnail)}" alt="" loading="lazy">` : `<span class="broken">Broken file</span>`}<span>${h(s.name.replace(/_/g, " "))}</span></button>`).join("");
    $$(".sample:not([disabled])").forEach(b => b.onclick = async () => {
      try {
        onJob(await api("/api/jobs/sample", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: b.dataset.name, use_mask: $("#sampleUseMask").checked, use_reference: $("#sampleUseRef").checked }) }));
        await startRun();
      } catch (err) { toast(err.message, true); }
    });
  } catch (err) { $("#samples").innerHTML = `<p class="empty">${h(err.message)}</p>`; }
}
function onJob(payload) {
  S.jobId = payload.status.job_id; S.live = true; S.status = payload.status; S.R = null; S.gap = 0; S.slide = 0;
  S.voice.announce((payload.results.voice || {}).upload || { en: "Image uploaded successfully. The analysis is starting.",
    ta: "படம் வெற்றிகரமாகப் பதிவேற்றப்பட்டது. பகுப்பாய்வு தொடங்குகிறது." });
  setChip();
}
function collectParams() {
  const form = $("#paramForm");
  if (!form.checkValidity()) {
    const bad = $$("input:invalid", form)[0];
    throw new Error(`Check “${bad ? $(`label[for="${bad.id}"]`).textContent : "a parameter"}”: the value is outside its allowed range.`);
  }
  const p = {};
  $$("input, select", form).forEach(el => { p[el.name] = el.type === "checkbox" ? el.checked : el.type === "number" ? Number(el.value) : el.value; });
  return p;
}
async function startRun() {
  let params;
  try { params = collectParams(); showErr($("#paramError"), ""); } catch (err) { showErr($("#paramError"), err.message); openDrawer("drawerParams"); return; }
  const res = await api(`/api/jobs/${S.jobId}/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ params }) });
  (res.notes || []).forEach(n => toast(n));
  S.status = res.status;
  $("#progressName").textContent = S.status.name;
  showErr($("#runError"), "");
  showScreen("progress");
  poll();
}
function renderRunStages(stages) {
  const auto = stages.slice(0, 9);
  $("#runStages").innerHTML = auto.map(s => `<li data-status="${h(s.status)}"><i></i>${h(s.label)}${s.seconds != null ? ` <span class="muted small">${s.seconds.toFixed(2)} s</span>` : ""}</li>`).join("");
  $("#runBar").style.width = `${Math.round(100 * auto.filter(s => s.status === "done").length / auto.length)}%`;
}
function poll() {
  clearTimeout(S.poll);
  const tick = async () => {
    try {
      const st = await api(`/api/jobs/${S.jobId}/status`);
      S.status = st; setChip(); renderRunStages(st.stages);
      if (st.state === "running") { S.poll = setTimeout(tick, 300); return; }
      if (st.state === "error") { showErr($("#runError"), st.error); toast(st.error, true); return; }
      await loadResults(); S.slide = 0; S.gap = Math.min(firstUndecided(), Math.max(gaps().length - 1, 0));
      showScreen("carousel"); buildCarousel(); goTo(0, { queue: true }); loadHistory();
    } catch (err) { toast(err.message, true); }
  };
  tick();
}
async function loadResults() {
  const p = await api(`/api/jobs/${S.jobId}/results`);
  S.status = p.status; S.R = p.results; S.live = p.status.state !== "saved";
  setChip(); renderExports(); $("#log").textContent = (S.R.log || []).join("\n") || "No messages yet.";
  $("#rerunBtn").disabled = !S.live; $("#pdfBtn").disabled = !S.live;
}
function setChip() {
  const st = S.status;
  $("#jobChip").hidden = !st;
  if (!st) return;
  $("#jobName").textContent = st.name;
  const lab = { uploaded: "ready", running: "analysing", review: "reviewing gaps", done: "review complete", error: "failed", saved: "saved result" };
  $("#jobState").textContent = lab[st.state] || st.state;
  $("#jobDot").className = "dot " + ({ running: "running", review: "review", done: "done", saved: "done", error: "error" }[st.state] || "");
}

// --------------------------------------------------------------------------- //
// carousel
// --------------------------------------------------------------------------- //
function buildCarousel() {
  $("#track").innerHTML = SLIDES.map((s, i) => `<section class="slide" id="slide-${i}" aria-label="${h(s.title)}"></section>`).join("");
  $("#steps").innerHTML = SLIDES.map((s, i) => `<li><button type="button" data-i="${i}"><span class="n">${i + 1}</span>${h(s.title)}</button></li>`).join("");
  $$("#steps button").forEach(b => b.onclick = () => {
    const i = Number(b.dataset.i);
    if (SLIDES[i].gap && !gaps().length) { toast("There are no candidate gaps to review."); return; }
    goTo(i);
  });
}
function goTo(i, opts = {}) {
  S.slide = Math.max(0, Math.min(SLIDES.length - 1, i));
  $("#track").style.transform = `translateX(-${S.slide * 100}%)`;
  renderSlide(S.slide);
  $("#slideCounter").textContent = `Slide ${S.slide + 1} of ${SLIDES.length}`;
  $("#slideBar").style.width = `${((S.slide + 1) / SLIDES.length) * 100}%`;
  $$("#steps button").forEach((b, k) => {
    if (k === S.slide) b.setAttribute("aria-current", "step"); else b.removeAttribute("aria-current");
    b.disabled = SLIDES[k].gap && !gaps().length;
  });
  const cur = $(`#steps button[data-i="${S.slide}"]`); if (cur) cur.scrollIntoView({ block: "nearest", inline: "nearest" });
  renderGapNav();
  const rev = SLIDES[S.slide].key === "review";
  $("#prevBtn").disabled = S.slide === 0;
  $("#nextBtn").disabled = S.slide === SLIDES.length - 1;
  $("#prevBtn").textContent = rev && S.gap > 0 ? "Previous gap" : "Previous";
  $("#nextBtn").textContent = rev ? (S.gap < gaps().length - 1 ? "Next gap" : "Final network") : "Next";
  speakSlide(!!opts.queue);
}
function next() {
  if (S.busy) { toast("Wait until the current reconstruction has finished."); return; }
  const key = SLIDES[S.slide].key, g = curGap();
  if (key === "gaps" && !gaps().length) return goTo(IDX.final);
  if (key === "review") {
    if (g && g.user_decision === "pending") toast(`${g.id} is left pending. You can return to it at any time.`);
    if (S.gap < gaps().length - 1) { S.gap += 1; return goTo(IDX.review); }
    return goTo(IDX.final);
  }
  goTo(S.slide + 1);
}
function prev() {
  if (S.busy) { toast("Wait until the current reconstruction has finished."); return; }
  const key = SLIDES[S.slide].key;
  if (key === "review" && S.gap > 0) { S.gap -= 1; return goTo(IDX.review); }
  if (key === "final") { if (gaps().length) { S.gap = gaps().length - 1; return goTo(IDX.review); } return goTo(IDX.gaps); }
  goTo(S.slide - 1);
}
function renderGapNav() {
  const nav = $("#gapNav");
  nav.hidden = !SLIDES[S.slide].gap || !gaps().length;
  if (nav.hidden) return;
  nav.innerHTML = `<span>Gap ${S.gap + 1} of ${gaps().length}</span>` + gaps().map((g, i) =>
    `<button type="button" data-g="${i}" class="${h(g.user_decision)}" ${i === S.gap ? 'aria-current="true"' : ""} ${S.busy ? "disabled" : ""}
      title="${h(g.recommendation.replace(/_/g, " "))} · ${h(g.user_decision)}">${h(g.id)}</button>`).join("");
  $$("button", nav).forEach(b => b.onclick = () => openGap(Number(b.dataset.g)));
}
function openGap(i) {
  if (S.busy) { toast("Wait until the current reconstruction has finished."); return; }
  const g = curGap();
  if (SLIDES[S.slide].key === "review" && g && g.user_decision === "pending" && i !== S.gap) toast(`${g.id} is left pending.`);
  S.gap = i; goTo(IDX.review);
}
function openZoom(url, title) {
  if (!url) return;
  $("#zoomTitle").textContent = title;
  $("#zoombox").hidden = false;
  const area = $("#zoomArea");
  area.innerHTML = "<div></div>";
  const img = new Image();
  img.onload = () => { const v = new ZoomViewer(area.firstChild, img.naturalWidth, img.naturalHeight, { label: "Scroll to zoom · drag to pan" }); v.setImage("img", url); };
  img.src = url;
}

// --------------------------------------------------------------------------- //
// slides
// --------------------------------------------------------------------------- //
function renderSlide(i) {
  const el = $(`#slide-${i}`);
  if (!el || !S.R) return;
  const fn = RENDER[SLIDES[i].key];
  const g = curGap();
  const sub = "";
  el.innerHTML = `<div class="slide-title"><h2>${i + 1}. ${h(SLIDES[i].title)}</h2>${sub}</div><div class="slide-body" id="body-${i}"></div>`;
  const body = $(`#body-${i}`);
  if (SLIDES[i].gap && !g) { body.innerHTML = `<div class="panel"><p class="empty">No candidate gaps were identified, so there is nothing to review.</p></div>`; return; }
  try { fn(body, g); } catch (err) { body.innerHTML = `<div class="panel"><p class="form-error">This view could not be drawn: ${h(err.message)}</p></div>`; }
}

const RENDER = {
  original(body) {
    const inp = S.R.input || {};
    body.className = "slide-body cols-main";
    body.innerHTML = fig("stages/original.png", "Original satellite/aerial image (kept unchanged)") +
      `<div class="panel"><h3>Input</h3><dl class="kv"><dt>File</dt><dd>${h(inp.name)}</dd><dt>Size</dt><dd>${h(inp.width)} × ${h(inp.height)} px</dd>
      <dt>Format</dt><dd>${h(inp.format || "-")}</dd><dt>Road mask</dt><dd>${inp.has_mask ? "uploaded" : "extracted from the image"}</dd>
      <dt>Reference</dt><dd>${inp.has_reference ? "uploaded (accuracy metrics enabled)" : "none"}</dd></dl>
      <h3 style="margin-top:12px">Image processing</h3><p class="small">${h(((S.R.preprocess || {}).steps || []).join(", "))}, applied to a copy.</p>
      ${(inp.notes || []).map(n => `<p class="note">${h(n)}</p>`).join("")}</div>`;
  },
  mask(body) {
    const seg = S.R.segmentation || {};
    body.style.gridTemplateRows = "minmax(0,1fr) auto";
    body.innerHTML = `<div class="cols-2" style="display:grid;gap:12px;min-height:0" id="maskFigs">${fig("stages/original.png", "Original image")}
      ${fig("stages/road_mask.png", "Binary road mask: roads white, background black")}</div>
      <div class="panel" style="overflow:visible"><div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">
        <span class="segmented"><button type="button" aria-pressed="true" data-m="side">Side by side</button><button type="button" aria-pressed="false" data-m="steps">How it was extracted</button></span>
        <span class="small"><b>${pct(seg.road_fraction)}</b> road · method <b>${h(seg.method || seg.mask_source)}</b>
        ${seg.excluded_vegetation_pct != null ? ` · excluded: ${h(seg.excluded_vegetation_pct)}% vegetation-like, ${h(seg.excluded_water_pct)}% water-like, ${h(seg.excluded_shadow_pct)}% very dark · ${h(seg.removed_wide_pct)}% removed as wider than a road · ${h(seg.removed_compact_components)} compact shapes removed` : ""}</span></div>
        <p class="note">No trained road-segmentation model is included. The contextual method combines a learned road colour model, line-likeness and width/shape filters; it can still misjudge long grey roofs, concrete channels or dirt tracks. Upload a road mask from a better source if you have one.</p></div>`;
    $$(".segmented button", body).forEach(b => b.onclick = () => {
      $$(".segmented button", body).forEach(x => x.setAttribute("aria-pressed", String(x === b)));
      const box = $("#maskFigs");
      if (b.dataset.m === "side") {
        box.className = "cols-2"; box.innerHTML = fig("stages/original.png", "Original image") + fig("stages/road_mask.png", "Binary road mask: roads white, background black");
      } else {
        box.className = "grid-2x2"; box.innerHTML = fig("stages/seg_candidates.png", "1. Road-coloured candidates") + fig("stages/seg_likelihood.png", "2. Road colour likelihood") +
          fig("stages/seg_line_likeness.png", "3. Line-likeness (ridge filter)") + fig("stages/seg_removed.png", "4. Removed: wide or compact regions");
      }
    });
  },
  graph(body) {
    body.className = "slide-body cols-wide";
    body.innerHTML = `<div id="graphBox"></div><div class="panel" id="graphInfo"><p class="empty">Loading the road graph…</p></div>`;
    const [w, hgt] = S.R.image_size;
    S.graphView = new RoadGraphView($("#graphBox"), w, hgt, file("stages/original.png"), (sel) => graphInfo(sel));
    api(`/api/jobs/${S.jobId}/graph`).then(d => { S.graphData = d; S.graphView.render(d); graphInfo(null); })
      .catch(err => { $("#graphInfo").innerHTML = `<p class="form-error">${h(err.message)}</p>`; });
  },
  fragments(body) {
    body.className = "slide-body cols-main";
    const fr = S.R.fragments || [];
    body.innerHTML = fig("stages/fragments.png", "Connected road components; open ends yellow, image-border ends grey") +
      `<div class="panel"><h3>${fr.filter(f => f.is_fragment).length} fragments with open ends (of ${fr.length} components)</h3>
      <p class="small muted">A fragment is a connected road piece with at least one end inside the image, where the road may be interrupted. Ends at the image border are roads leaving the image.</p>
      <div class="table-wrap">${table([["component", "Fragment", f => `F${f.component}`], ["length_px", "Length (px)"], ["open", "Open ends", f => h(f.open_ends.join(", ") || "–")],
        ["border", "Border ends", f => h(f.border_ends.join(", ") || "–")], ["junctions", "Junctions"]], fr)}</div></div>`;
  },
  gaps(body) {
    body.className = "slide-body cols-main";
    const d = S.R.gap_detection || {}, list = gaps(), s = S.R.summary || {};
    body.innerHTML = fig("stages/recommendations.png", "Candidate gaps: green recommended, amber uncertain, red not recommended") +
      `<div class="panel"><div class="tiles"><div class="tile"><div class="v">${list.length}</div><div class="k">genuine gaps</div></div>
      <div class="tile good"><div class="v">${s.recommended ?? 0}</div><div class="k">recommended</div></div><div class="tile warn"><div class="v">${s.uncertain ?? 0}</div><div class="k">uncertain</div></div>
      <div class="tile bad"><div class="v">${s.not_recommended ?? 0}</div><div class="k">not recommended</div></div></div>
      <p class="small muted">Only road-to-road disconnections reach this list: two ends of separate fragments that are both verified as road in the image, face each other, can be joined without an S-bend, have compatible widths, have no road or water between them, are each other's best match, and admit a continuous connection. Gaps around possible occlusions are listed first. ${(d.excluded || []).length} other endpoint pairs were excluded.</p>
      <div class="table-wrap">${table([["id", "Gap", g => `<b>${h(g.id)}</b>`], ["priority", "Region", g => h(g.priority)], ["distance", "Length (px)", g => num(g.distance, 0)], ["cars", "CARS", g => num(g.cars)],
        ["rec", "Recommendation", g => badge(g.recommendation)], ["dec", "Decision", g => badge(g.user_decision)]], list,
        { empty: "No genuine road gaps were identified.", attrs: (g, i) => `class="clickable" data-open="${i}"` })}</div>
      <details style="margin-top:8px"><summary class="small">Excluded endpoint pairs and why</summary>${table([["source", "End"], ["target", "End"], ["distance", "px"], ["reason", "Reason"]], d.excluded || [], { empty: "None." })}</details>
      ${list.length ? `<button class="btn btn-primary" style="margin-top:10px" id="startReview">Review ${h(gaps()[Math.min(firstUndecided(), list.length - 1)].id)}</button>` : ""}</div>`;
    $$("[data-open]", body).forEach(tr => tr.onclick = () => openGap(Number(tr.dataset.open)));
    const b = $("#startReview"); if (b) b.onclick = () => openGap(Math.min(firstUndecided(), list.length - 1));
  },
  review(body, g) {
    body.className = "slide-body";
    body.style.cssText = "display:flex;flex-direction:column;gap:8px;min-height:0";
    const lang = L(), x = g.xai, f = g.features, fi = g.fragments_info || {}, A = fi.A || {}, B = fi.B || {};
    const done = g.recon_status === "reconstructed";
    const decided = gaps().filter(q => q.user_decision !== "pending").length;
    const img = (n) => file(`gaps/${g.id}/${n}.png`);
    const view = (key, letter, title, n, extra = "") => `<figure class="view" data-key="${key}"><img src="${h(img(n) || "")}" alt="${h(title)}" data-title="${h(letter + ". " + title)}">
      <figcaption><b>${letter}</b> ${h(title)}${extra}</figcaption></figure>`;
    const frag = (k, F) => F.component === undefined ? "n/a" : `F${F.component}: ${num(F.length_px, 0)} px long, ${num(F.width_px, 0)} px wide; road likelihood ${num(F.road_likelihood)}${F.leaves_image ? "; continues beyond the image" : ""}; end ${F.end} heading ${num(F.direction_deg, 0)}°`;
    const covers = (prefix) => `<div class="covers">${COVER.map(([k, c]) => `<i style="width:${100 * (f[prefix + k] || 0)}%;background:${c}" title="${COVER_LABEL[lang][k]} ${pct(f[prefix + k] || 0)}"></i>`).join("")}</div>`;
    const top = (prefix) => COVER.map(([k]) => [k, f[prefix + k] || 0]).filter(([, v]) => v >= 0.1).sort((a, b) => b[1] - a[1]).map(([k, v]) => `${pct(v)} ${COVER_LABEL[lang][k]}`).join(", ") || "n/a";
    const thr = S.R.decisions.threshold;
    const canApprove = g.recommendation !== "not_recommended" && !done && S.live;
    let status;
    if (g.user_decision === "pending") status = `<div class="msg-info">No decision yet. Inspect views A–E, then approve or reject. Leaving this gap keeps it pending.</div>`;
    else if (done) status = `<div class="msg-ok">Approved and reconstructed. The road mask and road graph were updated: road components ${g.recon_metrics.components_before?.mask} → ${g.recon_metrics.components_after?.mask}.</div>`;
    else if (g.user_decision === "approved") status = `<div class="msg-bad">Approved, but not reconstructed: ${h(T(g.recon_explanation.result) || g.recon_reason)}</div>`;
    else status = `<div class="msg-info">${badge(g.user_decision)} at ${h(g.decided_at)}. Not reconstructed; the road network is unchanged.</div>`;
    const reconPts = done ? `<h3 style="margin-top:10px">How the connection was reconstructed</h3><ul class="points">${(g.recon_explanation.points || []).map(p => `<li><span>${h(p[lang])}</span></li>`).join("")}</ul>` : "";
    body.innerHTML = `<div class="review-head"><span class="gid">${h(g.id)}</span><span>candidate ${S.gap + 1} of ${gaps().length}</span>
      ${badge(g.recommendation)} ${badge(g.user_decision)} ${done ? badge("reconstructed") : ""} <span class="small muted">region: ${h(g.priority)}</span>
      <span class="review-progress">${decided} of ${gaps().length} decided <span class="bar"><span class="bar-fill" style="display:block;width:${100 * decided / Math.max(gaps().length, 1)}%"></span></span></span></div>
      <div class="review">
        <div class="views">
          ${view("A", "A", "Original image: fragments, ends, gap, occlusion", "original_view")}
          ${view("B", "B", "Enlarged gap with directions and distance", "enlarged")}
          ${view("C", "C", done ? "Road mask (updated, new segment green)" : "Road mask: fragment A cyan, B magenta", done ? "mask_after" : "mask_view")}
          ${view("D", "D", done ? "Local road graph (reconstructed edge green)" : "Local road graph (proposed edge dashed)", done ? "graph_after" : "graph_view")}
          ${view("E", "E", done ? "Reconstructed connection" : "Proposed path (preview only)", done ? "result" : "preview",
            `<span class="swap"><button type="button" data-swap="gap" aria-pressed="false">Gap</button><button type="button" data-swap="prop" aria-pressed="true">${done ? "Result" : "Proposed"}</button></span>`)}
          ${view("F", "·", "Measured cover inside and around the gap", "evidence")}
        </div>
        <div class="analysis"><div class="analysis-scroll">
          <table class="facts"><tbody>
            <tr><th>Gap ID</th><td><b>${h(g.id)}</b> · ${h(g.kind.replace(/_/g, " "))}</td></tr>
            <tr><th><span class="fa">Fragment A</span></th><td>${h(frag("A", A))}</td></tr>
            <tr><th><span class="fb">Fragment B</span></th><td>${h(frag("B", B))}</td></tr>
            <tr><th>Endpoint distance</th><td>${num(g.distance, 1)} px</td></tr>
            <tr><th>Direction difference</th><td>ends ${num(f.end_deviation_deg, 0)}° off the gap line · bend mismatch ${num(f.bend_inconsistency_deg, 0)}° · total turn ${num(f.turn_angle_deg, 0)}° (A = ${num(g.scores.A)})</td></tr>
            <tr><th>Width consistency</th><td>${num(f.width_source, 1)} vs ${num(f.width_target, 1)} px (W = ${num(g.scores.W)})</td></tr>
            <tr><th>Occlusion evidence</th><td>${f.corridor_pixels ? covers("frac_") : ""}${h(top("frac_"))} (O = ${num(g.scores.O)})</td></tr>
            <tr><th>Environment</th><td>${f.env_dominant ? covers("env_") : ""}${h(top("env_"))}</td></tr>
            <tr><th>Graph connectivity</th><td>joins components ${h(f.component_source)} and ${h(f.component_target)}${f.competing_candidates ? ` · ${f.competing_candidates} competing gap(s)` : ""} (N = ${num(g.scores.N)})</td></tr>
            <tr><th>CARS</th><td><b>${num(g.cars, 3)}</b> (threshold ${num(thr)})</td></tr>
            <tr><th>Recommendation</th><td><span class="headline" style="font-size:14px;margin:0"><span class="dotbig" style="background:${REC_COLOR[g.recommendation]}"></span>${h(T(x.headline))}</span></td></tr>
          </tbody></table>
          <div class="cover-legend">${COVER.map(([k, c]) => `<span><i style="background:${c}"></i>${h(COVER_LABEL[lang][k])}</span>`).join("")}</div>
          <h3 style="margin-top:8px">Why</h3>
          <ul class="points">${x.points.map(p => `<li><span class="pol ${h(p.polarity)}">${h(POLARITY[lang][p.polarity] || p.polarity)}</span><span>${h(p[lang])}</span></li>`).join("")}</ul>
          <p><b>${h(T(x.meaning))}</b> ${h(T(x.conclusion))}</p><p class="small muted">${h(T(x.influence_text))}</p>
          <details><summary class="small">CARS calculation</summary><div class="formula">${h(x.numeric.formula)}</div>
          ${table([["feature", "Feature", r => `<b style="color:${FCOLORS[r.feature]}">${r.feature}</b> ${h(S.R.feature_names[r.feature])}${r.missing ? " (missing)" : ""}`],
            ["value", "Score", r => num(r.value)], ["weight", "Weight", r => num(r.weight)], ["contribution", "Contribution", r => num(r.contribution, 3)]], x.numeric.rows)}</details>
          ${reconPts}
        </div>
        <div class="analysis-actions" id="actions">${status}
          ${!S.live ? `<p class="note">Saved result: decisions are only possible for an analysis run in this session.</p>` : ""}
          <div class="row"><button class="btn btn-good" id="approveBtn" ${canApprove ? "" : "disabled"}>Approve connection</button>
          <button class="btn btn-bad" id="rejectBtn" ${done || !S.live ? "disabled" : ""}>Reject connection</button></div>
          ${g.recommendation === "not_recommended" && !done ? `<p class="small muted" style="margin:0">Approval is disabled: ${h(T(x.conclusion))}</p>` : ""}
        </div></div>
      </div>`;
    $$(".view img", body).forEach(im => im.onclick = () => openZoom(im.getAttribute("src"), im.dataset.title));
    const ev = $('.view[data-key="E"]', body);
    $$("[data-swap]", ev).forEach(b => b.onclick = () => {
      $$("[data-swap]", ev).forEach(x2 => x2.setAttribute("aria-pressed", String(x2 === b)));
      $("img", ev).src = b.dataset.swap === "gap" ? img("enlarged") : img(done ? "result" : "preview");
    });
    $("#approveBtn").onclick = () => decide(g, "approve");
    $("#rejectBtn").onclick = () => decide(g, "reject");
  },
  final(body) {
    const s = S.R.summary || {};
    body.style.gridTemplateRows = "auto minmax(0,1fr) auto";
    body.innerHTML = `<div class="tiles"><div class="tile"><div class="v">${s.total ?? 0}</div><div class="k">candidate gaps</div></div>
      <div class="tile good"><div class="v">${s.approved ?? 0}</div><div class="k">approved</div></div><div class="tile bad"><div class="v">${s.rejected ?? 0}</div><div class="k">rejected</div></div>
      <div class="tile warn"><div class="v">${s.pending ?? 0}</div><div class="k">pending</div></div><div class="tile good"><div class="v">${s.reconstructed ?? 0}</div><div class="k">reconstructed</div></div>
      <div class="tile ${s.failed ? "bad" : ""}"><div class="v">${s.failed ?? 0}</div><div class="k">failed / blocked</div></div>
      <div class="tile"><div class="v">${s.components_before ?? "–"} → ${s.components_after ?? "–"}</div><div class="k">road components</div></div></div>
      <div class="grid-3x2">${fig("stages/original.png", "Original image")}${fig("masks/original_road_mask.png", "Original road mask")}${fig("stages/decisions.png", "All gaps: green reconstructed, red rejected, orange failed, grey pending")}
      ${fig("stages/current_mask.png", "Final road mask (new segments green)")}${fig("stages/final_overlay.png", "Final road network")}${fig("stages/reconstructed_only.png", "Reconstructed segments only")}</div>
      <div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">${legend()}
      ${!s.complete ? `<span class="badge uncertain">${s.pending} gap(s) still pending</span>` : ""}
      <button class="btn btn-primary" id="pdfInline" ${S.live ? "" : "disabled"} style="margin-left:auto">Generate PDF report</button></div>`;
    $("#pdfInline").onclick = makePdf;
  },
  compare(body) {
    const rec = gaps().filter(g => g.recon_status === "reconstructed");
    body.style.gridTemplateRows = "auto minmax(0,1fr)";
    body.innerHTML = `<div style="display:flex;gap:10px;align-items:center;flex-wrap:wrap">
      <span class="segmented"><button type="button" data-mode="image" aria-pressed="${S.cmpMode === "image"}">Images</button><button type="button" data-mode="mask" aria-pressed="${S.cmpMode === "mask"}">Masks</button></span>
      <span class="segmented"><button type="button" data-view="side" aria-pressed="${S.cmpView === "side"}">Side by side (synced)</button><button type="button" data-view="slider" aria-pressed="${S.cmpView === "slider"}">Slider</button></span>
      <span class="small"><b>${rec.length}</b> reconstructed connection(s):</span>${rec.map(g => `<button class="btn btn-quiet btn-small" data-zoom="${h(g.id)}">${h(g.id)}</button>`).join("") || '<span class="small muted">none yet</span>'}
      <button class="btn btn-quiet btn-small" data-zoom="">Whole image</button></div>
      <div id="cmpBox" class="cols-2" style="display:grid;gap:10px;min-height:0"></div>`;
    $$("[data-mode]", body).forEach(b => b.onclick = () => { S.cmpMode = b.dataset.mode; renderSlide(S.slide); });
    $$("[data-view]", body).forEach(b => b.onclick = () => { S.cmpView = b.dataset.view; renderSlide(S.slide); });
    const before = S.cmpMode === "image" ? "stages/before_overlay.png" : "masks/original_road_mask.png";
    const after = S.cmpMode === "image" ? "stages/final_overlay.png" : "stages/current_mask.png";
    const box = $("#cmpBox"), [w, hgt] = S.R.image_size;
    if (S.cmpView === "slider") {
      box.className = ""; box.style.display = "flex"; box.style.minHeight = "0";
      box.innerHTML = `<div class="swipe" style="flex:1"><img src="${h(file(after))}" alt="After"><img class="top" src="${h(file(before))}" alt="Before"><input type="range" min="0" max="100" value="50" aria-label="Reveal original"></div>`;
      $("input", box).oninput = (e) => { $(".top", box).style.clipPath = `inset(0 ${100 - e.target.value}% 0 0)`; };
      $$("[data-zoom]", body).forEach(b => b.disabled = true);
      return;
    }
    box.innerHTML = `<div id="cmpA"></div><div id="cmpB"></div>`;
    const a = new ZoomViewer($("#cmpA"), w, hgt, { label: S.cmpMode === "image" ? "Original roads" : "Original road mask" });
    const b = new ZoomViewer($("#cmpB"), w, hgt, { label: S.cmpMode === "image" ? "Final network (new in green)" : "Final road mask (new in green)" });
    a.setImage("img", file(before)); b.setImage("img", file(after)); a.link(b);
    $$("[data-zoom]", body).forEach(btn => btn.onclick = () => {
      const id = btn.dataset.zoom;
      if (!id) { a.reset(); return; }
      a.zoomTo(S.R.regions[id], 0.1);
    });
  },
  map(body) {
    body.className = "slide-body cols-wide";
    const [w, hgt] = S.R.image_size;
    body.innerHTML = `<div id="mapBox"></div><div class="panel"><h3>Layers</h3><div class="layers" id="layerList"></div>${legend()}
      <h3 style="margin-top:12px">Selected gap</h3><div id="mapSel"><p class="empty">Click a gap marker on the map.</p></div>
      <p class="small muted">Coordinates are image pixels; the input carries no geographic coordinates.</p></div>`;
    const v = new ZoomViewer($("#mapBox"), w, hgt, { label: "Scroll to zoom · drag to pan · click a gap" });
    S.map = v;
    const layers = [
      ["image", "Original image", "img", "stages/original.png", true, "#94a3b8"],
      ["mask", "Binary road mask", "img", "layers/current_mask.png", false, "#ffffff"],
      ["roads", "Existing roads", "img", "layers/roads.png", true, "#4287f5"],
      ["environment", "Environment cues (heuristic)", "img", "layers/environment.png", false, "#16a34a"],
      ["recon", "Reconstructed paths", "img", "layers/reconstructed.png", true, "#22c55e"],
      ["gaps", "Candidate gaps (pending)", "vec", null, true, "#facc15"],
      ["approved", "Approved connections", "vec", null, true, "#22c55e"],
      ["rejected", "Rejected connections", "vec", null, true, "#ef4444"],
    ];
    const mk = (list, color, dash) => list.map(g => {
      const [x1, y1] = g.source_xy, [x2, y2] = g.target_xy, mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
      const line = g.path ? `<polyline points="${g.path.map(p => p.join(",")).join(" ")}" fill="none" stroke="#fff" stroke-width="1.5" vector-effect="non-scaling-stroke"/>` :
        `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${color}" stroke-width="3" ${dash ? 'stroke-dasharray="6 4"' : ""} vector-effect="non-scaling-stroke"/>`;
      return `<g class="gapmark clickable-mark" data-gap="${h(g.id)}">${line}<circle cx="${mx}" cy="${my}" data-r="9" r="9" fill="${color}" stroke="#111" stroke-width="1.5" vector-effect="non-scaling-stroke"/>
        <text x="${mx}" y="${my}" data-fs="10" text-anchor="middle" dominant-baseline="central" font-weight="700" fill="#111" pointer-events="none">${h(g.id.slice(1))}</text><title>${h(g.id)}</title></g>`;
    }).join("");
    for (const [id, , kind, rel, on] of layers) {
      if (kind === "img") v.setImage(id, file(rel), { visible: on, pixelated: id !== "image" });
    }
    v.setVector("gaps", mk(gaps().filter(g => g.user_decision === "pending" || g.user_decision === "skipped"), "#facc15", true));
    v.setVector("approved", mk(gaps().filter(g => g.user_decision === "approved"), "#22c55e", false));
    v.setVector("rejected", mk(gaps().filter(g => g.user_decision === "rejected"), "#ef4444", true));
    $("#layerList").innerHTML = layers.map(([id, name, kind, rel, on, c]) => `<label><input type="checkbox" data-layer="${id}" ${on ? "checked" : ""} ${kind === "img" && !file(rel) ? "disabled" : ""}>
      <i style="background:${c}"></i>${h(name)}${kind === "img" && !file(rel) ? " (not available)" : ""}</label>`).join("");
    $$("[data-layer]", body).forEach(cb => cb.onchange = () => v.setVisible(cb.dataset.layer, cb.checked));
    $$("[data-gap]", $("#mapBox")).forEach(el => el.addEventListener("click", () => mapSelect(el.dataset.gap)));
    if (S.mapSel) mapSelect(S.mapSel);
  },
};

function legend() {
  const items = [["Existing road", "#4287f5", ""], ["Reconstructed path", "#22c55e", ""], ["Approved connection", "#22c55e", "dash"],
    ["Rejected candidate", "#ef4444", "dash"], ["Pending candidate", "#facc15", "dash"]];
  return `<ul class="legend">${items.map(([n, c, d]) => `<li><i class="${d}" style="--c:${c};background:${c}"></i>${h(n)}</li>`).join("")}</ul>`;
}
function mapSelect(id) {
  S.mapSel = id;
  const i = gaps().findIndex(g => g.id === id), g = gaps()[i];
  if (!g) return;
  S.map.zoomTo(S.R.regions[id], 0.15);
  $("#mapSel").innerHTML = `${fig(`gaps/${g.id}/enlarged.png`, `${g.id} enlarged`)}<p><b>${h(g.id)}</b> · CARS ${num(g.cars)} · ${badge(g.recommendation)} ${badge(g.user_decision)} ${g.recon_status === "reconstructed" ? badge("reconstructed") : ""}</p>
    <p class="small">${h(T(g.xai.headline))}. ${h(T(g.xai.conclusion))}</p><button class="btn btn-primary btn-small" id="openFromMap">Open its analysis</button>`;
  $("#mapSel .fig").style.height = "200px";
  $("#openFromMap").onclick = () => openGap(i);
}
function graphInfo(sel) {
  const d = S.graphData; if (!d) return;
  const st = d.stats, b = st.before, a = st.after;
  let selHtml = `<p class="empty">Click a node or an edge.</p>`;
  if (sel && sel.item) {
    const it = sel.item;
    if (sel.type === "node") {
      selHtml = `<dl class="kv"><dt>Node</dt><dd>${it.id} (${h(it.type)}${it.at_border ? ", at image border" : ""})</dd><dt>Position</dt><dd>x ${it.x}, y ${it.y} px</dd>
        <dt>Neighbours</dt><dd>${h(it.neighbors.join(", ") || "none")}</dd><dt>Component</dt><dd>${it.component}</dd><dt>Road width</dt><dd>${num(it.width, 1)} px</dd></dl>`;
    } else {
      selHtml = `<dl class="kv"><dt>Edge</dt><dd>${h(it.id)} (${h(it.kind)})</dd><dt>Connects</dt><dd>nodes ${it.source} and ${it.target}</dd>
        <dt>Length (weight)</dt><dd>${num(it.length, 1)} px</dd><dt>Width</dt><dd>${num(it.width, 1)} px</dd><dt>Component</dt><dd>${it.component}</dd>
        ${it.gap_id ? `<dt>Gap</dt><dd>${h(it.gap_id)} · CARS ${num(it.cars)} · path cost ${num(it.path_cost, 1)}</dd>` : ""}</dl>
        <p class="small muted">The view zoomed to this edge; tick "show image" to see its image region.</p>`;
    }
  }
  $("#graphInfo").innerHTML = `<h3>Network statistics</h3>${table([["k", ""], ["b", "Original"], ["a", "Now"]], [
      { k: "Nodes", b: b.nodes, a: a.nodes }, { k: "Edges", b: b.edges, a: a.edges }, { k: "Connected components", b: b.components, a: a.components },
      { k: "Fragments with open ends", b: b.fragments, a: a.fragments }, { k: "Intersections", b: b.junctions, a: a.junctions },
      { k: "Reconstructed edges", b: 0, a: (a.edges_by_kind || {}).reconstructed || 0 }])}
    <label class="small" style="display:block;margin:8px 0"><input type="checkbox" id="graphImg" checked> show image under the graph</label>
    <h3>Selection</h3>${selHtml}
    <h3 style="margin-top:10px">Components</h3><div class="table-wrap">${table([["id", "", c => `<i style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${COMPONENT_COLORS[c.id % COMPONENT_COLORS.length]}"></i> ${c.id}`],
      ["nodes", "Nodes"], ["edges", "Edges"], ["length", "Length"], ["reconstructed_edges", "New"]], d.components, { attrs: c => `class="clickable" data-comp="${c.id}"` })}</div>
    <button class="btn btn-quiet btn-small" id="allComps" style="margin-top:6px">Show all components</button>`;
  $("#graphImg").onchange = (e) => S.graphView.setImageVisible(e.target.checked);
  $$("[data-comp]").forEach(tr => tr.onclick = () => S.graphView.focus(Number(tr.dataset.comp)));
  $("#allComps").onclick = () => S.graphView.focus(null);
}

// --------------------------------------------------------------------------- //
// decisions
// --------------------------------------------------------------------------- //
async function decide(g, action) {
  if (S.busy) return;
  S.busy = true;
  if (action === "approve") S.voice.announce(g.xai.voice.reconstruct);
  $$("#actions button").forEach(b => b.disabled = true);
  $("#actions .msg-info, #actions .msg-ok, #actions .msg-bad") && ($("#actions").firstElementChild.outerHTML =
    `<div class="msg-info">${action === "approve" ? "Reconstructing this connection with the graph-based trellis method…" : "Recording the rejection…"}</div>`);
  renderGapNav();
  try {
    const res = await api(`/api/jobs/${S.jobId}/gaps/${g.id}/decision`, { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action }) });
    S.status = res.status; S.R = res.results; setChip(); renderExports(); $("#log").textContent = S.R.log.join("\n");
    S.busy = false;
    renderSlide(S.slide); renderGapNav();
    const ng = curGap();
    if (action === "approve") {
      if (ng.recon_status === "reconstructed") toast(`${ng.id} reconstructed successfully. Use Next to continue.`);
      else toast(`${ng.id} was not reconstructed: ${ng.recon_reason}`, true);
      speakSlide(true);          // the result is spoken only after the server confirmed it
    } else {
      S.voice.announce(ng.recon_explanation.result);
      toast(`${ng.id} rejected. Use Next to continue.`);
    }
  } catch (err) {
    S.busy = false;
    toast(err.message, true);
    renderSlide(S.slide); renderGapNav();
  }
}

// --------------------------------------------------------------------------- //
// drawers: params, reports, history, log
// --------------------------------------------------------------------------- //
function openDrawer(id) { $$(".drawer").forEach(d => d.hidden = d.id !== id); $("#scrim").hidden = false; }
function closeDrawers() { $$(".drawer").forEach(d => d.hidden = true); $("#scrim").hidden = true; }
async function loadSchema() {
  S.schema = (await api("/api/config")).groups;
  buildParams();
}
function buildParams() {
  $("#paramForm").innerHTML = S.schema.map(g => `<fieldset><legend>${h(g.group)}</legend>${g.items.map(p => {
    const id = `p-${p.key.replace(".", "-")}`;
    let c;
    if (p.kind === "select") c = `<select id="${id}" name="${p.key}">${p.extra.map(o => `<option ${o === p.default ? "selected" : ""}>${h(o)}</option>`).join("")}</select>`;
    else if (p.kind === "bool") c = `<input type="checkbox" id="${id}" name="${p.key}" ${p.default ? "checked" : ""}>`;
    else c = `<input type="number" id="${id}" name="${p.key}" value="${p.default}" min="${p.extra[0]}" max="${p.extra[1]}" step="${p.extra[2]}" required>`;
    return `<div class="param-row ${p.kind === "bool" ? "bool" : ""}"><label for="${id}">${h(p.label)}</label>${c}${p.help ? `<span class="param-help">${h(p.help)}</span>` : ""}</div>`;
  }).join("")}</fieldset>`).join("");
}
function renderExports() {
  if (!S.jobId || !S.status || !["review", "done", "saved", "error"].includes(S.status.state)) return;
  const base = `/api/jobs/${S.jobId}/download/`;
  const pdfs = (S.status.reports || []).map(n => `<a href="${base}${h(n)}"><strong>PDF report</strong><span>${h(n)}</span></a>`).join("");
  const items = [["all.zip", "Everything (ZIP)", "Images, masks, gap analyses, tables, log"], ["gap_analysis.csv", "Gap analysis (CSV)", "Features, CARS, recommendation, decision"],
    ["user_decisions.csv", "User decisions (CSV)", "Every approve / reject / skip with time"], ["evaluation_metrics.json", "Evaluation metrics (JSON)", ""],
    ["road_graph.json", "Road graph (JSON)", "Current nodes and edges"], ["results.json", "All results (JSON)", "Including Tamil explanations"],
    ["processing_log.txt", "Processing log", ""], ["configuration.json", "Configuration", "Parameters of this run"]];
  $("#exports").innerHTML = pdfs + items.map(([f, t, d]) => `<a href="${base}${f}"><strong>${h(t)}</strong><span>${h(d)}</span></a>`).join("");
}
async function makePdf() {
  $("#pdfMsg").textContent = "Generating the PDF report…";
  try {
    const r = await api(`/api/jobs/${S.jobId}/report`, { method: "POST" });
    S.status = await api(`/api/jobs/${S.jobId}/status`);
    renderExports();
    $("#pdfMsg").innerHTML = `Ready: <a href="${h(r.url)}">${h(r.name)}</a>`;
    toast("PDF report generated.");
    window.location.href = r.url;
  } catch (err) { $("#pdfMsg").textContent = err.message; toast(err.message, true); }
}
async function loadHistory() {
  try {
    const d = await api("/api/history");
    $("#historyTable").innerHTML = table([["updated", "Updated"], ["name", "Image"], ["gaps", "Gaps"], ["approved", "Approved"], ["rejected", "Rejected"],
      ["pending", "Pending"], ["reconstructed", "Rebuilt"], ["c", "Components", x => `${x.components_before ?? "–"} → ${x.components_after ?? "–"}`],
      ["o", "", x => `<button class="btn btn-quiet btn-small" data-open-job="${h(x.job_id)}">Open</button>`]], d.history, { empty: "No analyses yet." });
    $$("[data-open-job]").forEach(b => b.onclick = () => openJob(b.dataset.openJob));
  } catch (err) { $("#historyTable").innerHTML = `<p class="empty">${h(err.message)}</p>`; }
}
async function openJob(id) {
  try {
    S.jobId = id; await loadResults(); S.gap = Math.min(firstUndecided(), Math.max(gaps().length - 1, 0));
    closeDrawers(); showScreen("carousel"); buildCarousel(); goTo(0);
  } catch (err) { toast(err.message, true); }
}

// --------------------------------------------------------------------------- //
document.addEventListener("DOMContentLoaded", () => {
  setupVoice();
  bindUpload();
  $("#prevBtn").onclick = prev;
  $("#nextBtn").onclick = next;
  $$("[data-drawer]").forEach(b => b.onclick = () => openDrawer(b.dataset.drawer));
  $$("[data-close]").forEach(b => b.onclick = closeDrawers);
  $("#scrim").onclick = closeDrawers;
  $("#goStart").onclick = () => { closeDrawers(); S.voice.stop(); showScreen("start"); };
  $("#resetParams").onclick = () => { buildParams(); toast("Parameters reset to defaults."); };
  $("#rerunBtn").onclick = async () => { closeDrawers(); try { await startRun(); } catch (err) { toast(err.message, true); } };
  $("#pdfBtn").onclick = makePdf;
  $("#zoomClose").onclick = () => { $("#zoombox").hidden = true; };
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeDrawers(); $("#zoombox").hidden = true; }
    if ($("#carousel").hidden || e.target.closest("input, select, textarea")) return;
    if (e.key === "ArrowRight") next();
    if (e.key === "ArrowLeft") prev();
  });
  loadSchema().catch(err => toast(err.message, true));
  loadSamples();
  loadHistory();
});
