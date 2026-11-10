/* ZoomViewer: an SVG canvas in image-pixel coordinates.
   Image layers and vector overlays share one coordinate system, so overlays stay
   aligned while zooming and panning. Supports wheel zoom at the cursor, drag to
   pan, buttons, fit-to-region, fullscreen and synchronised viewers. */
"use strict";

const SVGNS = "http://www.w3.org/2000/svg";

class ZoomViewer {
  constructor(container, width, height, opts = {}) {
    this.el = container;
    this.W = width; this.H = height;
    this.linked = [];
    this.onView = opts.onView || null;
    this.el.classList.add("zv");
    this.el.innerHTML = "";
    this.svg = document.createElementNS(SVGNS, "svg");
    this.svg.setAttribute("preserveAspectRatio", "xMidYMid meet");
    this.root = document.createElementNS(SVGNS, "g");
    this.svg.appendChild(this.root);
    this.el.appendChild(this.svg);
    this.layers = {};
    if (opts.label) {
      const l = document.createElement("div");
      l.className = "zv-label"; l.textContent = opts.label; this.el.appendChild(l);
    }
    const tools = document.createElement("div");
    tools.className = "zv-tools";
    tools.innerHTML = `<button type="button" title="Zoom in" aria-label="Zoom in">+</button>
      <button type="button" title="Zoom out" aria-label="Zoom out">−</button>
      <button type="button" title="Reset view" aria-label="Reset view">⟲</button>
      <button type="button" title="Full screen" aria-label="Full screen">⛶</button>`;
    const [bin, bout, breset, bfull] = tools.querySelectorAll("button");
    bin.onclick = () => this.zoomBy(1 / 1.4);
    bout.onclick = () => this.zoomBy(1.4);
    breset.onclick = () => this.reset();
    bfull.onclick = () => this.fullscreen();
    this.el.appendChild(tools);
    this.view = { x: 0, y: 0, w: width, h: height };
    this._bind();
    this.reset();
  }

  _bind() {
    this.svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      const p = this.toImage(e.clientX, e.clientY);
      this.zoomBy(e.deltaY > 0 ? 1.15 : 1 / 1.15, p.x, p.y);
    }, { passive: false });
    let drag = null;
    this.svg.addEventListener("pointerdown", (e) => {
      if (e.button !== 0 || e.target.closest(".clickable-mark")) return;
      drag = { x: e.clientX, y: e.clientY, view: { ...this.view }, moved: false };
      this.svg.setPointerCapture(e.pointerId);
      this.svg.classList.add("dragging");
    });
    this.svg.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const scale = this._unitsPerPx();
      const dx = (e.clientX - drag.x) * scale, dy = (e.clientY - drag.y) * scale;
      if (Math.abs(dx) + Math.abs(dy) > 0) drag.moved = true;
      this.setView({ ...drag.view, x: drag.view.x - dx, y: drag.view.y - dy });
    });
    const end = () => { drag = null; this.svg.classList.remove("dragging"); };
    this.svg.addEventListener("pointerup", end);
    this.svg.addEventListener("pointercancel", end);
    this.svg.addEventListener("dblclick", (e) => { const p = this.toImage(e.clientX, e.clientY); this.zoomBy(1 / 2, p.x, p.y); });
    new ResizeObserver(() => this._apply(true)).observe(this.el);
  }

  _unitsPerPx() {
    const r = this.svg.getBoundingClientRect();
    if (!r.width || !r.height) return 1;
    return Math.max(this.view.w / r.width, this.view.h / r.height);
  }

  toImage(cx, cy) {
    const pt = this.svg.createSVGPoint();
    pt.x = cx; pt.y = cy;
    const m = this.svg.getScreenCTM();
    if (!m) return { x: this.view.x + this.view.w / 2, y: this.view.y + this.view.h / 2 };
    const p = pt.matrixTransform(m.inverse());
    return { x: p.x, y: p.y };
  }

  setView(v, silent = false) {
    const minW = Math.max(this.W, this.H) / 40, maxW = Math.max(this.W, this.H) * 1.5;
    const w = Math.min(Math.max(v.w, minW), maxW);
    const h = w * (this.H / this.W);
    this.view = { x: v.x, y: v.y, w, h };
    this._apply();
    if (!silent) {
      this.linked.forEach(o => o.setView(this.view, true));
      if (this.onView) this.onView(this.view);
    }
  }

  _apply() {
    const v = this.view;
    this.svg.setAttribute("viewBox", `${v.x} ${v.y} ${v.w} ${v.h}`);
    const upp = this._unitsPerPx();
    this.svg.querySelectorAll("[data-r]").forEach(c => c.setAttribute("r", Number(c.dataset.r) * upp));
    this.svg.querySelectorAll("[data-fs]").forEach(t => t.setAttribute("font-size", Number(t.dataset.fs) * upp));
  }

  zoomBy(f, cx, cy) {
    const v = this.view;
    cx = cx ?? v.x + v.w / 2; cy = cy ?? v.y + v.h / 2;
    const w = v.w * f, h = v.h * f;
    this.setView({ x: cx - (cx - v.x) * (w / v.w), y: cy - (cy - v.y) * (h / v.h), w });
  }

  reset() { this.setView({ x: 0, y: 0, w: this.W }); }

  zoomTo(box, pad = 0.35) {
    const [x0, y0, x1, y1] = box;
    const bw = Math.max(x1 - x0, 10), bh = Math.max(y1 - y0, 10);
    const w = Math.max(bw, bh * (this.W / this.H)) * (1 + 2 * pad);
    const h = w * (this.H / this.W);
    this.setView({ x: (x0 + x1) / 2 - w / 2, y: (y0 + y1) / 2 - h / 2, w });
  }

  link(other) {
    if (!this.linked.includes(other)) this.linked.push(other);
    if (!other.linked.includes(this)) other.linked.push(this);
    other.setView(this.view, true);
  }

  fullscreen() {
    try {
      if (document.fullscreenElement) document.exitFullscreen();
      else this.el.requestFullscreen();
    } catch { /* fullscreen not allowed: ignore */ }
  }

  _layer(id) {
    let g = this.layers[id];
    if (!g) {
      g = document.createElementNS(SVGNS, "g");
      g.dataset.layer = id;
      this.root.appendChild(g);
      this.layers[id] = g;
    }
    return g;
  }

  setImage(id, href, opts = {}) {
    const g = this._layer(id);
    g.innerHTML = "";
    if (!href) return;
    const img = document.createElementNS(SVGNS, "image");
    img.setAttribute("href", href);
    img.setAttribute("x", 0); img.setAttribute("y", 0);
    img.setAttribute("width", this.W); img.setAttribute("height", this.H);
    img.setAttribute("preserveAspectRatio", "none");
    if (opts.pixelated) img.style.imageRendering = "pixelated";
    if (opts.opacity != null) img.setAttribute("opacity", opts.opacity);
    g.appendChild(img);
    if (opts.visible === false) g.style.display = "none";
  }

  setVector(id, markup, opts = {}) {
    const g = this._layer(id);
    g.innerHTML = markup;
    if (opts.visible === false) g.style.display = "none";
    this._apply();
    return g;
  }

  setVisible(id, on) { const g = this.layers[id]; if (g) g.style.display = on ? "" : "none"; }
}

window.ZoomViewer = ZoomViewer;
