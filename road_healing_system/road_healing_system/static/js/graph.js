/* Interactive road graph: draws the actual NetworkX graph (from /api/jobs/<id>/graph)
   in image coordinates on a ZoomViewer, so every node and edge sits on its image region. */
"use strict";

const COMPONENT_COLORS = ["#4287f5", "#14b8a6", "#a855f7", "#ec4899", "#0ea5e9", "#84cc16", "#f59e0b", "#6366f1",
  "#10b981", "#f43f5e"];

class RoadGraphView {
  constructor(container, width, height, imageHref, onSelect) {
    this.viewer = new ZoomViewer(container, width, height, { label: "Road graph (scroll to zoom, drag to pan)" });
    this.onSelect = onSelect || (() => {});
    this.imageHref = imageHref;
    this.data = null;
    this.focusComponent = null;
    this.selected = null;
    this.viewer.setImage("image", imageHref, { opacity: 0.45 });
  }

  setImageVisible(on) { this.viewer.setVisible("image", on); }

  render(data) {
    this.data = data;
    const esc = (s) => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
    let edges = "";
    for (const e of data.edges) {
      const pts = e.polyline.map(p => `${p[0]},${p[1]}`).join(" ");
      const recon = e.kind === "reconstructed";
      const col = recon ? "#22c55e" : COMPONENT_COLORS[(e.component || 0) % COMPONENT_COLORS.length];
      const dim = this.focusComponent != null && e.component !== this.focusComponent ? 0.15 : 1;
      edges += `<polyline class="clickable-mark" data-edge="${esc(e.id)}" points="${pts}" fill="none" stroke="${col}"
        stroke-width="${recon ? 5 : 3}" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"
        opacity="${dim}" style="cursor:pointer"><title>edge ${esc(e.id)}${e.gap_id ? " (" + esc(e.gap_id) + ")" : ""}</title></polyline>`;
    }
    let nodes = "";
    for (const n of data.nodes) {
      if (n.type === "waypoint") continue;
      const col = n.type === "endpoint" ? (n.at_border ? "#9ca3af" : "#facc15") : n.type === "junction" ? "#d946ef" : "#e5e7eb";
      const dim = this.focusComponent != null && n.component !== this.focusComponent ? 0.15 : 1;
      nodes += `<circle class="clickable-mark" data-node="${n.id}" cx="${n.x}" cy="${n.y}" data-r="${n.type === "junction" ? 6 : 5}"
        r="5" fill="${col}" stroke="#111" stroke-width="1" vector-effect="non-scaling-stroke" opacity="${dim}" style="cursor:pointer">
        <title>node ${n.id} (${n.type})</title></circle>`;
    }
    const sel = this._selectionMarkup();
    this.viewer.setVector("graph", edges + nodes + sel);
    this.viewer.layers.graph.querySelectorAll("[data-edge]").forEach(el =>
      el.addEventListener("click", () => this.select("edge", el.dataset.edge)));
    this.viewer.layers.graph.querySelectorAll("[data-node]").forEach(el =>
      el.addEventListener("click", () => this.select("node", Number(el.dataset.node))));
  }

  _selectionMarkup() {
    if (!this.selected || !this.data) return "";
    if (this.selected.type === "node") {
      const n = this.data.nodes.find(x => x.id === this.selected.id);
      return n ? `<circle cx="${n.x}" cy="${n.y}" data-r="11" r="11" fill="none" stroke="#fff" stroke-width="3" vector-effect="non-scaling-stroke"/>` : "";
    }
    const e = this.data.edges.find(x => x.id === this.selected.id);
    return e ? `<polyline points="${e.polyline.map(p => p.join(",")).join(" ")}" fill="none" stroke="#fff" stroke-width="9"
      stroke-opacity="0.45" stroke-linecap="round" vector-effect="non-scaling-stroke" pointer-events="none"/>` : "";
  }

  select(type, id) {
    this.selected = { type, id };
    this.render(this.data);
    if (type === "node") {
      const n = this.data.nodes.find(x => x.id === id);
      this.onSelect({ type, item: n });
    } else {
      const e = this.data.edges.find(x => x.id === id);
      if (e) {
        const xs = e.polyline.map(p => p[0]), ys = e.polyline.map(p => p[1]);
        this.viewer.zoomTo([Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)], 0.6);
      }
      this.onSelect({ type, item: e });
    }
  }

  focus(component) {
    this.focusComponent = component;
    this.render(this.data);
  }
}

window.RoadGraphView = RoadGraphView;
window.COMPONENT_COLORS = COMPONENT_COLORS;
