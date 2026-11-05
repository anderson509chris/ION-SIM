// Ion Sim 3D viewer: lit electrodes and ion paths in a page that works offline.
// The scene comes from Python (ionsim/gui/view3d/__init__.py, class Scene) as window.SCENE.
// Coordinates are in mm; y is up. Build: src/build.sh (bundles three.js into ../viewer.js).
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";

const SC = window.SCENE;
const qs = new URLSearchParams(location.search);
const S = {
  stretch: +(qs.get("stretch") || (SC.stretch && SC.stretch.value) || 1),
  n: +(qs.get("n") || SC.nPaths || 100),
  colorTime: (qs.get("color") || SC.pathColor || "group") === "time",
  clip: qs.has("clip") ? qs.get("clip") === "1" : !!(SC.clip && SC.clip.on),
  ui: qs.get("ui") !== "0",
  still: qs.get("still") === "1",
  view: qs.get("view") || SC.view || null,
  hidden: new Set((qs.get("hide") || "").split(",").filter(Boolean).concat((SC.groups || []).filter((g) => g.visible === false).map((g) => g.name))),
  pathsOn: new Set((SC.paths || []).map((p) => p.name)),
};

for (const g of (qs.get("show") || "").split(",").filter(Boolean)) S.hidden.delete(g);

function dec(b64) {
  if (!b64) return new Float32Array(0);
  const s = atob(b64), u = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i);
  return new Float32Array(u.buffer);
}

// ---------------------------------------------------------------- renderer
const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(innerWidth, innerHeight);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.localClippingEnabled = true;
document.getElementById("view").appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(SC.bg || 0x0d1117);
const camera = new THREE.PerspectiveCamera(30, innerWidth / Math.max(innerHeight, 1) || 1.6, 0.05, 1e5);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.HemisphereLight(0xdfe8ff, 0x20242c, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
scene.add(sun);
const fill = new THREE.DirectionalLight(0x9fb4ff, 0.5);
scene.add(fill);

const root = new THREE.Group();          // everything; scale.y is the height stretch
scene.add(root);
const geoRoot = new THREE.Group(), pathRoot = new THREE.Group();
root.add(geoRoot, pathRoot);

const clipPlane = new THREE.Plane(new THREE.Vector3(...((SC.clip && SC.clip.normal) || [1, 0, 0])), -((SC.clip && SC.clip.offset) || 0));
const clipMats = [];

function material(o) {
  const col = new THREE.Color(o.color || "#b8bcc4");
  const op = o.opacity == null ? 1 : o.opacity;
  const m = new THREE.MeshStandardMaterial({
    color: col, roughness: o.rough == null ? 0.5 : o.rough, metalness: o.metal == null ? 0.3 : o.metal,
    emissive: col.clone().multiplyScalar(o.glow || 0), side: THREE.DoubleSide,
    transparent: op < 1, opacity: op, depthWrite: op >= 0.6,
    polygonOffset: !!o.offset, polygonOffsetFactor: o.offset ? -2 : 0, polygonOffsetUnits: o.offset ? -2 : 0,
  });
  if (o.clip) { clipMats.push(m); m.clippingPlanes = S.clip ? [clipPlane] : []; }
  return m;
}
function shapeOf(outer, holes) {
  const sh = new THREE.Shape(outer.map(([a, b]) => new THREE.Vector2(a, b)));
  for (const h of holes || []) sh.holes.push(new THREE.Path(h.map(([a, b]) => new THREE.Vector2(a, b))));
  return sh;
}
// shape coordinates (u, v) and extrusion w -> scene coordinates for each plane
function remap(geo, axis, at) {
  const p = geo.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const u = p.getX(i), v = p.getY(i), w = p.getZ(i);
    if (axis === "y") p.setXYZ(i, u, at + w, v);            // polygon in (x, z), thickness along y
    else if (axis === "z") p.setXYZ(i, u, v, at + w);       // polygon in (x, y), thickness along z
    else p.setXYZ(i, at + w, v, u);                          // axis x: polygon in (z, y)
  }
  geo.computeVertexNormals();
  return geo;
}

const groups = {};
function group(name) {
  if (!groups[name]) { groups[name] = new THREE.Group(); geoRoot.add(groups[name]); }
  return groups[name];
}

function buildGeometry() {
  for (const o of SC.objects || []) {
    const g = group(o.group || "Electrodes");
    let geo;
    if (o.type === "mesh") {
      geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.BufferAttribute(dec(o.tris), 3));
      geo.computeVertexNormals();
    } else if (o.type === "box") {
      geo = new THREE.BoxGeometry(o.x[1] - o.x[0], o.y[1] - o.y[0], o.z[1] - o.z[0]);
      geo.translate((o.x[0] + o.x[1]) / 2, (o.y[0] + o.y[1]) / 2, (o.z[0] + o.z[1]) / 2);
    } else if (o.type === "shape") {
      geo = new THREE.BufferGeometry();
      const gs = o.polys.map((pg) => remap(new THREE.ShapeGeometry(shapeOf(pg.outer, pg.holes)), o.axis || "y", o.at));
      for (const s of gs) g.add(new THREE.Mesh(s, material({ ...o, offset: true })));
      continue;
    } else if (o.type === "slab") {
      geo = remap(new THREE.ExtrudeGeometry(shapeOf(o.outer, o.holes), { depth: o.at[1] - o.at[0], bevelEnabled: false, curveSegments: 32 }), o.axis || "y", o.at[0]);
    } else continue;
    const mesh = new THREE.Mesh(geo, material(o));
    g.add(mesh);
    if (o.edges) {
      const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo, 30), new THREE.LineBasicMaterial({ color: o.edges, transparent: true, opacity: 0.4 }));
      g.add(e);
    }
  }
  for (const [name, g] of Object.entries(groups)) g.visible = !S.hidden.has(name);
}

// ---------------------------------------------------------------- paths
const P = (SC.paths || []).map((p) => ({ ...p, arr: dec(p.pts) }));
const tA = new THREE.Color((SC.timeColors || ["#2fd8ff"])[0]), tB = new THREE.Color((SC.timeColors || [0, "#ff4fd8"])[1]);
let tMax = 0;
for (const p of P) if (p.dim === 4) for (let i = 0; i < p.arr.length; i += 4) tMax = Math.max(tMax, p.arr[i]);
const nMax = Math.max(1, ...P.map((p) => p.off.length - 1));
let lineMats = [];
function buildPaths() {
  while (pathRoot.children.length) pathRoot.remove(pathRoot.children[0]);
  lineMats = [];
  for (const p of P) {
    if (!S.pathsOn.has(p.name)) continue;
    const nAll = p.off.length - 1, n = Math.min(S.n, nAll);
    const pos = [], col = [], base = new THREE.Color(p.color && p.color !== "time" ? p.color : "#2fd8ff"), o = p.dim === 4 ? 1 : 0;
    for (let k = 0; k < n; k++) {
      const j = Math.floor(k * nAll / n);           // the same ions in every group of equal size
      const a = p.off[j], b = p.off[j + 1];
      for (let i = a; i < b - 1; i++) {
        const i0 = i * p.dim, i1 = (i + 1) * p.dim;
        pos.push(p.arr[i0 + o], p.arr[i0 + o + 1], p.arr[i0 + o + 2], p.arr[i1 + o], p.arr[i1 + o + 1], p.arr[i1 + o + 2]);
        if (p.dim === 4 && (S.colorTime || p.color === "time") && tMax > 0) {
          const c0 = tA.clone().lerp(tB, p.arr[i0] / tMax), c1 = tA.clone().lerp(tB, p.arr[i1] / tMax);
          col.push(c0.r, c0.g, c0.b, c1.r, c1.g, c1.b);
        } else col.push(base.r, base.g, base.b, base.r, base.g, base.b);
      }
    }
    if (!pos.length) continue;
    const geo = new LineSegmentsGeometry();
    geo.setPositions(pos);
    geo.setColors(col);
    const mat = new LineMaterial({ linewidth: p.width || 1.5, vertexColors: true, transparent: true, opacity: p.opacity == null ? 0.85 : p.opacity, depthWrite: false });
    mat.resolution.set(innerWidth * devicePixelRatio, innerHeight * devicePixelRatio);
    lineMats.push(mat);
    const ls = new LineSegments2(geo, mat);
    ls.renderOrder = 10;
    pathRoot.add(ls);
  }
}

// ---------------------------------------------------------------- views
const box = new THREE.Box3();
function bounds() {
  box.makeEmpty();
  for (const p of P) for (let i = 0; i < p.arr.length; i += p.dim) {
    const o = p.dim === 4 ? 1 : 0;
    box.expandByPoint(new THREE.Vector3(p.arr[i + o], p.arr[i + o + 1], p.arr[i + o + 2]));
  }
  const gb = new THREE.Box3().setFromObject(geoRoot);
  if (!gb.isEmpty()) box.union(gb);
  if (box.isEmpty()) box.set(new THREE.Vector3(-1, -1, -1), new THREE.Vector3(1, 1, 1));
  return box;
}
function autoViews() {
  // Fit each view to the box from its own direction, so long thin devices fill the window.
  // The camera sits on the cut-away side (x < 0) with z running left to right.
  const b = bounds(), c = b.getCenter(new THREE.Vector3());
  const corners = [];
  for (const x of [b.min.x, b.max.x]) for (const y of [b.min.y, b.max.y]) for (const z of [b.min.z, b.max.z])
    corners.push(new THREE.Vector3(x, y, z).sub(c));
  const tv = Math.tan(15 * Math.PI / 180), th = tv * Math.max(innerWidth / Math.max(innerHeight, 1), 0.3);
  const at = (d) => {
    const dir = new THREE.Vector3(...d).normalize(), F = dir.clone().negate();
    let R = new THREE.Vector3().crossVectors(F, new THREE.Vector3(0, 1, 0));
    if (R.lengthSq() < 1e-9) R.set(0, 0, 1);
    R.normalize();
    const U = new THREE.Vector3().crossVectors(R, F);
    let dist = 1e-6;
    for (const p of corners)
      dist = Math.max(dist, Math.abs(p.dot(R)) / th - p.dot(F), Math.abs(p.dot(U)) / tv - p.dot(F));
    return c.clone().add(dir.multiplyScalar(dist * 1.08)).toArray();
  };
  return [
    { name: "3D", pos: at([-0.75, 0.55, -0.45]), tgt: c.toArray() },
    { name: "top", pos: at([-1e-3, 1, 0]), tgt: c.toArray() },
    { name: "side", pos: at([-1, 0.02, 0]), tgt: c.toArray() },
    { name: "end", pos: at([0, 0.02, -1]), tgt: c.toArray() },
  ];
}
let VIEWS = [], current = null, moved = false;
function setView(name) {
  const v = VIEWS.find((x) => x.name === name) || VIEWS[0];
  current = v.name; moved = false;
  camera.position.set(...v.pos);
  controls.target.set(...v.tgt);
  camera.fov = v.fov || 30;
  camera.updateProjectionMatrix();
  controls.update();
  const d = camera.position.distanceTo(controls.target);
  sun.position.copy(controls.target).add(new THREE.Vector3(-0.6, 1, -0.4).multiplyScalar(d));
  fill.position.copy(controls.target).add(new THREE.Vector3(0.7, 0.4, 0.8).multiplyScalar(d));
}

// ---------------------------------------------------------------- legend and panel
function legend() {
  const el = document.getElementById("legend");
  const rows = P.filter((p) => p.legend !== false).map((p) => {
    const sw = p.dim === 4 && (S.colorTime || p.color === "time") ? `linear-gradient(90deg,#${tA.getHexString()},#${tB.getHexString()})` : p.color;
    const label = p.dim === 4 && (S.colorTime || p.color === "time") ? `${p.name}, early → late` : p.name;
    return `<div class="row"><span class="sw" style="background:${sw}"></span>${label}</div>`;
  });
  let bar = "";
  if (SC.colorbar) {
    const st = SC.colorbar.stops.map(([f, c]) => `${c} ${f * 100}%`).join(",");
    bar = `<div class="bar" style="background:linear-gradient(90deg,${st})"></div><div class="ticks">${SC.colorbar.ticks.map((t) => `<span>${t}</span>`).join("")}</div>`;
  }
  const notes = [SC.colorbar && SC.colorbar.label, S.stretch !== 1 && `heights stretched ${S.stretch}×`, SC.note].filter(Boolean);
  el.innerHTML = rows.join("") + bar + (notes.length ? `<div class="note">${notes.join(" · ")}</div>` : "");
  el.style.display = rows.length || bar ? "block" : "none";
  document.getElementById("title").innerHTML = SC.title ? `<b>${SC.title}</b>${SC.subtitle ? "<br>" + SC.subtitle : ""}` : "";
  document.getElementById("foot").textContent = SC.footer || "";
}
function panel() {
  const p = document.getElementById("panel");
  if (!S.ui) { p.style.display = "none"; return; }
  const gnames = (SC.groups || []).filter((g) => g.toggle !== false).map((g) => g.name);
  p.innerHTML = `
    <div class="btns">${VIEWS.map((v) => `<button data-v="${v.name}">${v.name}</button>`).join("")}</div>
    ${nMax > 1 ? `<label>Ions <input id="n" type="range" min="1" max="${nMax}" value="${Math.min(S.n, nMax)}"><span id="nv">${Math.min(S.n, nMax)}</span></label>` : ""}
    ${SC.stretch ? `<label>${SC.stretch.label || "Height stretch"} <input id="st" type="range" min="1" max="${SC.stretch.max || 8}" step="1" value="${S.stretch}"><span id="stv">${S.stretch}×</span></label>` : ""}
    ${P.some((q) => q.dim === 4) ? `<label><input id="ct" type="checkbox" ${S.colorTime ? "checked" : ""}> colour paths by time</label>` : ""}
    ${SC.clip ? `<label><input id="clip" type="checkbox" ${S.clip ? "checked" : ""}> ${SC.clip.label || "cut away the front half"}</label>` : ""}
    ${P.length > 1 ? P.map((q, i) => `<label><input class="pt" data-i="${i}" type="checkbox" ${S.pathsOn.has(q.name) ? "checked" : ""}> ${q.name}</label>`).join("") : ""}
    ${gnames.map((g, i) => `<label><input class="gt" data-g="${g}" type="checkbox" ${S.hidden.has(g) ? "" : "checked"}> ${g}</label>`).join("")}
    <div class="hint">drag to turn · right-drag to move · scroll to zoom</div>`;
  p.querySelectorAll("button").forEach((b) => (b.onclick = () => setView(b.dataset.v)));
  const on = (id, ev, f) => { const e = p.querySelector(id); if (e) e[ev] = f; };
  on("#n", "oninput", (e) => { S.n = +e.target.value; p.querySelector("#nv").textContent = S.n; buildPaths(); });
  on("#st", "oninput", (e) => { S.stretch = +e.target.value; p.querySelector("#stv").textContent = S.stretch + "×"; root.scale.y = S.stretch; legend(); });
  on("#ct", "onchange", (e) => { S.colorTime = e.target.checked; buildPaths(); legend(); });
  on("#clip", "onchange", (e) => { S.clip = e.target.checked; clipMats.forEach((m) => (m.clippingPlanes = S.clip ? [clipPlane] : [])); });
  p.querySelectorAll(".pt").forEach((e) => (e.onchange = () => { const n = P[+e.dataset.i].name; e.checked ? S.pathsOn.add(n) : S.pathsOn.delete(n); buildPaths(); }));
  p.querySelectorAll(".gt").forEach((e) => (e.onchange = () => { groups[e.dataset.g] && (groups[e.dataset.g].visible = e.checked); }));
}

buildGeometry();
buildPaths();
root.scale.y = S.stretch;
VIEWS = (SC.views && SC.views.length) ? SC.views : autoViews();
setView(S.view || VIEWS[0].name);
legend();
panel();
controls.addEventListener("start", () => { moved = true; });
addEventListener("resize", () => {
  if (!innerWidth || !innerHeight) return;            // hidden (a tab in the app not shown yet)
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  lineMats.forEach((m) => m.resolution.set(innerWidth * devicePixelRatio, innerHeight * devicePixelRatio));
  if (!moved) {                                       // refit the view until the user turns it
    if (!(SC.views && SC.views.length)) VIEWS = autoViews();
    setView(current);
  }
});
let frames = 0;
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
  if (++frames === 3) { window.READY = true; if (S.still) renderer.setAnimationLoop(null); }
});
window.setView = setView;
