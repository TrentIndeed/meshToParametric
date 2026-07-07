import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';
// Loaders for client-side folder thumbnails (parse meshes in the browser to
// render a small preview, no server round-trip). 3MFLoader pulls fflate itself.
import { STLLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/STLLoader.js';
import { OBJLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/OBJLoader.js';
import { PLYLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/PLYLoader.js';
import { ThreeMFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/3MFLoader.js';
import * as fflate from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/libs/fflate.module.js';

console.log('%c[paramesh] app.js BUILD 20260707-anim1 loaded', 'color:#0969DA;font-weight:bold');
console.debug('[init] app.js loading, Three.js revision:', THREE.REVISION);

// Sample a sketch-overlay arc descriptor {ax,pos,cu,cv,r,a0,a1} into a smooth
// polyline of THREE.Vector3 (a TRUE curve, not the old chord between sampled
// outline points). The in-plane axes are the two indices other than `ax`,
// ascending — matching the backend's (u_axis, v_axis) convention.
function arcToPoints(arc) {
  const axes = [0, 1, 2].filter(a => a !== arc.ax);
  const ua = axes[0], va = axes[1];
  const sweep = Math.abs(arc.a1 - arc.a0);
  const n = Math.max(8, Math.ceil(sweep / (Math.PI / 48)));
  const pts = [];
  for (let i = 0; i <= n; i++) {
    const t = arc.a0 + (arc.a1 - arc.a0) * (i / n);
    const p = [0, 0, 0];
    p[ua] = arc.cu + arc.r * Math.cos(t);
    p[va] = arc.cv + arc.r * Math.sin(t);
    p[arc.ax] = arc.pos;
    pts.push(new THREE.Vector3(p[0], p[1], p[2]));
  }
  return pts;
}

// Shared feature-tree row rendering — used by BOTH the main left-sidebar tree
// and the live-build pane's in-pane tree, so the two are guaranteed identical
// (same icons, same name, same detail). Single source of truth.
const FEATURE_ICON_MAP = {
  'Sketch': ['sketch.png', 'sketch'],
  'Extrude': ['extrude.png', 'extrude'],
  'Through': ['hole.png', 'hole'],
  'Blind': ['hole.png', 'hole'],
  'Hole': ['hole.png', 'hole'],
  'Chamfer': ['chamfer.png', 'chamfer'],
  'Side Profile': ['sketch.png', 'sketch'],
  'Side Cut': ['chamfer.png', 'chamfer'],
  'Fillet': ['fillet.png', 'fillet'],
  'Pocket': ['shell.png', 'pocket'],
  'Boss': ['extrude.png', 'boss'],
  'Join': ['extrude.png', 'boss'],
  'Slot': ['shell.png', 'slot'],
  'Loft': ['loft.png', 'extrude'],
  'Revolve': ['revolve.png', 'extrude'],
  'Sweep': ['sweep.png', 'extrude'],
  'Shell': ['shell.png', 'pocket'],
  'Original': ['bodies.png', 'sketch'],
};
function featureIconInfo(name) {
  for (const [key, [file, cls]] of Object.entries(FEATURE_ICON_MAP)) {
    if (name.includes(key)) return { file, cls };
  }
  return { file: 'component.png', cls: 'extrude' };
}
// Inner HTML for a `.feature-item` row (icon + name + detail).
function featureRowInnerHTML(name, volume) {
  const { file, cls } = featureIconInfo(name);
  const shortName = (name || '').split(/\s*[—→]\s*/)[0].trim();
  const detail = (volume > 0) ? volume.toLocaleString() + ' mm³' : '2D';
  return '<div class="icon ' + cls + '">' +
    '<img src="/static/icons/' + file + '" style="width:22px;height:22px;border-radius:50%;"></div>' +
    '<div class="texts"><div class="name">' + shortName + '</div>' +
    '<div class="detail">' + detail + '</div></div>';
}

// Horizontal scroll on timeline with mouse wheel
document.getElementById('timeline').addEventListener('wheel', (e) => {
  e.preventDefault();
  document.getElementById('timeline').scrollLeft += e.deltaY;
}, { passive: false });

// ── State ──
let currentStep = -1;
let steps = [];
let currentJobId = null;
// Check if we should restore a mesh preview (set before anything loads)
const _hasPreviewToRestore = !!sessionStorage.getItem('paramesh_preview');
// Expose for automation
Object.defineProperty(window, '_currentJobId', {
  get() { return currentJobId; },
  set(v) { currentJobId = v; },
});

// ── Three.js Setup ──
const container = document.getElementById('viewer');
const scene = new THREE.Scene();
scene.background = new THREE.Color(
  document.documentElement.getAttribute('data-theme') === 'dark' ? 0x0D1117 : 0xFFFFFF
);

const camera = new THREE.PerspectiveCamera(
  45, container.clientWidth / container.clientHeight, 0.1, 5000
);
camera.position.set(200, 150, 200);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.localClippingEnabled = true;   // feature build animations
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.4;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);
console.debug('[init] WebGL renderer created, size:', container.clientWidth, 'x', container.clientHeight, 'pixelRatio:', Math.min(window.devicePixelRatio, 2));

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.enableZoom = false; // we handle zoom ourselves with smooth interpolation

// Track user dragging for idle rotation pause
window._userDragging = false;
controls.addEventListener('start', () => { window._userDragging = true; window._idleDist = null; });
controls.addEventListener('end', () => { window._userDragging = false; });

const ZOOM_MIN = 30;
const ZOOM_MAX = 1500;
const ZOOM_FACTOR = 0.08; // 8% per scroll tick
const ZOOM_LERP = 0.12; // smoothing factor per frame (higher = snappier)
let _zoomTargetDist = null; // null = no pending zoom

renderer.domElement.addEventListener('wheel', (e) => {
  e.preventDefault();
  const dist = _zoomTargetDist ?? camera.position.distanceTo(controls.target);
  const sign = e.deltaY > 0 ? 1 : -1;
  _zoomTargetDist = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, dist * (1 + sign * ZOOM_FACTOR)));
}, { passive: false });

// ── Studio Lighting — bright, even illumination ──
const hemi = new THREE.HemisphereLight(0xE0E8F0, 0x3D4450, 0.8);
scene.add(hemi);

const ambient = new THREE.AmbientLight(0x556677, 0.5);
scene.add(ambient);

const key = new THREE.DirectionalLight(0xFFFFFF, 1.4);
key.position.set(200, 300, 150);
key.castShadow = true;
key.shadow.mapSize.width = 1024;
key.shadow.mapSize.height = 1024;
scene.add(key);

const fill = new THREE.DirectionalLight(0x99AACC, 0.7);
fill.position.set(-150, 100, -100);
scene.add(fill);

const back = new THREE.DirectionalLight(0xFFFFFF, 0.4);
back.position.set(-50, 150, -200);
scene.add(back);

const bottom = new THREE.DirectionalLight(0x8899AA, 0.5);
bottom.position.set(0, -200, 50);
scene.add(bottom);

// ── Cursor coordinate readout ──
// Raycasts the mouse against the visible modelGroup and shows the
// world-space (x, y, z) under the cursor in a fixed div. Useful for
// confirming which region you're hovering when diagnosing fillet
// detection or other per-region issues.
const _coordRaycaster = new THREE.Raycaster();
const _coordMouse = new THREE.Vector2();
const _coordEl = document.getElementById('cursor-coords');
const _coordText = document.getElementById('cursor-coords-text');
let _coordHideTimer = null;
renderer.domElement.addEventListener('mousemove', (ev) => {
  const rect = renderer.domElement.getBoundingClientRect();
  _coordMouse.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  _coordMouse.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  _coordRaycaster.setFromCamera(_coordMouse, camera);
  const hits = _coordRaycaster.intersectObject(modelGroup, true);
  if (hits.length > 0 && _coordEl && _coordText) {
    const p = hits[0].point;
    _coordText.textContent =
      `x ${p.x.toFixed(2)}  y ${p.y.toFixed(2)}  z ${p.z.toFixed(2)}`;
    _coordEl.style.display = '';
    if (_coordHideTimer) { clearTimeout(_coordHideTimer); _coordHideTimer = null; }
  } else if (_coordEl) {
    if (_coordHideTimer) clearTimeout(_coordHideTimer);
    _coordHideTimer = setTimeout(() => { _coordEl.style.display = 'none'; }, 200);
  }
});

// ── Grid + Origin axes ──
let sceneGrid = null;
let originAxes = [];

function createGrid() {
  const gridSize = 300;
  const divisions = 30;
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  sceneGrid = new THREE.GridHelper(gridSize, divisions,
    isDark ? 0x222830 : 0xE0E0E0,
    isDark ? 0x1A1E24 : 0xEEEEEE
  );
  sceneGrid.material.transparent = true;
  sceneGrid.material.opacity = 0.5;
  sceneGrid.visible = false; // hidden by default
  scene.add(sceneGrid);

  // Origin axes — full-length colored lines through origin (off by default)
  const axLen = 500;
  [[axLen,0,0],[0,axLen,0],[0,0,axLen]].forEach(([x,y,z], i) => {
    const color = [0xF85149, 0x3FB950, 0x58A6FF][i];
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-x,-y,-z),
      new THREE.Vector3(x,y,z),
    ]);
    const m = new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.4 });
    const line = new THREE.Line(g, m);
    line.visible = false; // off by default
    originAxes.push(line);
    scene.add(line);
  });
}
createGrid();

// Toggle functions for menu
window.toggleGrid = function() {
  if (sceneGrid) sceneGrid.visible = !sceneGrid.visible;
  document.getElementById('app-menu').classList.remove('show');
};
window.toggleOriginAxes = function() {
  const visible = !originAxes[0]?.visible;
  originAxes.forEach(a => a.visible = visible);
  const eye = document.querySelector('#origin-eye i');
  if (eye) eye.className = visible ? 'ri-eye-line' : 'ri-eye-off-line';
  const menu = document.getElementById('app-menu');
  if (menu) menu.classList.remove('show');
};

// ── Theme-aware scene background ──
function updateSceneBg() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  scene.background = new THREE.Color(isDark ? 0x0D1117 : 0xFFFFFF);
  // Update grid colors
  if (sceneGrid) {
    sceneGrid.material.color.set(isDark ? 0x222830 : 0xE0E0E0);
    if (Array.isArray(sceneGrid.material)) {
      sceneGrid.material[0].color.set(isDark ? 0x222830 : 0xE0E0E0);
      sceneGrid.material[1].color.set(isDark ? 0x1A1E24 : 0xEEEEEE);
    }
  }
  // Update axis gizmo border
  axisRenderer.domElement.style.borderColor = isDark ? 'rgba(48,54,61,0.5)' : 'rgba(208,215,222,0.5)';
  // Re-render thumbnails with new background color
  if (cachedStatusData) populateUI(cachedStatusData);
}
window._updateSceneBg = updateSceneBg;

// Small XYZ axis gizmo in the bottom-left corner (rendered separately)
const axisScene = new THREE.Scene();
const axisCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
axisCamera.position.set(3, 2, 3);
axisCamera.lookAt(0, 0, 0);
const axisRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
axisRenderer.setSize(84, 84);
axisRenderer.setClearColor(0x000000, 0);
console.debug('[init] Axis gizmo renderer created');
const axIsDark = document.documentElement.getAttribute('data-theme') === 'dark';
axisRenderer.domElement.style.cssText = `position:fixed;top:56px;right:310px;z-index:60;pointer-events:none;border-radius:6px;`;
document.body.appendChild(axisRenderer.domElement);

// Axis lines with cone tips (short, in gizmo scene)
axisScene.add(new THREE.AmbientLight(0xffffff, 1));
[[1.2,0,0,0xF85149],[0,1.2,0,0x3FB950],[0,0,1.2,0x58A6FF]].forEach(([x,y,z,color]) => {
  const geo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3(x,y,z)]);
  axisScene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color, linewidth: 2 })));
  // Arrow tip
  const coneGeo = new THREE.ConeGeometry(0.08, 0.25, 8);
  const coneMat = new THREE.MeshBasicMaterial({ color });
  const cone = new THREE.Mesh(coneGeo, coneMat);
  cone.position.set(x, y, z);
  if (x > 0) cone.rotation.z = -Math.PI / 2;
  else if (z > 0) cone.rotation.x = Math.PI / 2;
  axisScene.add(cone);
});

// ── Model group ──
let modelGroup = new THREE.Group();
scene.add(modelGroup);

// Expose for automation/testing
window._camera = camera;
window._controls = controls;
window._modelGroup = modelGroup;
window._THREE = THREE;

// ── Progress geometry renderer (live pipeline visualization) ──
let _lastProgressStage = '';
function _renderProgressGeometry(pg) {
  if (!pg) return;
  if (window._pipelineCancelled) return;

  // If pieces are provided, render each as a separate mesh
  if (pg.pieces && pg.pieces.length > 0) {
    // Skip if same version already rendered
    const pVer = pg.version || 0;
    if (window._explodePiecesVersion === pVer) return;
    window._explodePiecesVersion = pVer;

    // Clear
    while (modelGroup.children.length) {
      const child = modelGroup.children[0];
      if (child.geometry) child.geometry.dispose();
      if (child.material) { if (Array.isArray(child.material)) child.material.forEach(m => m.dispose()); else child.material.dispose(); }
      modelGroup.remove(child);
    }
    // Compute overall center for explode directions
    let allMin = [Infinity, Infinity, Infinity];
    let allMax = [-Infinity, -Infinity, -Infinity];
    for (const piece of pg.pieces) {
      for (let i = 0; i < piece.vertices.length; i += 3) {
        for (let a = 0; a < 3; a++) {
          allMin[a] = Math.min(allMin[a], piece.vertices[i + a]);
          allMax[a] = Math.max(allMax[a], piece.vertices[i + a]);
        }
      }
    }
    const overallCenter = allMin.map((mn, i) => (mn + allMax[i]) / 2);
    const overallSize = allMax.map((mx, i) => mx - allMin[i]);

    const pieceMeshes = [];
    for (let pi = 0; pi < pg.pieces.length; pi++) {
      const piece = pg.pieces[pi];
      const verts = new Float32Array(piece.vertices);
      const indices = piece.faces;
      const hasPerFaceColors = piece.colors && piece.colors.length > 0;
      const numFaces = Math.floor(indices.length / 3);

      let geo;
      if (hasPerFaceColors) {
        // Build non-indexed geometry with per-face colors
        const positions = new Float32Array(numFaces * 9);
        const colors = new Float32Array(numFaces * 9);
        for (let i = 0; i < numFaces; i++) {
          const a = indices[i * 3], b = indices[i * 3 + 1], c = indices[i * 3 + 2];
          const fR = piece.colors[i * 3], fG = piece.colors[i * 3 + 1], fB = piece.colors[i * 3 + 2];
          for (let j = 0; j < 3; j++) {
            const vi = [a, b, c][j];
            positions[i * 9 + j * 3]     = verts[vi * 3];
            positions[i * 9 + j * 3 + 1] = verts[vi * 3 + 1];
            positions[i * 9 + j * 3 + 2] = verts[vi * 3 + 2];
            colors[i * 9 + j * 3]     = fR;
            colors[i * 9 + j * 3 + 1] = fG;
            colors[i * 9 + j * 3 + 2] = fB;
          }
        }
        geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geo.computeVertexNormals();
      } else {
        geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
        geo.setIndex(indices);
        geo.computeVertexNormals();
      }

      if (hasPerFaceColors) {
        // Stage viz: gray base + vivid per-face color overlay + edges
        const baseMat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(0.52, 0.55, 0.58),
          metalness: 0.12, roughness: 0.65, side: THREE.FrontSide,
        });
        const mesh = new THREE.Mesh(geo, baseMat);
        mesh.castShadow = true;
        modelGroup.add(mesh);

        const overlayMat = new THREE.MeshPhongMaterial({
          vertexColors: true,
          emissive: new THREE.Color(0.15, 0.15, 0.15),
          specular: new THREE.Color(0.15, 0.15, 0.15),
          shininess: 25, transparent: false, side: THREE.FrontSide,
          polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
        });
        mesh.add(new THREE.Mesh(geo.clone(), overlayMat));

        const edges = new THREE.EdgesGeometry(geo, 8);
        mesh.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
          color: new THREE.Color(0.06, 0.06, 0.08), transparent: true, opacity: 0.85,
        })));

        const pBox = new THREE.Box3().setFromBufferAttribute(geo.attributes.position);
        const pCenter = pBox.getCenter(new THREE.Vector3());
        const dir = pCenter.clone().sub(new THREE.Vector3(...overallCenter)).normalize();
        const pSize = pBox.getSize(new THREE.Vector3());
        const explodeDist = Math.max(pSize.x, pSize.y, pSize.z) * 0.20;
        pieceMeshes.push({ mesh, dir, explodeDist, restPos: mesh.position.clone() });
      } else {
        // Decomposition viz: gray base + solid tint overlay
        const baseMat = new THREE.MeshStandardMaterial({
          color: new THREE.Color(0.52, 0.55, 0.58),
          metalness: 0.12, roughness: 0.65,
          transparent: true, opacity: 0.5, side: THREE.DoubleSide,
        });
        const mesh = new THREE.Mesh(geo, baseMat);
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        modelGroup.add(mesh);

        const tint = piece.color || [0.55, 0.58, 0.62];
        const tintColor = new THREE.Color(tint[0], tint[1], tint[2]);
        const overlayMat = new THREE.MeshPhongMaterial({
          color: tintColor,
          emissive: tintColor.clone().multiplyScalar(0.5),
          specular: new THREE.Color(0.15, 0.15, 0.15),
          shininess: 25, transparent: true, opacity: 0.85,
          depthWrite: false, side: THREE.FrontSide,
          polygonOffset: true, polygonOffsetFactor: -1, polygonOffsetUnits: -1,
        });
        mesh.add(new THREE.Mesh(geo.clone(), overlayMat));

        const edgeColor = tintColor.clone().lerp(new THREE.Color(0, 0, 0), 0.55);
        const edges = new THREE.EdgesGeometry(geo, 8);
        mesh.add(new THREE.LineSegments(edges, new THREE.LineBasicMaterial({
          color: edgeColor, transparent: true, opacity: 0.85,
        })));

        const pBox = new THREE.Box3().setFromBufferAttribute(geo.attributes.position);
        const pCenter = pBox.getCenter(new THREE.Vector3());
        const dir = pCenter.clone().sub(new THREE.Vector3(...overallCenter)).normalize();
        const pSize = pBox.getSize(new THREE.Vector3());
        const explodeDist = Math.max(pSize.x, pSize.y, pSize.z) * 0.20;
        pieceMeshes.push({ mesh, dir, explodeDist, restPos: mesh.position.clone() });
      }
    }

    // Center the group
    modelGroup.position.set(0, 0, 0);
    try {
      const box = new THREE.Box3().setFromObject(modelGroup);
      const center = box.getCenter(new THREE.Vector3());
      modelGroup.position.sub(center);
    } catch (e) {}
    document.getElementById('empty-state').style.display = 'none';

    // Explode animation: only for decomposition stage (no per-face colors)
    const isDecompStage = pg.pieces[0] && !pg.pieces[0].colors;
    if (pieceMeshes.length > 1 && isDecompStage) {
      const duration = 2000;
      const startTime = performance.now();
      function animateExplode() {
        const elapsed = performance.now() - startTime;
        const t = Math.min(elapsed / duration, 1.0);
        // Smooth sine wave: out and back with no pause
        const factor = Math.sin(t * Math.PI);
        for (const p of pieceMeshes) {
          p.mesh.position.copy(p.restPos).addScaledVector(p.dir, p.explodeDist * factor);
        }
        if (t < 1.0) requestAnimationFrame(animateExplode);
      }
      requestAnimationFrame(animateExplode);
    }

    // Edge lines (topology stage) — also render when showing pieces
    if (pg.edges && pg.edges.length >= 6) {
      const edgeVerts = pg.edges;
      const edgeColors = pg.edge_colors;
      const pts = [];
      const eCols = [];
      for (let i = 0; i < edgeVerts.length; i += 3) {
        pts.push(new THREE.Vector3(edgeVerts[i], edgeVerts[i+1], edgeVerts[i+2]));
        if (edgeColors && edgeColors.length > i) {
          eCols.push(edgeColors[i], edgeColors[i+1], edgeColors[i+2]);
        }
      }
      const edgeGeo = new THREE.BufferGeometry().setFromPoints(pts);
      if (eCols.length > 0) {
        edgeGeo.setAttribute('color', new THREE.Float32BufferAttribute(eCols, 3));
      }
      const edgeMat = new THREE.LineBasicMaterial({
        vertexColors: eCols.length > 0,
        color: eCols.length > 0 ? 0xffffff : 0x58A6FF,
        linewidth: 1,
        depthTest: false,
      });
      modelGroup.add(new THREE.LineSegments(edgeGeo, edgeMat));
    }

    return;
  }

  if (!pg.vertices || !pg.faces) return;

  // Clear previous model
  while (modelGroup.children.length) {
    const child = modelGroup.children[0];
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else child.material.dispose();
    }
    modelGroup.remove(child);
  }

  const verts = new Float32Array(pg.vertices);
  const indices = pg.faces;
  const numFaces = Math.floor(indices.length / 3);

  // Build non-indexed geometry with per-face colors
  const positions = new Float32Array(numFaces * 9);
  const colors = new Float32Array(numFaces * 9);
  const hasColors = pg.colors && pg.colors.length > 0;
  const perFace = pg.colors_per_face; // colors array is (numFaces*3) not (numVerts*3)

  for (let i = 0; i < numFaces; i++) {
    const a = indices[i * 3], b = indices[i * 3 + 1], c = indices[i * 3 + 2];
    // Get face color (same for all 3 vertices of this face)
    let faceR = 0.55, faceG = 0.58, faceB = 0.62;
    if (hasColors && perFace) {
      faceR = pg.colors[i * 3];
      faceG = pg.colors[i * 3 + 1];
      faceB = pg.colors[i * 3 + 2];
    } else if (hasColors) {
      // per-vertex colors (legacy)
      faceR = pg.colors[a * 3];
      faceG = pg.colors[a * 3 + 1];
      faceB = pg.colors[a * 3 + 2];
    }
    for (let j = 0; j < 3; j++) {
      const vi = [a, b, c][j];
      positions[i * 9 + j * 3]     = verts[vi * 3];
      positions[i * 9 + j * 3 + 1] = verts[vi * 3 + 1];
      positions[i * 9 + j * 3 + 2] = verts[vi * 3 + 2];
      colors[i * 9 + j * 3]     = faceR;
      colors[i * 9 + j * 3 + 1] = faceG;
      colors[i * 9 + j * 3 + 2] = faceB;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();

  // Fillet/chamfer intent stage: split the mesh into the gray base
  // (rendered transparent) and the bright blend-surface patches
  // (rendered opaque on top, with depthTest off so fillets tucked under
  // the surface stay visible through the see-through body).
  const isIntentStage = hasColors && perFace && pg.stage && pg.stage.includes('intent');
  if (isIntentStage) {
    const basePos = [];
    const patchPos = [];
    const patchCol = [];
    const isGray = (r, g, b) =>
      Math.abs(r - 0.62) < 0.04 && Math.abs(g - 0.64) < 0.04 && Math.abs(b - 0.67) < 0.04;
    for (let i = 0; i < numFaces; i++) {
      const r = colors[i * 9], g = colors[i * 9 + 1], b = colors[i * 9 + 2];
      const tgt = isGray(r, g, b) ? basePos : patchPos;
      for (let k = 0; k < 9; k++) tgt.push(positions[i * 9 + k]);
      if (!isGray(r, g, b)) for (let k = 0; k < 9; k++) patchCol.push(colors[i * 9 + k]);
    }
    // Transparent base body
    if (basePos.length > 0) {
      const bGeo = new THREE.BufferGeometry();
      bGeo.setAttribute('position', new THREE.Float32BufferAttribute(basePos, 3));
      bGeo.computeVertexNormals();
      const bMat = new THREE.MeshStandardMaterial({
        color: new THREE.Color(0.55, 0.58, 0.62),
        metalness: 0.1, roughness: 0.7,
        transparent: true, opacity: 0.22,
        depthWrite: false, side: THREE.DoubleSide,
      });
      const bMesh = new THREE.Mesh(bGeo, bMat);
      modelGroup.add(bMesh);
    }
    // Opaque, glowing fillet surfaces — drawn on top so under-surface
    // fillets remain visible through the transparent body.
    if (patchPos.length > 0) {
      const pGeo = new THREE.BufferGeometry();
      pGeo.setAttribute('position', new THREE.Float32BufferAttribute(patchPos, 3));
      pGeo.setAttribute('color', new THREE.Float32BufferAttribute(patchCol, 3));
      pGeo.computeVertexNormals();
      const pMat = new THREE.MeshPhongMaterial({
        vertexColors: true,
        emissive: new THREE.Color(0.0, 0.4, 0.45),
        specular: new THREE.Color(0.2, 0.2, 0.2),
        shininess: 30, side: THREE.DoubleSide,
        depthTest: false, transparent: true, opacity: 0.95,
      });
      const pMesh = new THREE.Mesh(pGeo, pMat);
      pMesh.renderOrder = 2;
      modelGroup.add(pMesh);
    }
    document.getElementById('empty-state').style.display = 'none';
    modelGroup.position.set(0, 0, 0);
    try {
      const box = new THREE.Box3().setFromObject(modelGroup);
      modelGroup.position.sub(box.getCenter(new THREE.Vector3()));
    } catch (e) {}
    return;
  }

  // Check if this is a colored stage (segmentation, fitting, features)
  const isColoredStage = hasColors && pg.stage &&
    (pg.stage.includes('Segment') || pg.stage.includes('Fitting') ||
     pg.stage.includes('Recogniz') || pg.stage.includes('topology'));

  if (isColoredStage) {
    // Gray base mesh
    const baseMat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(0.52, 0.55, 0.58),
      metalness: 0.12,
      roughness: 0.65,
      side: THREE.FrontSide,
    });
    const baseMesh = new THREE.Mesh(geo, baseMat);
    baseMesh.castShadow = true;
    modelGroup.add(baseMesh);

    // Vivid color overlay with per-face vertex colors
    const overlayGeo = geo.clone();
    const overlayMat = new THREE.MeshPhongMaterial({
      vertexColors: true,
      emissive: new THREE.Color(0.15, 0.15, 0.15),
      specular: new THREE.Color(0.15, 0.15, 0.15),
      shininess: 25,
      transparent: false,
      side: THREE.FrontSide,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const overlay = new THREE.Mesh(overlayGeo, overlayMat);
    baseMesh.add(overlay);

    // Edge outlines
    const edges = new THREE.EdgesGeometry(geo, 8);
    const edgeMat = new THREE.LineBasicMaterial({
      color: new THREE.Color(0.06, 0.06, 0.08),
      opacity: 0.85,
      transparent: true,
      linewidth: 1,
    });
    baseMesh.add(new THREE.LineSegments(edges, edgeMat));
  } else {
    const matOpts = {
      metalness: 0.2,
      roughness: 0.6,
      side: THREE.FrontSide,
    };
    if (hasColors) matOpts.vertexColors = true;
    else matOpts.color = new THREE.Color(0.55, 0.58, 0.62);
    const mat = new THREE.MeshStandardMaterial(matOpts);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    modelGroup.add(mesh);
  }

  // Edge lines (for topology stage)
  if (pg.edges && pg.edges.length >= 6) {
    const edgeVerts = pg.edges;
    const edgeColors = pg.edge_colors;
    const pts = [];
    const eCols = [];
    for (let i = 0; i < edgeVerts.length; i += 3) {
      pts.push(new THREE.Vector3(edgeVerts[i], edgeVerts[i+1], edgeVerts[i+2]));
      if (edgeColors && edgeColors.length > i) {
        eCols.push(edgeColors[i], edgeColors[i+1], edgeColors[i+2]);
      }
    }
    const edgeGeo = new THREE.BufferGeometry().setFromPoints(pts);
    if (eCols.length > 0) {
      edgeGeo.setAttribute('color', new THREE.Float32BufferAttribute(eCols, 3));
    }
    const edgeMat = new THREE.LineBasicMaterial({
      vertexColors: eCols.length > 0,
      color: eCols.length > 0 ? 0xffffff : 0x58A6FF,
      linewidth: 1,
      depthTest: false,
    });
    const lines = new THREE.LineSegments(edgeGeo, edgeMat);
    lines.renderOrder = 1;
    modelGroup.add(lines);
  }

  // Center model
  modelGroup.position.set(0, 0, 0);
  try {
    const box = new THREE.Box3().setFromObject(modelGroup);
    const center = box.getCenter(new THREE.Vector3());
    modelGroup.position.sub(center);
  } catch (e) {}

  document.getElementById('empty-state').style.display = 'none';
}

// ── Live dual-pane build (right pane) ──
// While the pipeline runs, the left pane (#viewer) shows analysis stages on
// the original mesh; this module owns the right pane (#live-build-pane), where
// the CAD model accretes feature-by-feature as construction steps stream in
// from the build subprocess. Self-contained: its own renderer/scene/camera so
// it never touches the main viewer's state.
const LiveBuild = (() => {
  let pane, renderer, scene, camera, controls, group, sub, treeList;
  let started = false, frameDone = false, userOrbited = false, userPaused = false;

  // Live feature-tree rows use the SAME `.feature-item` markup + shared renderer
  // as the main sidebar tree, so the two trees are identical (user request).
  function _treeAdd(name, idx, volume) {
    if (!treeList) return;
    const row = document.createElement('div');
    row.className = 'feature-item';
    row.dataset.idx = idx;
    row.innerHTML = featureRowInnerHTML(name, volume);
    row.style.pointerEvents = 'auto';
    row.onclick = () => loadStepInPane(idx);
    treeList.querySelectorAll('.feature-item.active').forEach(r => r.classList.remove('active'));
    row.classList.add('active');
    treeList.appendChild(row);
    treeList.scrollTop = treeList.scrollHeight;
  }

  function _highlightRow(idx) {
    if (!treeList) return;
    treeList.querySelectorAll('.feature-item.active').forEach(r => r.classList.remove('active'));
    const row = treeList.querySelector('.feature-item[data-idx="' + idx + '"]');
    if (row) { row.classList.add('active'); row.scrollIntoView({ block: 'nearest' }); }
  }

  // Render a specific construction step into the pane (clicking a tree row).
  async function loadStepInPane(idx) {
    userPaused = true;   // stop the auto-advance so the picked step stays up
    let data = null;
    try {
      const res = await fetch('/api/job/' + curJob + '/step/' + idx);
      if (res.ok) data = await res.json();
    } catch (e) { return; }
    if (!data || data.error || !data.vertices || !data.vertices.length) return;
    if (data.is_sketch && !data.sketch_overlay) {
      _renderSketchLines(data);
    } else {
      try { _renderSolid(data, false); } catch (e) { return; }
      // A "Sketch X" step carries the solid-so-far PLUS the new sketch as a
      // blue overlay — draw it so clicking a sketch actually SHOWS the sketch,
      // not just the unchanged-looking solid.
      if (data.sketch_overlay && data.sketch_overlay.length >= 6) {
        _addSketchOverlay(data.sketch_overlay);
      }
      _addOverlayArcs(data.overlay_arcs);
    }
    _highlightRow(idx);
    if (sub) sub.textContent = (data.name || '') + '  ·  step ' + (idx + 1);
  }

  // Blue sketch lines on top of the current solid (overlay = flat array of
  // 3D points, consecutive pairs forming segments). Added to `group` so it
  // inherits the same centering offset as the solid.
  function _addSketchOverlay(overlay) {
    const pts = [];
    for (let i = 0; i + 2 < overlay.length; i += 3) {
      pts.push(new THREE.Vector3(overlay[i], overlay[i + 1], overlay[i + 2]));
    }
    if (pts.length < 2) return;
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    const mat = new THREE.LineBasicMaterial({ color: 0x58A6FF, depthTest: false });
    const lines = new THREE.LineSegments(geo, mat);
    lines.renderOrder = 3;
    group.add(lines);
  }

  // Rounded corners as TRUE arcs (one smooth THREE.Line per arc), not chords.
  function _addOverlayArcs(arcs) {
    if (!arcs || !arcs.length) return;
    const mat = new THREE.LineBasicMaterial({ color: 0x58A6FF, depthTest: false });
    for (const arc of arcs) {
      const pts = arcToPoints(arc);
      if (pts.length < 2) continue;
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat);
      line.renderOrder = 3;
      group.add(line);
    }
  }

  function _renderSketchLines(data) {
    _clearGroup();
    const _plS = sketchPlaneFromOverlay(data.vertices, data.overlay_arcs);
    if (_plS) _paneLastPlane = _plS;
    const verts = data.vertices, idxs = data.faces;
    const pts = [];
    for (let f = 0; f < idxs.length; f += 3) {
      const a = idxs[f], b = idxs[f + 1];
      pts.push(new THREE.Vector3(verts[a * 3], verts[a * 3 + 1], verts[a * 3 + 2]));
      pts.push(new THREE.Vector3(verts[b * 3], verts[b * 3 + 1], verts[b * 3 + 2]));
    }
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    group.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x58A6FF })));
    _addOverlayArcs(data.overlay_arcs);
    group.position.set(0, 0, 0);
    const box = new THREE.Box3().setFromObject(group);
    group.position.sub(box.getCenter(new THREE.Vector3()));
  }
  // Animation pump state: steps stream in (targetCount grows); the pump walks
  // shownIdx up to targetCount one step at a time with a delay, so features
  // visibly pop in one-by-one instead of jumping straight to the final solid.
  let targetCount = 0, shownIdx = -1, pumping = false, curJob = null;
  let _paneLastPlane = null;   // sketch plane feeding the feature animations
  const STEP_MS = 480;

  function _sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

  function ensure() {
    if (pane) return;
    pane = document.createElement('div');
    pane.id = 'live-build-pane';
    pane.innerHTML =
      '<div class="pane-label">Building model</div>' +
      '<div class="lbp-tree"><div class="lbp-tree-title">Feature tree</div>' +
      '<div class="lbp-tree-list"></div></div>' +
      '<div class="lbp-sub"></div>';
    document.body.appendChild(pane);
    sub = pane.querySelector('.lbp-sub');
    treeList = pane.querySelector('.lbp-tree-list');

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, 1, 0.1, 5000);
    camera.position.set(120, 90, 120);
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.localClippingEnabled = true;   // feature build animations
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.4;
    pane.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enableZoom = false;  // custom wheel zoom below (matches main viewer)

    // Explicit wheel-zoom: dolly the camera along the view direction. The main
    // viewer disables OrbitControls' built-in zoom and rolls its own; the live
    // pane needs the same or the scroll wheel does nothing here.
    renderer.domElement.addEventListener('wheel', (e) => {
      e.preventDefault();
      const dir = new THREE.Vector3().subVectors(camera.position, controls.target);
      let dist = dir.length() || 1;
      const sign = e.deltaY > 0 ? 1 : -1;
      dist = Math.max(1, Math.min(1e5, dist * (1 + sign * 0.1)));
      dir.setLength(dist);
      camera.position.copy(controls.target).add(dir);
      camera.updateProjectionMatrix();
      userOrbited = true;  // user took over zoom — stop auto-orienting
    }, { passive: false });
    controls.addEventListener('start', () => { userOrbited = true; });

    scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 0.9));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(80, 120, 100); scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.6);
    fill.position.set(-90, 40, -60); scene.add(fill);

    group = new THREE.Group();
    scene.add(group);

    (function loop() {
      requestAnimationFrame(loop);
      if (!pane || pane.style.display === 'none') return;
      controls.update();
      renderer.render(scene, camera);
    })();
  }

  function resize() {
    if (!renderer) return;
    const w = pane.clientWidth || 1, h = pane.clientHeight || 1;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function _clearGroup() {
    while (group.children.length) {
      const c = group.children[0];
      if (c.geometry) c.geometry.dispose();
      if (c.material) {
        if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
        else c.material.dispose();
      }
      group.remove(c);
    }
  }

  function _renderSolid(data, frame) {
    // Keep the previous solid aside so the new feature animates against it.
    const prevSolid = group.children.find(
      c => c.userData && c.userData.solidStep) || null;
    for (const c of group.children.slice()) {
      if (c === prevSolid) continue;
      if (c.geometry) c.geometry.dispose();
      if (c.material) {
        if (Array.isArray(c.material)) c.material.forEach(m => m.dispose());
        else c.material.dispose();
      }
      group.remove(c);
    }
    const verts = new Float32Array(data.vertices);
    const indices = data.faces;
    const numFaces = Math.floor(indices.length / 3);
    const positions = new Float32Array(numFaces * 9);
    for (let i = 0; i < numFaces; i++) {
      const a = indices[i*3], b = indices[i*3+1], c = indices[i*3+2];
      positions[i*9]   = verts[a*3]; positions[i*9+1] = verts[a*3+1]; positions[i*9+2] = verts[a*3+2];
      positions[i*9+3] = verts[b*3]; positions[i*9+4] = verts[b*3+1]; positions[i*9+5] = verts[b*3+2];
      positions[i*9+6] = verts[c*3]; positions[i*9+7] = verts[c*3+1]; positions[i*9+8] = verts[c*3+2];
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      color: 0xB8C0C8, metalness: 0.35, roughness: 0.45, side: THREE.FrontSide,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
    const solid = new THREE.Mesh(geo, mat);
    solid.userData.solidStep = true;
    group.add(solid);
    // Model centered at origin; frame the camera only when asked (first solid,
    // and again on expand-to-full) so the orbit isn't reset on every accretion.
    group.position.set(0, 0, 0);
    const box = new THREE.Box3().setFromObject(group);
    const center = box.getCenter(new THREE.Vector3());
    group.position.sub(center);
    if (frame) _reframe();
    // Feature build animation (0.25s) against the previous solid.
    const kind = classifyStepKind(data.name || '');
    if (data.sketch_overlay) {
      const pl = sketchPlaneFromOverlay(data.sketch_overlay, data.overlay_arcs);
      if (pl) _paneLastPlane = pl;
    }
    if (kind === 'sketch') {
      if (prevSolid) {
        if (prevSolid.geometry) prevSolid.geometry.dispose();
        if (prevSolid.material) prevSolid.material.dispose();
        group.remove(prevSolid);
      }
    } else {
      const effKind = (kind === 'revolve' || kind === 'fade') ? kind
        : (_paneLastPlane ? kind : 'fade');
      animateFeatureTransition(group, prevSolid, solid, effKind,
                               _paneLastPlane, renderer);
    }
  }

  function _reframe() {
    const box = new THREE.Box3().setFromObject(group);
    if (box.isEmpty()) return;
    const size = box.getSize(new THREE.Vector3());
    const r = Math.max(size.x, size.y, size.z) || 50;
    // Orient toward the broad face that has the MOST extrudes/bosses, so the
    // user sees the most features built. The plate's broad face is normal to
    // the thinnest bbox axis; we look along that axis from whichever side
    // carries more geometry (more vertices = the side with the bosses/screws).
    const dims = [size.x, size.y, size.z];
    let thin = 0;
    if (dims[1] < dims[thin]) thin = 1;
    if (dims[2] < dims[thin]) thin = 2;
    // Vote on +/- side of center along `thin` by vertex count.
    let plus = 0, minus = 0;
    const mesh = group.children.find(c => c.geometry && c.geometry.attributes && c.geometry.attributes.position);
    if (mesh) {
      const pos = mesh.geometry.attributes.position;
      const c = box.getCenter(new THREE.Vector3());
      const cAxis = [c.x, c.y, c.z][thin];
      for (let i = 0; i < pos.count; i++) {
        const val = pos.getComponent(i, thin) + mesh.position.getComponent(thin) - cAxis;
        if (val > 0) plus++; else minus++;
      }
    }
    const sign = (minus > plus) ? -1 : 1;  // camera on the busier side
    const eye = [0, 0, 0];
    const u = (thin + 1) % 3, v = (thin + 2) % 3;
    eye[thin] = sign * r * 1.7;     // mostly looking down the plate normal
    eye[u] = r * 0.55;              // slight tilt so it reads as 3D
    eye[v] = r * 0.45;
    camera.position.set(eye[0], eye[1], eye[2]);
    controls.target.set(0, 0, 0);
    camera.near = r * 0.01; camera.far = r * 50;
    camera.updateProjectionMatrix();
  }

  // Walk shownIdx up to targetCount, rendering one step at a time with a delay
  // so features accrete visibly. Re-enters safely as more steps stream in.
  async function _pump() {
    if (pumping) return;
    pumping = true;
    try {
      while (shownIdx + 1 < targetCount && curJob && !userPaused) {
        const idx = shownIdx + 1;
        let data = null;
        try {
          const res = await fetch(`/api/job/${curJob}/step/${idx}`);
          if (res.ok) data = await res.json();
        } catch (e) { /* transient — treat as empty */ }
        shownIdx = idx;
        if (!data || data.error || !data.vertices || data.vertices.length === 0) continue;
        _treeAdd(data.name, idx, data.volume);
        if (data.is_sketch && !data.sketch_overlay) {
          // Pure sketch (just construction lines): keep the prior solid on
          // screen, flash the label, advance quickly.
          if (sub) sub.textContent = (data.name || '') + '  ·  ' + (idx + 1) + '/' + targetCount;
          await _sleep(160);
          continue;
        }
        // Re-frame on EVERY new solid (until the user takes over) so the camera
        // keeps facing the busiest side as bosses accrete — the final view ends
        // up looking at the side with the MOST extrudes, not the base plate's
        // symmetric default.
        try { _renderSolid(data, !userOrbited); frameDone = true; }
        catch (e) { console.warn('[LiveBuild] render failed', e); continue; }
        // Sketch steps carry the solid-so-far PLUS the new profile as a blue
        // overlay (straight segments) + true arcs — draw them so the sketch
        // actually shows during the live accretion, not just the solid.
        if (data.sketch_overlay && data.sketch_overlay.length >= 6) {
          _addSketchOverlay(data.sketch_overlay);
        }
        _addOverlayArcs(data.overlay_arcs);
        if (sub) sub.textContent = (data.name || '') + '  ·  ' + (idx + 1) + '/' + targetCount;
        await _sleep(STEP_MS);
      }
    } finally { pumping = false; }
  }

  function update(jobId, numSteps, version) {
    ensure();
    curJob = jobId;
    if (numSteps > targetCount) targetCount = numSteps;
    _pump();
  }

  function show() {
    ensure();
    if (started && document.body.classList.contains('live-split')) return;
    started = true;
    document.body.classList.remove('live-full');
    document.body.classList.add('live-split');
    pane.style.display = 'block';
    resize();
    _pumpViewerResize();  // resize the LEFT (main) viewer through the glide
  }

  // On build completion: expand the right pane to fill the viewer and show the
  // FINISHED model immediately — NO reload/flash, and NO further accretion
  // animation. The accretion pump is stopped here so the build doesn't keep
  // "building" for several seconds after it's actually done.
  function expandToFull() {
    ensure();
    started = true;
    userPaused = true;   // stop the accretion pump — the build is complete
    document.body.classList.remove('live-split');
    document.body.classList.add('live-full');
    pane.style.display = 'block';
    const label = pane.querySelector('.pane-label');
    if (label) label.textContent = 'Parametric model';
    // Jump straight to the final construction step so the complete model shows
    // at once, instead of waiting for the pump to walk the remaining steps.
    if (curJob && targetCount > 0) {
      const lastIdx = targetCount - 1;
      fetch('/api/job/' + curJob + '/step/' + lastIdx)
        .then(r => r.ok ? r.json() : null)
        .then(data => {
          if (data && data.vertices && data.vertices.length) {
            try { _renderSolid(data, false); } catch (e) {}
            if (data.sketch_overlay && data.sketch_overlay.length >= 6) {
              _addSketchOverlay(data.sketch_overlay);
            }
            _addOverlayArcs(data.overlay_arcs);
          }
          if (sub) sub.textContent = 'complete · click a step to edit';
        })
        .catch(() => { if (sub) sub.textContent = 'complete · click a step to edit'; });
    } else if (sub) {
      sub.textContent = 'complete · click a step to edit';
    }
    // The model just SLIDES from the right half to full — keep the camera
    // exactly where the user left it (no reframe/jump), only resize the canvas
    // through the glide so it's the same view, bigger.
    _pumpViewerResize();
  }

  // Hand back to the interactive main viewer (user clicked a step / feature).
  function dismiss() {
    if (!started && !document.body.classList.contains('live-full')
        && !document.body.classList.contains('live-split')) return;
    // Hand the EXACT current view to the main viewer so revealing it shows the
    // same model from the same angle — no jump, the view just "stays put" while
    // the underlying canvas swaps.
    try { _syncMainCamera(camera.position, controls.target); } catch (e) {}
    started = false;
    document.body.classList.remove('live-split');
    document.body.classList.remove('live-full');
    if (pane) pane.style.display = 'none';
    _pumpViewerResize();
  }

  function isActive() {
    return document.body.classList.contains('live-full')
        || document.body.classList.contains('live-split');
  }

  function reset() {
    targetCount = 0; shownIdx = -1; frameDone = false; curJob = null;
    userOrbited = false; userPaused = false;
    if (group) _clearGroup();
    if (sub) sub.textContent = '';
    if (treeList) treeList.innerHTML = '';
    const label = pane && pane.querySelector('.pane-label');
    if (label) label.textContent = 'Building model';
  }

  window.addEventListener('resize', () => { if (started) resize(); });
  return { show, hide: dismiss, dismiss, update, reset, expandToFull, isActive, resize };
})();

// Keep both renderers sized correctly DURING the CSS split/expand transition
// (~0.45s) so the model glides from center to the left half instead of
// stretching and snapping.
function _pumpViewerResize(ms = 520) {
  const t0 = (window.performance || Date).now();
  (function step() {
    try { handleResize(); } catch (e) {}
    try { LiveBuild.resize(); } catch (e) {}
    if (((window.performance || Date).now() - t0) < ms) requestAnimationFrame(step);
  })();
}

// Copy a camera pose onto the MAIN viewer so handing off from the live-build
// pane reveals the identical view (both models are centered at the origin).
function _syncMainCamera(pos, target) {
  try {
    camera.position.copy(pos);
    controls.target.copy(target);
    camera.updateProjectionMatrix();
    controls.update();
  } catch (e) { /* main camera not ready */ }
}

// Clicking anywhere in the timeline or feature tree means the user wants the
// interactive main viewer — dismiss the expanded live-build pane to reveal it.
['timeline', 'feature-list'].forEach((id) => {
  const el = document.getElementById(id);
  if (el) el.addEventListener('click', () => { try { LiveBuild.dismiss(); } catch (e) {} }, true);
});

// ── Load step geometry ──
// ---------------------------------------------------------------------------
// Feature build animations (0.25s per feature): extrusions grow out of their
// sketch plane, cuts/holes vanish along the drill direction, revolves sweep
// around their axis, fillets/booleans cross-fade. Implemented with world-space
// clipping planes over the (previous solid, new solid) pair — the new solid is
// identical to the old outside the feature, so sweeping a clip plane reads as
// the feature itself growing/cutting.
const FEATURE_ANIM_MS = 250;
let _featAnim = null;          // { raf, cleanup }

function cancelFeatureAnim() {
  if (_featAnim) {
    try { _featAnim.cleanup(); } catch (e) {}
    if (_featAnim.raf) cancelAnimationFrame(_featAnim.raf);
    _featAnim = null;
  }
}

function classifyStepKind(name) {
  const n = name || '';
  if (/^sketch/i.test(n)) return 'sketch';
  if (/revolve/i.test(n)) return 'revolve';
  if (/hole|\(cut|^cut\b|pocket|slot|recess|oval cut|cutout/i.test(n)) return 'cut';
  if (/extrude|join|boss|pads|member|union/i.test(n)) return 'add';
  return 'fade';               // fillet / chamfer / boolean / refined
}

// Sketch plane from a step's overlay: the axis along which every overlay
// point is ~constant, in MODEL-LOCAL coordinates.
function sketchPlaneFromOverlay(overlayFlat, arcs) {
  try {
    if (overlayFlat && overlayFlat.length >= 9) {
      const n = Math.floor(overlayFlat.length / 3);
      const mins = [Infinity, Infinity, Infinity], maxs = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < n; i++) {
        for (let a = 0; a < 3; a++) {
          const v = overlayFlat[i * 3 + a];
          if (v < mins[a]) mins[a] = v;
          if (v > maxs[a]) maxs[a] = v;
        }
      }
      for (let a = 0; a < 3; a++) {
        if (maxs[a] - mins[a] < 0.5) return { axis: a, pos: (maxs[a] + mins[a]) / 2 };
      }
    }
    if (arcs && arcs.length && arcs[0].ax !== undefined) {
      return { axis: arcs[0].ax, pos: arcs[0].pos };
    }
  } catch (e) {}
  return null;
}

// Animate the transition prevMesh -> newMesh inside `group`. Both must already
// be children of the (possibly re-centred) group; on completion prevMesh is
// removed + disposed and all clipping cleared.
function animateFeatureTransition(group, prevMesh, newMesh, kind, plane, rendererRef) {
  cancelFeatureAnim();
  const disposePrev = () => {
    if (prevMesh && prevMesh.parent) prevMesh.parent.remove(prevMesh);
    if (prevMesh) {
      if (prevMesh.geometry) prevMesh.geometry.dispose();
      if (prevMesh.material) prevMesh.material.dispose();
    }
  };
  if (!newMesh) { disposePrev(); return; }
  // World-space offset of the model along each axis (group recentring).
  newMesh.updateWorldMatrix(true, false);
  const w = new THREE.Vector3();
  newMesh.getWorldPosition(w);
  const off = [w.x, w.y, w.z];
  if (!newMesh.geometry.boundingBox) newMesh.geometry.computeBoundingBox();
  const bb = newMesh.geometry.boundingBox;
  const mins = [bb.min.x, bb.min.y, bb.min.z];
  const maxs = [bb.max.x, bb.max.y, bb.max.z];

  const t0 = performance.now();
  const finish = () => {
    newMesh.material.clippingPlanes = null;
    newMesh.material.clipIntersection = false;
    newMesh.material.transparent = false;
    newMesh.material.opacity = 1.0;
    newMesh.material.needsUpdate = true;
    disposePrev();
    _featAnim = null;
  };

  if (kind === 'fade' || (!plane && kind !== 'revolve')) {
    newMesh.material.transparent = true;
    newMesh.material.opacity = 0.0;
    const tick = () => {
      const t = Math.min(1, (performance.now() - t0) / FEATURE_ANIM_MS);
      newMesh.material.opacity = t;
      if (t < 1) _featAnim.raf = requestAnimationFrame(tick);
      else finish();
    };
    _featAnim = { raf: requestAnimationFrame(tick), cleanup: finish };
    return;
  }

  if (kind === 'revolve') {
    // Wedge sweep about the model's thinnest axis through the bbox centre.
    const dims = [maxs[0] - mins[0], maxs[1] - mins[1], maxs[2] - mins[2]];
    let ax = 0;
    if (dims[1] < dims[ax]) ax = 1;
    if (dims[2] < dims[ax]) ax = 2;
    const u = (ax + 1) % 3, v = (ax + 2) % 3;
    const c = [(mins[0] + maxs[0]) / 2 + off[0], (mins[1] + maxs[1]) / 2 + off[1],
               (mins[2] + maxs[2]) / 2 + off[2]];
    const mkN = (cu, cv) => {
      const n = new THREE.Vector3();
      n.setComponent(u, cu);
      n.setComponent(v, cv);
      return n;
    };
    const p1 = new THREE.Plane(mkN(0, 1), 0);
    const p2 = new THREE.Plane(mkN(0, -1), 0);
    const cvec = new THREE.Vector3(c[0], c[1], c[2]);
    const setTheta = (th) => {
      p1.normal.copy(mkN(0, 1));
      p1.constant = -p1.normal.dot(cvec);
      p2.normal.copy(mkN(Math.sin(th), -Math.cos(th)));
      p2.constant = -p2.normal.dot(cvec);
      newMesh.material.clipIntersection = th > Math.PI;
    };
    newMesh.material.clippingPlanes = [p1, p2];
    setTheta(0.02);
    const tick = () => {
      const t = Math.min(1, (performance.now() - t0) / FEATURE_ANIM_MS);
      setTheta(Math.max(0.02, t * 2 * Math.PI * 0.999));
      if (t < 1) _featAnim.raf = requestAnimationFrame(tick);
      else finish();
    };
    _featAnim = { raf: requestAnimationFrame(tick), cleanup: finish };
    return;
  }

  // add / cut: linear clip sweep along the sketch plane's normal axis,
  // starting at the sketch side of the model.
  const ax = plane.axis;
  const lo = mins[ax], hi = maxs[ax];
  const fromLow = Math.abs(plane.pos - lo) <= Math.abs(plane.pos - hi);
  const start = (fromLow ? lo : hi) + off[ax];
  const end = (fromLow ? hi : lo) + off[ax];
  const e = new THREE.Vector3();
  e.setComponent(ax, fromLow ? 1 : -1);
  const clip = new THREE.Plane();
  const target = (kind === 'cut' && prevMesh) ? prevMesh : newMesh;
  if (kind === 'cut' && prevMesh) {
    // OLD material vanishes along the drill direction: keep the far side.
    clip.normal.copy(e);
    prevMesh.material.polygonOffset = true;
    prevMesh.material.polygonOffsetFactor = 0;   // draw over the new mesh
    prevMesh.material.polygonOffsetUnits = 0;
  } else {
    // NEW material appears along the extrude direction: keep the near side.
    clip.normal.copy(e).negate();
  }
  target.material.clippingPlanes = [clip];
  target.material.needsUpdate = true;
  const tick = () => {
    const t = Math.min(1, (performance.now() - t0) / FEATURE_ANIM_MS);
    const L = start + (end - start) * t;
    clip.constant = -(clip.normal.getComponent(ax)) * L;
    if (t < 1) _featAnim.raf = requestAnimationFrame(tick);
    else finish();
  };
  _featAnim = { raf: requestAnimationFrame(tick), cleanup: finish };
}

// Per-viewer sketch-plane memory (a feature step animates from the plane of
// the sketch step displayed just before it).
let _mainLastSketchPlane = null;
let _mainLastStepIndex = -2;

async function loadStep(index) {
  console.debug('[loadStep]', index, 'of', steps.length, 'jobId:', currentJobId);
  // Leave edge-pick mode if active (the model is about to be replaced).
  if (typeof _pick !== 'undefined' && _pick.active && !_pick.dragging) exitEdgePick();
  // Show feature tree when clicking a step
  document.getElementById('sidebar').classList.remove('closed');
  const url = currentJobId ? `/api/job/${currentJobId}/step/${index}` : `/api/step/${index}`;
  let res;
  try {
    res = await fetch(url);
  } catch (e) {
    console.error('[loadStep] network error fetching', url, e);
    return;
  }
  if (!res.ok) {
    console.error('[loadStep] HTTP', res.status, 'for', url);
    return;
  }
  let data;
  try {
    data = await res.json();
  } catch (e) {
    console.error('[loadStep] JSON parse failed for step', index, e);
    return;
  }
  if (data.error) {
    console.error('[loadStep] server error for step', index, data.error);
    return;
  }
  if (!data.vertices || data.vertices.length === 0) {
    console.warn('[loadStep] step', index, 'has no geometry');
    return;
  }

  // Clear previous model — dispose geometry/materials to free GPU memory.
  // The previous SOLID is kept aside so the new feature can animate against
  // it (0.25s grow/cut sweep); the animation disposes it when done.
  cancelFeatureAnim();
  const prevSolid = modelGroup.children.find(
    c => c.userData && c.userData.solidStep) || null;
  const disposed = modelGroup.children.length;
  for (const child of modelGroup.children.slice()) {
    if (child === prevSolid) continue;
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else child.material.dispose();
    }
    modelGroup.remove(child);
  }
  const dropPrevSolid = () => {
    if (!prevSolid) return;
    if (prevSolid.geometry) prevSolid.geometry.dispose();
    if (prevSolid.material) prevSolid.material.dispose();
    modelGroup.remove(prevSolid);
  };
  if (disposed > 0) console.debug('[loadStep] disposed', disposed, 'previous objects');

  try {
    const verts = new Float32Array(data.vertices);
    const indices = data.faces;

    if (data.is_sketch && !data.sketch_overlay) {
      dropPrevSolid();
      const _pl0 = sketchPlaneFromOverlay(
        Array.from(new Float32Array(data.vertices)), data.overlay_arcs);
      if (_pl0) _mainLastSketchPlane = _pl0;
      _mainLastStepIndex = index;
      console.debug('[loadStep] rendering sketch, edges:', Math.floor(indices.length / 3));
      // Draw as lines using face indices (each degenerate tri = one edge)
      const points = [];
      for (let f = 0; f < indices.length; f += 3) {
        const a = indices[f], b = indices[f + 1];
        points.push(
          new THREE.Vector3(verts[a*3], verts[a*3+1], verts[a*3+2]),
          new THREE.Vector3(verts[b*3], verts[b*3+1], verts[b*3+2]),
        );
      }
      const geo = new THREE.BufferGeometry().setFromPoints(points);
      const mat = new THREE.LineBasicMaterial({ color: 0x58A6FF, linewidth: 2 });
      modelGroup.add(new THREE.LineSegments(geo, mat));
      // Rounded corners as TRUE arcs (smooth curves), never chord polylines.
      if (data.overlay_arcs && data.overlay_arcs.length) {
        for (const arc of data.overlay_arcs) {
          const apts = arcToPoints(arc);
          if (apts.length < 2) continue;
          modelGroup.add(new THREE.Line(
            new THREE.BufferGeometry().setFromPoints(apts), mat));
        }
      }
    } else {
      console.debug('[loadStep] rendering solid mesh, verts:', Math.round(verts.length / 3), 'faces:', Math.round(indices.length / 3));
      // Build non-indexed geometry so each face has its own vertices
      // and correct per-face normals — eliminates Z-fighting artifacts
      // from BRepMesh's duplicate vertices at face boundaries.
      const geo = new THREE.BufferGeometry();
      const numFaces = Math.floor(indices.length / 3);
      const positions = new Float32Array(numFaces * 9);
      for (let i = 0; i < numFaces; i++) {
        const a = indices[i * 3], b = indices[i * 3 + 1], c = indices[i * 3 + 2];
        positions[i * 9]     = verts[a * 3];     positions[i * 9 + 1] = verts[a * 3 + 1]; positions[i * 9 + 2] = verts[a * 3 + 2];
        positions[i * 9 + 3] = verts[b * 3];     positions[i * 9 + 4] = verts[b * 3 + 1]; positions[i * 9 + 5] = verts[b * 3 + 2];
        positions[i * 9 + 6] = verts[c * 3];     positions[i * 9 + 7] = verts[c * 3 + 1]; positions[i * 9 + 8] = verts[c * 3 + 2];
      }
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      geo.computeVertexNormals();

      const mat = new THREE.MeshStandardMaterial({
        color: 0xB8C0C8,
        metalness: 0.35,
        roughness: 0.45,
        side: THREE.FrontSide,
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.userData.solidStep = true;
      modelGroup.add(mesh);

      // Feature build animation (0.25s): grow/cut/revolve against the
      // previous solid. Sketch steps just record their plane; a jump to a
      // non-adjacent step cross-fades (the cached plane belongs elsewhere).
      const stepName = data.name
        || (typeof steps !== 'undefined' && steps[index] && steps[index].name) || '';
      const kind = classifyStepKind(stepName);
      if (data.sketch_overlay) {
        const _pl = sketchPlaneFromOverlay(data.sketch_overlay, data.overlay_arcs);
        if (_pl) _mainLastSketchPlane = _pl;
      }
      if (kind === 'sketch') {
        dropPrevSolid();
      } else {
        const seq = index === _mainLastStepIndex + 1;
        const plane = seq ? _mainLastSketchPlane : null;
        const effKind = (kind === 'revolve' || kind === 'fade') ? kind
          : (plane ? kind : 'fade');
        animateFeatureTransition(modelGroup, prevSolid, mesh, effKind, plane, renderer);
      }
      _mainLastStepIndex = index;

      // If this step has a sketch overlay, draw visible blue edges on top of the solid
      if (data.sketch_overlay && data.sketch_overlay.length >= 6) {
        const skVerts = data.sketch_overlay;
        const pts = [];
        for (let i = 0; i < skVerts.length; i += 3) {
          pts.push(new THREE.Vector3(skVerts[i], skVerts[i+1], skVerts[i+2]));
        }

        // Draw each edge as a thin cylinder tube so it's visible at any zoom
        const tubeMat = new THREE.MeshBasicMaterial({ color: 0x58A6FF, depthTest: false });
        for (let i = 0; i < pts.length; i += 2) {
          const a = pts[i], b = pts[i+1];
          const dir = new THREE.Vector3().subVectors(b, a);
          const len = dir.length();
          if (len < 0.01) continue;
          const tubeR = Math.max(0.3, len * 0.008);
          const tubeGeo = new THREE.CylinderGeometry(tubeR, tubeR, len, 6, 1);
          tubeGeo.rotateX(Math.PI / 2);
          const tube = new THREE.Mesh(tubeGeo, tubeMat);
          tube.position.copy(a).add(b).multiplyScalar(0.5);
          tube.lookAt(b);
          tube.renderOrder = 1;
          modelGroup.add(tube);
        }

        // (Removed: the old code drew a translucent triangle from the first
        // three overlay vertices as a visual cue. For rectangular/polygonal
        // sketches those first three vertices are corners on the sketch
        // outline, and the triangle cut across the interior — showing up
        // in the viewer as spurious "spikes" or bumps crossing the sketch.)

        console.debug('[loadStep] added sketch overlay:', pts.length / 2, 'edges');
      }

      // Rounded corners as TRUE arcs: sample each into a smooth polyline and
      // draw thin tubes along it — so curves read as curves, never chords.
      if (data.overlay_arcs && data.overlay_arcs.length) {
        const arcTubeMat = new THREE.MeshBasicMaterial({ color: 0x58A6FF, depthTest: false });
        for (const arc of data.overlay_arcs) {
          const apts = arcToPoints(arc);
          for (let i = 0; i + 1 < apts.length; i++) {
            const a = apts[i], b = apts[i + 1];
            const len = a.distanceTo(b);
            if (len < 0.01) continue;
            const tubeR = Math.max(0.3, len * 0.04);
            const tubeGeo = new THREE.CylinderGeometry(tubeR, tubeR, len, 6, 1);
            tubeGeo.rotateX(Math.PI / 2);
            const tube = new THREE.Mesh(tubeGeo, arcTubeMat);
            tube.position.copy(a).add(b).multiplyScalar(0.5);
            tube.lookAt(b);
            tube.renderOrder = 1;
            modelGroup.add(tube);
          }
        }
      }
    }
  } catch (e) {
    console.error('[loadStep] geometry/material creation failed for step', index, e);
    return;
  }

  // Center model at origin — reset position first to avoid cumulative drift
  try {
    modelGroup.position.set(0, 0, 0);
    const box = new THREE.Box3().setFromObject(modelGroup);
    const center = box.getCenter(new THREE.Vector3());
    modelGroup.position.sub(center);
  } catch (e) {
    console.error('[loadStep] bounding box calculation failed for step', index, e);
  }

  // Keep camera where it is — don't animate on step clicks or reloads
  controls.target.set(0, 0, 0);
  controls.update();

  // Update UI
  currentStep = index;
  localStorage.setItem('paramesh_last_step', index.toString());
  console.debug('[loadStep] step loaded:', index, 'name:', data.name, 'verts:', Math.round(data.vertices.length / 3));
  updateActiveStates();

  // Show step name toast briefly
  let toast = document.getElementById('step-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'step-toast';
    toast.style.cssText = 'position:fixed;top:60px;left:50%;transform:translateX(-50%);background:var(--bg-secondary);border:1px solid var(--border);border-radius:8px;padding:8px 16px;font-size:13px;color:var(--text-primary);z-index:100;opacity:0;transition:opacity 0.3s;pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,0.1);';
    document.body.appendChild(toast);
  }
  const toastName = data.name.split(/\s*[—→]\s*/)[0].trim();
  toast.textContent = `Step ${index + 1}/${steps.length}: ${toastName}`;
  toast.style.opacity = '1';
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { toast.style.opacity = '0'; }, 1500);
  document.getElementById('empty-state').style.display = 'none';
}

window._loadStep = loadStep;

// ── Build Animation ──
let _animPlaying = false;
window.playBuildAnimation = async function() {
  const btn = document.getElementById('play-btn');
  if (!btn) return;

  // Toggle: if already playing, stop
  if (_animPlaying) {
    _animPlaying = false;
    return;
  }

  if (!steps || steps.length === 0) return;

  // If this mesh's Onshape document is already open, the live Onshape build
  // DRIVES this animation (it advances loadStep as each feature lands) so
  // the model rebuilds in Paramesh and in the already-open Onshape tab at
  // once. No NEW tab is opened here — that's what "Open in Onshape" is for;
  // Play just re-syncs the document the user already opened.
  if (_boundForCurrentMesh && _boundForCurrentMesh()
      && _onshapeTarget && _onshapeTarget.opened
      && _onshapeStatus && _onshapeStatus.connected) {
    _runOnshapeBuild(null);
    return;
  }

  // The play animation renders into the MAIN viewer via loadStep(). After a
  // build the viewer is in live-full mode with the main viewer HIDDEN behind
  // the LiveBuild pane — so loadStep would draw invisibly (label advances, no
  // visual). Hand off to the main viewer first so the replay is actually seen.
  try { LiveBuild.dismiss(); } catch (e) {}

  _animPlaying = true;
  window._pipelineRotate = true;
  btn.classList.add('playing');
  btn.querySelector('i').className = 'ri-stop-fill';

  const stepDelay = 350; // 350ms per feature

  for (let i = 0; i < steps.length; i++) {
    if (!_animPlaying) break;
    await loadStep(i);
    await new Promise(r => setTimeout(r, stepDelay));
  }

  _animPlaying = false;
  window._pipelineRotate = false;
  modelGroup.rotation.y = 0;
  btn.classList.remove('playing');
  btn.querySelector('i').className = 'ri-play-fill';
};

// ── UI Updates ──
function updateActiveStates() {
  document.querySelectorAll('.feature-item').forEach((el, i) => {
    el.classList.toggle('active', i === currentStep);
  });
  // Skip .timeline-special buttons (Mesh, Compare) when highlighting
  document.querySelectorAll('.timeline-step:not(.timeline-special)').forEach((el, i) => {
    el.classList.toggle('active', i === currentStep);
  });
  // Deactivate special buttons
  document.querySelectorAll('.timeline-special').forEach(el => el.classList.remove('active'));
  // NOTE: the right-sidebar Stages cards are highlighted by the stages module
  // itself (selectStage / _setActiveStageCard), not here — they no longer map
  // 1:1 to construction steps (the whole build is a single "Final model" card).
}

// ─── Live stages panel (right sidebar, under Dimensions) ─────────────────────
// One card per VISUAL STAGE that displays on the left during a run: the analysis
// stages (segmentation, topology, … captured from progress_geometry), then a
// SINGLE card for the whole CAD build — "Build model" while building, relabelled
// "Final model" when done. (We deliberately do NOT show one card per applied
// feature — the bottom timeline / feature tree already do that.)
//
// FOLLOW vs PINNED. While "following" (the default) the panel tracks the live
// run: the status poll renders the latest stage on the left and the list
// auto-scrolls to the newest card. Clicking an OLDER stage PINS it — the poll
// stops overwriting the left pane (`stagesFollowing()` gate) AND, once the build
// pane has expanded to full screen, the pinned stage is shown in the main screen
// (LiveBuild is dismissed to reveal it). It stays until you click that card
// again (toggle off) or click the "Final/Build model" card at the bottom, which
// resumes follow and returns to the final model. Auto-scroll is suppressed while
// pinned so a manual scroll-up isn't yanked back.
let _stages = [];            // [{kind:'analysis'|'build'|'final', label, pg?, el}]
let _stageFollow = true;     // track the live run/final model vs. a pinned stage
let _stagesJobId = null;
let _currentStageLabel = null;  // the stage running right now (highlight + loader)
let _stageNumSteps = 0;      // latest construction step count (for "Final model")

function stagesFollowing() { return _stageFollow; }

function resetStagesPanel() {
  _stages = []; _stageFollow = true; _stagesJobId = null;
  _currentStageLabel = null; _stageNumSteps = 0;
  const list = document.getElementById('stages-list');
  const panel = document.getElementById('stages-panel');
  if (list) list.innerHTML = '';
  if (panel) panel.style.display = 'none';
  const cnt = document.getElementById('stages-count');
  if (cnt) cnt.textContent = '';
}

function _addStageCard(stage, idx) {
  const list = document.getElementById('stages-list');
  if (!list) return;
  const card = document.createElement('div');
  card.className = 'stage-card';
  const label = stage.label || ('Stage ' + (idx + 1));
  const safe = label.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  card.innerHTML =
    '<div class="stage-idx">' + (idx + 1) + '</div>' +
    '<div class="stage-name" title="' + safe + '">' + safe + '</div>' +
    '<i class="ri-check-line stage-tick"></i>' +
    '<div class="stage-loader"></div>';
  card.onclick = () => selectStage(idx);
  stage.el = card;
  list.appendChild(card);
  if (_stageFollow) list.scrollTop = list.scrollHeight;  // only auto-scroll while following
}

function _setActiveStageCard(ci) {
  _stages.forEach((s, i) => { if (s.el) s.el.classList.toggle('active', i === ci); });
}

function _stageSetLabel(stage, label) {
  if (stage.label === label) return;
  stage.label = label;
  if (stage.el) {
    const nm = stage.el.querySelector('.stage-name');
    if (nm) { nm.textContent = label; nm.title = label; }
  }
}

// Index of the card for the stage currently running (by label).
function _currentStageIdx() {
  if (_currentStageLabel == null) return _stages.length - 1;
  const i = _stages.findIndex(s => s.label === _currentStageLabel);
  return i >= 0 ? i : _stages.length - 1;
}

// Resume following the live run / return to the final model.
function _stageResumeFollow() {
  _stageFollow = true;
  const finalIdx = _stages.findIndex(s => s.kind === 'final');
  if (finalIdx >= 0 && _stageNumSteps > 0) {
    // Build finished → reveal the final model in the interactive main viewer.
    try { LiveBuild.dismiss(); } catch (e) {}
    loadStep(_stageNumSteps - 1);
    _setActiveStageCard(finalIdx);
  } else {
    // Still running → highlight + re-show the current stage; the poll keeps it advancing.
    const ci = _currentStageIdx();
    const cur = _stages[ci];
    _setActiveStageCard(ci);
    if (cur && cur.pg) {
      window._explodePiecesVersion = null;
      try { _renderProgressGeometry(cur.pg); } catch (e) {}
    }
  }
}

// Click a stage card. The "Final model" card, the current/live stage, or
// re-clicking the active pinned card RESUME follow; clicking any other stage
// that HAS a stored visual PINS it into the main screen.
function selectStage(ci) {
  const stage = _stages[ci];
  if (!stage) return;
  const isCurrent = (ci === _currentStageIdx());
  const alreadyActive = !!(stage.el && stage.el.classList.contains('active'));
  if (stage.kind === 'final' || isCurrent || alreadyActive || !stage.pg) {
    _stageResumeFollow();
    return;
  }
  // Pin this stage: reveal the main viewer (in case the build pane expanded to
  // full) and render the stored snapshot there. Reset the pieces version guard
  // so _renderProgressGeometry doesn't early-return.
  _stageFollow = false;
  try { LiveBuild.dismiss(); } catch (e) {}
  window._explodePiecesVersion = null;
  try { _renderProgressGeometry(stage.pg); } catch (e) { console.warn('[stage] render failed', e); }
  _setActiveStageCard(ci);
}

function updateStagesPanel(jobId, d) {
  const panel = document.getElementById('stages-panel');
  const list = document.getElementById('stages-list');
  if (!panel || !list) return;
  if (jobId !== _stagesJobId) {            // new run → start fresh
    _stagesJobId = jobId; _stages = []; _currentStageLabel = null;
    _stageFollow = true; _stageNumSteps = 0; list.innerHTML = '';
  }
  _stageNumSteps = (d.num_steps || (d.steps || []).length) || _stageNumSteps;

  // ONE card per UNIQUE pipeline STAGE NAME — exactly what the loading text /
  // left panel shows (segmentation, fitting, topology, recognizing, blend,
  // stitch, …). Driven purely by the stage NAME, never by build-state: on a
  // multi-body part the CAD build of body 1 streams (building_live / num_steps)
  // WHILE later bodies are still in topology/recognize, so build-state would
  // wrongly cut the list off. The per-feature construction STEPS (d.steps) are
  // deliberately NOT carded. "Building CAD model" collapses to one "Build model"
  // card. The geometry snapshot is stored when present so a click re-shows it.
  const raw = d.stage || (d.progress_geometry && d.progress_geometry.stage) || '';
  const isCad = /cad model/i.test(raw);
  const lbl = isCad ? 'Build model' : raw;
  if (lbl && d.status === 'processing') {
    let card = _stages.find(s => s.label === lbl);
    if (!card) {
      card = { kind: isCad ? 'build' : 'analysis', label: lbl, pg: d.progress_geometry || null };
      _stages.push(card); _addStageCard(card, _stages.length - 1);
    } else if (d.progress_geometry) {
      card.pg = d.progress_geometry;       // refresh snapshot for re-show
    }
    _currentStageLabel = lbl;
  }

  // "Final model" — a dedicated card at the BOTTOM once the build is done; this
  // is the default view and the "return to the final model" target.
  if (d.status === 'done' && _stageNumSteps > 0
      && !_stages.some(s => s.kind === 'final')) {
    const fin = { kind: 'final', label: 'Final model', pg: null };
    _stages.push(fin); _addStageCard(fin, _stages.length - 1);
    _currentStageLabel = 'Final model';
  }

  if (_stages.length > 0) panel.style.display = 'flex';
  const cnt = document.getElementById('stages-count');
  if (cnt) cnt.textContent = _stages.length ? (_stages.length + ' stage' + (_stages.length > 1 ? 's' : '')) : '';

  // Loading bar on the CURRENT stage's card while the pipeline is still working.
  const building = (d.status === 'processing' || d.status === 'queued') && !d.paused;
  const ci = _currentStageIdx();
  _stages.forEach((s, i) => { if (s.el) s.el.classList.toggle('loading', building && i === ci); });

  // While following, highlight the CURRENT stage card (it's what's on the left).
  if (_stageFollow && _stages.length) _setActiveStageCard(ci);
}
window.toggleStagesPanel = function() {
  document.getElementById('stages-panel').classList.toggle('collapsed');
};

// ─── Feature-tree editor (StudioCAD-style) ───────────────────────────────────
// Top bar adds finishing features (fillet/chamfer) and the feature list lets you
// mark recognized features for deletion; "Rebuild" POSTs the full edit state to
// /api/job/{id}/edit, which re-runs the CadQuery build and returns a new step.
window._editState = { deletes: new Set(), finishes: [] };
let _treeFeatures = [];

function setupEditToolbar(statusData) {
  const bar = document.getElementById('edit-toolbar');
  if (!bar) return;
  const editable = !!statusData.editable && statusData.status === 'done';
  bar.style.display = editable ? 'flex' : 'none';
  if (!editable) { document.getElementById('feature-editor').style.display = 'none'; return; }
  _treeFeatures = statusData.tree_features || [];
  // Reset edit state for a fresh build (new job_id) so stale edits don't leak.
  if (bar.dataset.job !== statusData.job_id) {
    bar.dataset.job = statusData.job_id;
    window._editState = { deletes: new Set(), finishes: [] };
    document.getElementById('feature-editor').style.display = 'none';
  }
  renderEditChips();
}

window.addFinish = function (kind) {
  const dflt = kind === 'fillet' ? '2' : '1';
  const raw = prompt(`${kind === 'fillet' ? 'Fillet radius' : 'Chamfer distance'} (mm):`, dflt);
  if (raw === null) return;
  const v = parseFloat(raw);
  if (!(v > 0)) { alert('Enter a positive number.'); return; }
  window._editState.finishes.push({ type: kind, value: v });
  renderEditChips();
};

window.toggleFeatureEditor = function () {
  const fe = document.getElementById('feature-editor');
  const show = fe.style.display === 'none';
  fe.style.display = show ? 'block' : 'none';
  if (show) renderFeatureEditor();
};

function renderFeatureEditor() {
  const list = document.getElementById('fe-list');
  list.innerHTML = '';
  if (!_treeFeatures.length) { list.innerHTML = '<div style="color:var(--text-muted);font-size:12px;">No editable features.</div>'; return; }
  _treeFeatures.forEach(f => {
    const row = document.createElement('div');
    const del = window._editState.deletes.has(f.index);
    row.className = 'fe-item' + (del ? ' del' : '');
    const info = featureIconInfo(f.type);
    row.innerHTML = `<span class="fe-idx">${f.index}</span>` +
      `<img src="/static/icons/${info.file}" style="width:20px;height:20px;border-radius:4px;">` +
      `<span>${f.type}${f.depth ? ' · ' + f.depth + 'mm' : ''}</span>` +
      `<i class="fe-x ri-${del ? 'arrow-go-back-line' : 'delete-bin-line'}"></i>`;
    row.onclick = () => { toggleDeleteFeature(f.index); };
    list.appendChild(row);
  });
}

function toggleDeleteFeature(idx) {
  const d = window._editState.deletes;
  if (d.has(idx)) d.delete(idx); else d.add(idx);
  renderFeatureEditor();
  renderEditChips();
}

function renderEditChips() {
  const chips = document.getElementById('et-chips');
  if (!chips) return;
  chips.innerHTML = '';
  window._editState.finishes.forEach((f, i) => {
    const c = document.createElement('span');
    c.className = 'et-chip';
    c.innerHTML = `<span><b>${f.type === 'fillet' ? 'Fillet' : 'Chamfer'}</b> ${f.value}mm</span><span class="x">&times;</span>`;
    c.querySelector('.x').onclick = () => { window._editState.finishes.splice(i, 1); renderEditChips(); };
    chips.appendChild(c);
  });
  const nDel = window._editState.deletes.size;
  if (nDel) {
    const c = document.createElement('span');
    c.className = 'et-chip';
    c.style.borderColor = '#F85149';
    c.innerHTML = `<span><b>${nDel}</b> deleted</span>`;
    chips.appendChild(c);
  }
}

window.resetEdits = function () {
  window._editState = { deletes: new Set(), finishes: [] };
  renderEditChips();
  document.getElementById('feature-editor').style.display = 'none';
  document.getElementById('et-status').textContent = '';
  // Reload the original final model.
  if (steps && steps.length) loadStep(steps.length - 1);
};

window.applyEdits = async function () {
  if (!currentJobId) { alert('Build a model first.'); return; }
  const es = window._editState;
  if (!es.finishes.length && !es.deletes.size) { document.getElementById('et-status').textContent = 'No edits to apply.'; return; }
  const btn = document.getElementById('et-rebuild');
  const status = document.getElementById('et-status');
  btn.disabled = true;
  status.innerHTML = '<i class="ri-loader-4-line fspin"></i> Rebuilding…';
  try {
    const res = await fetch(`/api/job/${currentJobId}/edit`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ deletes: [...es.deletes], finishes: es.finishes }),
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      status.textContent = '';
      alert('Rebuild failed: ' + (data.error || res.status) + (data.detail ? '\n' + data.detail : ''));
      return;
    }
    await loadStep(data.step_index);
    const warn = (data.warnings || []).length ? ' (' + data.warnings.join('; ') + ')' : '';
    const napp = (data.applied || []).length;
    status.textContent = `✓ Rebuilt · ${napp} finish${napp === 1 ? '' : 'es'}${es.deletes.size ? ', ' + es.deletes.size + ' deleted' : ''}${warn}`;
  } catch (e) {
    console.error('[edit] rebuild error', e);
    status.textContent = '';
    alert('Rebuild error: ' + e.message);
  } finally {
    btn.disabled = false;
  }
};

window.downloadEditedStep = function () {
  if (!currentJobId) return;
  window.location.href = `/api/job/${currentJobId}/download-step`;
};

// ─── Per-edge fillet/chamfer picking: hover-highlight + drag-arrow ────────────
const _pick = { active: false, kind: null, edges: [], lines: [], group: null,
  hovered: -1, selected: -1, arrow: null, dragging: false, radius: 0,
  maxR: 5, diag: 10, mid: null, dir: null, center: [0, 0, 0] };
const _pickRay = new THREE.Raycaster();

function _pickXY(ev) {
  const r = renderer.domElement.getBoundingClientRect();
  return new THREE.Vector2(
    ((ev.clientX - r.left) / r.width) * 2 - 1,
    -((ev.clientY - r.top) / r.height) * 2 + 1);
}

window.startEdgePick = async function (kind) {
  if (!currentJobId) { alert('Build a model first.'); return; }
  exitEdgePick();
  const status = document.getElementById('et-status');
  status.innerHTML = '<i class="ri-loader-4-line fspin"></i> Loading edges…';
  let data;
  try { data = await (await fetch(`/api/job/${currentJobId}/edges`)).json(); }
  catch (e) { status.textContent = ''; alert('Could not load edges.'); return; }
  if (!data.ok || !(data.edges || []).length) {
    status.textContent = ''; alert(data.error || 'No edges to pick.'); return;
  }
  _pick.active = true; _pick.kind = kind; _pick.edges = data.edges;
  const lo = data.bbox_min, hi = data.bbox_max;
  _pick.diag = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]) || 10;
  _pick.maxR = _pick.diag * 0.18;
  _pick.center = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2];
  _pickRay.params.Line.threshold = _pick.diag * 0.012;

  const grp = new THREE.Group();
  _pick.lines = [];
  data.edges.forEach((e, i) => {
    const pts = e.pts.map(p => new THREE.Vector3(p[0], p[1], p[2]));
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints(pts),
      new THREE.LineBasicMaterial({ color: 0x3FA7FF, transparent: true, opacity: 0.4, depthTest: false }));
    line.renderOrder = 4; line.userData.ei = i;
    grp.add(line); _pick.lines.push(line);
  });
  _pick.group = grp; modelGroup.add(grp);

  setPickHint(`Hover an edge — click to place a ${kind}, then drag the arrow. Esc to cancel.`);
  status.textContent = '';
  const dom = renderer.domElement;
  dom.addEventListener('pointermove', onPickMove);
  dom.addEventListener('pointerdown', onPickDown, true);
  window.addEventListener('keydown', onPickKey);
  document.body.style.cursor = 'crosshair';
};

function exitEdgePick() {
  if (_pick.group) { modelGroup.remove(_pick.group); _pick.group.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); }); }
  removeArrow();
  const dom = renderer.domElement;
  dom.removeEventListener('pointermove', onPickMove);
  dom.removeEventListener('pointerdown', onPickDown, true);
  window.removeEventListener('keydown', onPickKey);
  window.removeEventListener('pointerup', endArrowDrag);
  Object.assign(_pick, { active: false, kind: null, edges: [], lines: [], group: null, hovered: -1, selected: -1, dragging: false });
  if (controls) controls.enabled = true;
  document.body.style.cursor = '';
  setPickHint(null);
  document.getElementById('pick-radius-label').style.display = 'none';
}
window.exitEdgePick = exitEdgePick;

function onPickKey(ev) { if (ev.key === 'Escape') exitEdgePick(); }

function setPickHint(txt) {
  const h = document.getElementById('pick-hint');
  if (!h) return;
  if (txt) { h.textContent = txt; h.style.display = 'block'; } else h.style.display = 'none';
}

function _styleLine(i, mode) {
  const l = _pick.lines[i]; if (!l) return;
  if (mode === 'hover') { l.material.color.set(0xFFC83D); l.material.opacity = 1; }
  else if (mode === 'sel') { l.material.color.set(0x3DDC84); l.material.opacity = 1; }
  else { l.material.color.set(0x3FA7FF); l.material.opacity = 0.4; }
}

function onPickMove(ev) {
  if (!_pick.active) return;
  if (_pick.dragging) { onArrowDrag(ev); return; }
  _pickRay.setFromCamera(_pickXY(ev), camera);
  const hits = _pickRay.intersectObjects(_pick.lines, false);
  const ei = hits.length ? hits[0].object.userData.ei : -1;
  if (ei === _pick.hovered) return;
  if (_pick.hovered >= 0 && _pick.hovered !== _pick.selected) _styleLine(_pick.hovered, 'base');
  _pick.hovered = ei;
  if (ei >= 0 && ei !== _pick.selected) _styleLine(ei, 'hover');
  document.body.style.cursor = ei >= 0 ? 'pointer' : 'crosshair';
}

function onPickDown(ev) {
  if (!_pick.active || ev.button !== 0) return;
  // Clicking the arrow handle starts a radius drag.
  if (_pick.arrow) {
    _pickRay.setFromCamera(_pickXY(ev), camera);
    if (_pickRay.intersectObject(_pick.arrow, true).length) {
      ev.preventDefault(); ev.stopPropagation();
      startArrowDrag(); return;
    }
  }
  if (_pick.hovered >= 0) {
    ev.preventDefault(); ev.stopPropagation();
    selectEdge(_pick.hovered);
  }
}

function selectEdge(ei) {
  if (_pick.selected >= 0 && _pick.selected !== ei) _styleLine(_pick.selected, 'base');
  _pick.selected = ei; _styleLine(ei, 'sel');
  const e = _pick.edges[ei];
  _pick.mid = e.mid;
  const M = new THREE.Vector3(...e.mid);
  const C = new THREE.Vector3(..._pick.center);
  let D = M.clone().sub(C);
  if (D.lengthSq() < 1e-6) D.set(0, 0, 1);
  D.normalize();
  _pick.dir = D;
  _pick.radius = Math.max(0.5, Math.round(_pick.diag * 0.02 * 2) / 2);
  buildArrow();
  setPickHint(`Drag the arrow to set the ${_pick.kind} size, then release to apply. Esc to cancel.`);
}

function buildArrow() {
  removeArrow();
  const M = new THREE.Vector3(..._pick.mid);
  const arrow = new THREE.ArrowHelper(_pick.dir, M, _arrowLen(), 0x3DDC84,
    _arrowLen() * 0.35, _pick.diag * 0.03);
  arrow.line.material.depthTest = false;
  arrow.cone.material.depthTest = false;
  arrow.renderOrder = 6;
  // Fatten the shaft for easier grabbing.
  arrow.line.material.linewidth = 3;
  _pick.arrow = arrow; modelGroup.add(arrow);
  updateRadiusLabel();
}

function _arrowLen() { return Math.max(_pick.diag * 0.04, _pick.radius * 2.2); }

function updateArrow() {
  if (!_pick.arrow) return;
  _pick.arrow.setLength(_arrowLen(), _arrowLen() * 0.35, _pick.diag * 0.03);
  updateRadiusLabel();
}

function removeArrow() {
  if (_pick.arrow) {
    modelGroup.remove(_pick.arrow);
    _pick.arrow.traverse(o => { if (o.geometry) o.geometry.dispose(); if (o.material) o.material.dispose(); });
    _pick.arrow = null;
  }
}

function updateRadiusLabel() {
  const lbl = document.getElementById('pick-radius-label');
  if (!lbl || _pick.selected < 0) { if (lbl) lbl.style.display = 'none'; return; }
  // Project the arrow tip to screen.
  const tip = new THREE.Vector3(..._pick.mid).add(_pick.dir.clone().multiplyScalar(_arrowLen()));
  tip.project(camera);
  const r = renderer.domElement.getBoundingClientRect();
  lbl.style.left = (r.left + (tip.x * 0.5 + 0.5) * r.width) + 'px';
  lbl.style.top = (r.top + (-tip.y * 0.5 + 0.5) * r.height) + 'px';
  const rtxt = _pick.radius % 1 === 0 ? String(_pick.radius) : _pick.radius.toFixed(1);
  lbl.textContent = `${_pick.kind === 'fillet' ? 'R' : 'C'} ${rtxt} mm`;
  lbl.style.display = 'block';
}

function startArrowDrag() {
  _pick.dragging = true;
  if (controls) controls.enabled = false;
  document.body.style.cursor = 'ns-resize';
  window.addEventListener('pointerup', endArrowDrag);
}

function onArrowDrag(ev) {
  _pickRay.setFromCamera(_pickXY(ev), camera);
  // Closest param along the axis (M + t·D) to the camera ray → radius.
  const M = new THREE.Vector3(..._pick.mid), D = _pick.dir;
  const O = _pickRay.ray.origin, Rd = _pickRay.ray.direction;
  const w0 = new THREE.Vector3().subVectors(M, O);
  const a = Rd.dot(Rd), b = Rd.dot(D), c = D.dot(D), d = Rd.dot(w0), e = D.dot(w0);
  const den = a * c - b * b;
  let t = Math.abs(den) < 1e-9 ? e / c : (a * e - b * d) / den;
  // Snap to whole / half-whole millimetres (0.5 increments), min 0.5.
  const raw = Math.min(_pick.maxR, Math.max(0.5, t / 2.2));
  _pick.radius = Math.max(0.5, Math.round(raw * 2) / 2);
  updateArrow();
}

function endArrowDrag() {
  if (!_pick.dragging) return;
  _pick.dragging = false;
  window.removeEventListener('pointerup', endArrowDrag);
  if (controls) controls.enabled = true;
  document.body.style.cursor = 'crosshair';
  confirmEdgeFinish();
}

function confirmEdgeFinish() {
  if (_pick.selected < 0 || !(_pick.radius > 0)) return;
  window._editState.finishes.push({ type: _pick.kind, value: +_pick.radius.toFixed(3), edge_mid: _pick.mid });
  renderEditChips();
  const kind = _pick.kind;
  exitEdgePick();
  document.getElementById('et-status').textContent = `Applying ${kind}…`;
  applyEdits();
}

function populateUI(statusData) {
  console.debug('[populateUI] building tree and timeline for', statusData.steps?.length, 'steps, filename:', statusData.filename);
  const list = document.getElementById('feature-list');
  const timeline = document.getElementById('timeline');
  list.innerHTML = '';
  timeline.innerHTML = '';
  steps = statusData.steps;

  // Keep the right-sidebar Stages panel in sync on every full render — covers
  // non-live paths the status poll doesn't drive (variant switch, cached
  // restore). It rebuilds when the step set changed; status 'done' here means
  // all cards settle (no loaders).
  updateStagesPanel(statusData.job_id || currentJobId, statusData);

  // (Icon mapping moved to the module-level featureRowInnerHTML helper so the
  // sidebar and live-build trees render identically.)

  // Hide sketches folder — sketches go inline under Body
  const sketchesFolder = document.getElementById('sketches-folder');
  sketchesFolder.style.display = 'none';

  // Add "Mesh" button at the start of timeline. We prefer the PIPELINE mesh
  // (/api/job/{id}/mesh) — the decimated, aligned frame the CAD was built in —
  // so the Mesh view, its thumbnail and the Compare view/thumbnail all share
  // ONE orientation (fixes the misaligned compare thumbnail) and the button is
  // never grayed just because the raw upload preview wasn't kept.
  let cmpMeshData = savedMeshPreview
    ? { vertices: savedMeshPreview.mesh_vertices, faces: savedMeshPreview.mesh_faces }
    : null;
  const meshBtn = document.createElement('div');
  meshBtn.className = 'timeline-step timeline-special timeline-dashed';
  meshBtn.style.background = 'var(--bg-tertiary)';
  meshBtn.innerHTML = '<span style="font-size:10px;color:var(--text-secondary);">Mesh</span>';
  meshBtn.title = 'Show original mesh';
  meshBtn.onclick = () => {
    if (!cmpMeshData) return;
    try { LiveBuild.dismiss(); } catch (e) {}
    _showRawMesh({ mesh_vertices: cmpMeshData.vertices, mesh_faces: cmpMeshData.faces }, false);
    document.querySelectorAll('.timeline-step').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.feature-item').forEach(el => el.classList.remove('active'));
    meshBtn.classList.add('active');
  };
  timeline.appendChild(meshBtn);

  // Build feature tree items (left sidebar) — these navigate individual steps.
  // Uses the SHARED row renderer so the live-build pane tree is identical.
  statusData.steps.forEach((step, i) => {
    const item = document.createElement('div');
    item.className = 'feature-item';
    item.innerHTML = featureRowInnerHTML(step.name, step.volume);
    item.onclick = () => loadStep(i);
    list.appendChild(item);
  });

  setupEditToolbar(statusData);

  // --- Bottom timeline: variant thumbnails ---
  // Each variant is a different build order of the same part.
  const variants = statusData.variants || [];
  const activeVariant = statusData.active_variant || 0;

  variants.forEach((variant, vi) => {
    const vBtn = document.createElement('div');
    vBtn.className = 'timeline-step timeline-special variant-btn' + (vi === activeVariant ? ' active' : '');
    vBtn.innerHTML = `<span style="font-size:9px;color:var(--text-secondary);text-align:center;line-height:1.2">${variant.name}</span>`;
    vBtn.title = variant.strategy || variant.name;
    vBtn.onclick = async () => {
      // Switch variant on server
      const res = await fetch(`/api/job/${currentJobId}/variant/${vi}`, { method: 'POST' });
      if (!res.ok) return;
      const data = await res.json();
      // Update steps and rebuild feature tree
      statusData.steps = data.steps;
      statusData.num_steps = data.num_steps;
      statusData.active_variant = vi;
      statusData.cad_volume = data.cad_volume;
      // Rebuild UI
      populateUI(statusData);
      // Load final step of new variant
      loadStep(data.num_steps - 1);
    };
    timeline.appendChild(vBtn);

    // Queue thumbnail render for the final step of this variant
    // (use the variant's last step index — all variants share the same job steps endpoint
    // after switching, so we render from the currently active steps)
    if (vi === activeVariant) {
      thumbQueue.push({ index: statusData.steps.length - 1, element: vBtn });
    }
  });

  // If no variants, show a simple "CAD" button
  if (variants.length === 0) {
    const cadBtn = document.createElement('div');
    cadBtn.className = 'timeline-step timeline-special active';
    cadBtn.innerHTML = '<span style="font-size:10px;color:var(--text-secondary);">CAD</span>';
    cadBtn.onclick = () => loadStep(statusData.steps.length - 1);
    timeline.appendChild(cadBtn);
    thumbQueue.push({ index: statusData.steps.length - 1, element: cadBtn });
  }

  // Add "Compare" split-screen button at the end
  const compareBtn = document.createElement('div');
  compareBtn.className = 'timeline-step timeline-special timeline-dashed';
  compareBtn.style.background = 'var(--bg-tertiary)';
  compareBtn.innerHTML = '<span style="font-size:10px;color:var(--text-secondary);">Compare</span>';
  compareBtn.title = 'Split view: Mesh vs CAD';
  compareBtn.onclick = () => {
    if (!currentJobId) { console.warn('[compare] no job to compare'); return; }
    // Clear and show both mesh + CAD side by side
    while (modelGroup.children.length) {
      const child = modelGroup.children[0];
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
      modelGroup.remove(child);
    }

    const lastIdx = statusData.steps.length - 1;
    // Use the PIPELINE mesh (decimated + aligned) — the exact frame the CAD was
    // built in — so the mesh and the model compare in ONE orientation. No
    // canonical-orient / flip hacks needed; they're already the same frame.
    Promise.all([
      fetch(`/api/job/${currentJobId}/mesh`).then(r => {
        if (!r.ok) throw new Error('mesh HTTP ' + r.status); return r.json();
      }),
      fetch(`/api/job/${currentJobId}/step/${lastIdx}`).then(r => {
        if (!r.ok) throw new Error('step HTTP ' + r.status); return r.json();
      }),
    ]).then(([meshData, cadData]) => {
      if (!meshData.vertices || !cadData.vertices) {
        console.warn('[compare] missing geometry'); return;
      }
      try {
        // Mesh (left)
        const meshGeo = new THREE.BufferGeometry();
        meshGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(meshData.vertices), 3));
        meshGeo.setIndex(meshData.faces);
        meshGeo.computeVertexNormals();
        const meshMat = new THREE.MeshStandardMaterial({ color: 0x8B949E, metalness: 0.1, roughness: 0.8 });
        const meshObj = new THREE.Mesh(meshGeo, meshMat);

        // CAD (right)
        const cadGeo = new THREE.BufferGeometry();
        cadGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(cadData.vertices), 3));
        cadGeo.setIndex(cadData.faces);
        cadGeo.computeVertexNormals();
        const cadMat = new THREE.MeshPhysicalMaterial({
          color: 0xC0C8D0, metalness: 0.5, roughness: 0.28,
          clearcoat: 0.15, reflectivity: 0.6,
        });
        const cadObj = new THREE.Mesh(cadGeo, cadMat);

        // Center both, then place side by side with small gap
        const meshBox = new THREE.Box3().setFromBufferAttribute(meshGeo.attributes.position);
        const cadBox = new THREE.Box3().setFromBufferAttribute(cadGeo.attributes.position);
        const meshCenter = meshBox.getCenter(new THREE.Vector3());
        const cadCenter = cadBox.getCenter(new THREE.Vector3());
        const meshSize = meshBox.getSize(new THREE.Vector3());
        const cadSize = cadBox.getSize(new THREE.Vector3());
        const spacing = Math.max(meshSize.x, cadSize.x) * 0.15; // 15% of widest object
        const halfGap = (meshSize.x + cadSize.x) / 4 + spacing / 2;

        // Center each, then shift left/right
        meshObj.position.set(-halfGap - meshCenter.x, -meshCenter.y, -meshCenter.z);
        cadObj.position.set(halfGap - cadCenter.x, -cadCenter.y, -cadCenter.z);

        modelGroup.add(meshObj);
        modelGroup.add(cadObj);

        // Add wireframe to mesh side
        const wire = new THREE.LineSegments(
          new THREE.WireframeGeometry(meshGeo),
          new THREE.LineBasicMaterial({ color: 0x30363D, linewidth: 1 })
        );
        wire.position.copy(meshObj.position);
        modelGroup.add(wire);

        // Center the whole group
        modelGroup.position.set(0, 0, 0);
        const combinedBox = new THREE.Box3().setFromObject(modelGroup);
        const combinedCenter = combinedBox.getCenter(new THREE.Vector3());
        modelGroup.position.sub(combinedCenter);

        // Zoom out to fit both models
        const combinedSize = combinedBox.getSize(new THREE.Vector3()).length();
        const camY = Math.max(combinedSize * 0.9, combinedBox.getSize(new THREE.Vector3()).y * 2.0);
        animateCamera(
          { x: combinedSize * 0.5, y: camY, z: combinedSize * 0.5 },
          { x: 0, y: 0, z: 0 },
          1.0,
        );
        console.debug('[compare] split view rendered, spacing:', spacing.toFixed(1), 'combinedSize:', combinedSize.toFixed(1));
      } catch (e) {
        console.error('[compare] geometry creation failed', e);
      }

      // Deactivate all steps
      document.querySelectorAll('.timeline-step').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.feature-item').forEach(el => el.classList.remove('active'));
      compareBtn.classList.add('active');
    }).catch(e => { console.error('[compare] failed to load CAD for split view', e); });
  };
  timeline.appendChild(compareBtn);

  // Show body folder if there are features
  const bodyFolder = document.getElementById('body-folder');
  if (bodyFolder) bodyFolder.style.display = list.children.length > 0 ? '' : 'none';

  // Set component name from filename
  const compName = document.getElementById('component-name');
  if (compName && statusData.filename) {
    compName.textContent = statusData.filename.replace(/\.[^.]+$/, '');
  }
  const fnLabel = document.getElementById('viewer-filename');
  if (fnLabel && statusData.filename) fnLabel.textContent = statusData.filename;

  // Render thumbnails sequentially to avoid WebGL race conditions
  console.debug('[populateUI] queued', thumbQueue.length, 'thumbnails for rendering');
  renderThumbQueue().then(async () => {
    // Prefer the pipeline mesh (same frame as the CAD) for the Mesh view and
    // both thumbnails. Always available after the run, so the button never
    // grays out and the compare thumbnail matches the compare view.
    if (currentJobId) {
      try {
        const r = await fetch(`/api/job/${currentJobId}/mesh`);
        if (r.ok) {
          const pm = await r.json();
          if (pm && pm.vertices && pm.vertices.length) {
            cmpMeshData = { vertices: pm.vertices, faces: pm.faces };
          }
        }
      } catch (e) { /* fall back to savedMeshPreview */ }
    }
    // Render Mesh thumbnail
    if (cmpMeshData) {
      const url = renderMeshThumbnail(cmpMeshData.vertices, cmpMeshData.faces);
      if (url) {
        meshBtn.style.backgroundImage = `url(${url})`;
        meshBtn.style.backgroundSize = 'cover';
        meshBtn.style.backgroundPosition = 'center';
        meshBtn.querySelector('span').style.display = 'none';
      }
    }

    // Render thumbnails for non-active variants by switching, rendering, switching back
    const variantBtns = timeline.querySelectorAll('.timeline-step.variant-btn');
    for (let vi = 0; vi < variants.length; vi++) {
      if (vi === activeVariant) continue;
      const btn = variantBtns[vi];
      if (!btn) continue;
      try {
        // Temporarily switch to this variant
        const swRes = await fetch(`/api/job/${currentJobId}/variant/${vi}`, { method: 'POST' });
        if (!swRes.ok) continue;
        const swData = await swRes.json();
        // Render its final step
        const url = await renderThumbnail(swData.num_steps - 1);
        if (url && btn) {
          btn.style.backgroundImage = `url(${url})`;
          btn.style.backgroundSize = 'cover';
          btn.style.backgroundPosition = 'center';
          const label = btn.querySelector('span');
          if (label) label.style.display = 'none';
        }
      } catch (e) { console.warn('[variant thumb] failed for variant', vi, e); }
    }
    // Switch back to the active variant
    if (variants.length > 0) {
      await fetch(`/api/job/${currentJobId}/variant/${activeVariant}`, { method: 'POST' });
    }
    // Render Compare thumbnail showing both mesh + CAD side by side. Uses the
    // SAME pipeline mesh as the Compare view, so the thumbnail and the view
    // line up in one orientation.
    if (cmpMeshData && statusData.steps.length > 0) {
      const lastIdx = statusData.steps.length - 1;
      const cadUrl2 = currentJobId ? `/api/job/${currentJobId}/step/${lastIdx}` : `/api/step/${lastIdx}`;
      fetch(cadUrl2).then(r => r.json()).then(cadData => {
        if (!cadData.vertices) return;
        try {
          const ts = new THREE.Scene();
          const td = document.documentElement.getAttribute('data-theme') === 'dark';
          ts.background = new THREE.Color(td ? 0x161B22 : 0xF5F5F5);
          ts.add(new THREE.AmbientLight(0xffffff, 0.5));
          const tl = new THREE.DirectionalLight(0xffffff, 0.9);
          tl.position.set(150, 200, 100);
          ts.add(tl);

          // Mesh side
          const mv = new Float32Array(cmpMeshData.vertices);
          const mg = new THREE.BufferGeometry();
          mg.setAttribute('position', new THREE.BufferAttribute(mv, 3));
          mg.setIndex(cmpMeshData.faces);
          mg.computeVertexNormals();
          const mm = new THREE.Mesh(mg, new THREE.MeshStandardMaterial({ color: 0x8B949E, metalness: 0.1, roughness: 0.8 }));

          // CAD side
          const cv = new Float32Array(cadData.vertices);
          const cg = new THREE.BufferGeometry();
          cg.setAttribute('position', new THREE.BufferAttribute(cv, 3));
          cg.setIndex(cadData.faces);
          cg.computeVertexNormals();
          const cm = new THREE.Mesh(cg, new THREE.MeshStandardMaterial({ color: 0xB0B8C0, metalness: 0.4, roughness: 0.35 }));

          // Position side by side
          const mb = new THREE.Box3().setFromBufferAttribute(mg.attributes.position);
          const cb = new THREE.Box3().setFromBufferAttribute(cg.attributes.position);
          const mc = mb.getCenter(new THREE.Vector3());
          const cc = cb.getCenter(new THREE.Vector3());
          const ms = mb.getSize(new THREE.Vector3());
          const cs = cb.getSize(new THREE.Vector3());
          const sp = Math.max(ms.x, cs.x) * 0.15;
          const hg = (ms.x + cs.x) / 4 + sp / 2;
          mm.position.set(-hg - mc.x, -mc.y, -mc.z);
          cm.position.set(hg - cc.x, -cc.y, -cc.z);
          ts.add(mm);
          ts.add(cm);

          // Fit camera
          const tb = new THREE.Box3().setFromObject(ts);
          const tc = tb.getCenter(new THREE.Vector3());
          const tsize = tb.getSize(new THREE.Vector3()).length();
          const tcam = new THREE.PerspectiveCamera(45, 160 / 112, 0.1, 5000);
          tcam.position.set(tc.x + tsize * 0.7, tc.y + tsize * 0.5, tc.z + tsize * 0.7);
          tcam.lookAt(tc);
          thumbRenderer.render(ts, tcam);
          const url = thumbRenderer.domElement.toDataURL('image/png');
          compareBtn.style.backgroundImage = `url(${url})`;
          compareBtn.style.backgroundSize = 'cover';
          compareBtn.style.backgroundPosition = 'center';
          compareBtn.querySelector('span').style.display = 'none';
        } catch (e) { console.warn('[compare thumb] failed', e); }
      }).catch(() => {});
    }
  });
}

// ── Thumbnail renderer (offscreen, serialized) ──
const thumbRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
thumbRenderer.setSize(160, 112);
thumbRenderer.toneMapping = THREE.ACESFilmicToneMapping;
console.debug('[init] Thumbnail renderer created (160x112)');
let thumbQueue = [];
let thumbRendering = false;

async function renderThumbQueue() {
  if (thumbRendering) return;
  thumbRendering = true;
  console.debug('[renderThumbQueue] starting, items:', thumbQueue.length);
  let rendered = 0;
  while (thumbQueue.length > 0) {
    const { index, element } = thumbQueue.shift();
    try {
      const dataUrl = await renderThumbnail(index);
      if (dataUrl && element) {
        element.style.backgroundImage = `url(${dataUrl})`;
        element.style.backgroundSize = 'cover';
        element.style.backgroundPosition = 'center';
        const label = element.querySelector('.step-num') || element.querySelector('span');
        if (label) label.style.display = 'none';
        rendered++;
      }
    } catch (e) {
      console.error('[renderThumbQueue] failed for step', index, e);
    }
  }
  console.debug('[renderThumbQueue] done, rendered:', rendered);
  thumbRendering = false;
}

async function renderThumbnail(stepIndex) {
  try {
    const url = currentJobId ? `/api/job/${currentJobId}/step/${stepIndex}` : `/api/step/${stepIndex}`;
    const res = await fetch(url);
    if (!res.ok) { console.warn('[renderThumbnail] HTTP', res.status, 'for step', stepIndex, url); return null; }
    let data;
    try { data = await res.json(); } catch (e) { console.error('[renderThumbnail] JSON parse failed for step', stepIndex, e); return null; }
    if (!data.vertices || data.vertices.length === 0) return null;

    const thumbScene = new THREE.Scene();
    const thumbIsDark = document.documentElement.getAttribute('data-theme') === 'dark';
    thumbScene.background = new THREE.Color(thumbIsDark ? 0x161B22 : 0xF5F5F5);

    // Lights
    thumbScene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const light = new THREE.DirectionalLight(0xffffff, 0.9);
    light.position.set(150, 200, 100);
    thumbScene.add(light);

    const verts = new Float32Array(data.vertices);
    const indices = data.faces;

    if (data.is_sketch) {
      // Use face indices for edge segments (same as main viewer)
      const points = [];
      for (let f = 0; f < indices.length; f += 3) {
        const a = indices[f], b = indices[f + 1];
        points.push(
          new THREE.Vector3(verts[a*3], verts[a*3+1], verts[a*3+2]),
          new THREE.Vector3(verts[b*3], verts[b*3+1], verts[b*3+2]),
        );
      }
      const geo = new THREE.BufferGeometry().setFromPoints(points);
      thumbScene.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x58A6FF })));
    } else {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
      geo.setIndex(indices);
      geo.computeVertexNormals();
      const mat = new THREE.MeshStandardMaterial({ color: 0xB0B8C0, metalness: 0.4, roughness: 0.35 });
      thumbScene.add(new THREE.Mesh(geo, mat));
      thumbScene.add(new THREE.LineSegments(
        new THREE.EdgesGeometry(geo, 25),
        new THREE.LineBasicMaterial({ color: 0x30363D }),
      ));
    }

    // Fit camera to content
    const box = new THREE.Box3().setFromObject(thumbScene);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    if (size < 0.001) return null; // degenerate geometry
    const thumbCam = new THREE.PerspectiveCamera(45, 160 / 112, 0.1, 5000);
    thumbCam.position.set(center.x + size * 0.8, center.y + size * 0.6, center.z + size * 0.8);
    thumbCam.lookAt(center);

    thumbRenderer.render(thumbScene, thumbCam);
    return thumbRenderer.domElement.toDataURL('image/png');
  } catch (e) {
    console.warn('Thumbnail render failed for step', stepIndex, e);
    return null;
  }
}

// Render a thumbnail from raw mesh vertex/face data (for Mesh button)
function renderMeshThumbnail(meshVerts, meshFaces) {
  try {
    const thumbScene = new THREE.Scene();
    const thumbIsDark = document.documentElement.getAttribute('data-theme') === 'dark';
    thumbScene.background = new THREE.Color(thumbIsDark ? 0x161B22 : 0xF5F5F5);
    thumbScene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const light = new THREE.DirectionalLight(0xffffff, 0.9);
    light.position.set(150, 200, 100);
    thumbScene.add(light);

    const verts = new Float32Array(meshVerts);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    geo.setIndex(meshFaces);
    geo.computeVertexNormals();
    thumbScene.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x8B949E, metalness: 0.1, roughness: 0.8 })));
    thumbScene.add(new THREE.LineSegments(new THREE.WireframeGeometry(geo), new THREE.LineBasicMaterial({ color: 0x30363D })));

    const box = new THREE.Box3().setFromObject(thumbScene);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length();
    if (size < 0.001) return null;
    const thumbCam = new THREE.PerspectiveCamera(45, 160 / 112, 0.1, 5000);
    thumbCam.position.set(center.x + size * 0.8, center.y + size * 0.6, center.z + size * 0.8);
    thumbCam.lookAt(center);
    thumbRenderer.render(thumbScene, thumbCam);
    return thumbRenderer.domElement.toDataURL('image/png');
  } catch (e) {
    console.warn('[renderMeshThumbnail] failed', e);
    return null;
  }
}

// ── File Upload: two-step flow (preview mesh → analyze & build) ──
let pendingFile = null;
let pendingTempPath = null;
let savedMeshPreview = null; // saved mesh data for comparison view

async function previewFile(file) {
  if (!file) return;
  console.debug('[upload] file selected:', file.name, 'size:', (file.size / 1024).toFixed(1), 'KB');

  const loading = document.getElementById('loading');
  const loadingText = document.getElementById('loading-text');
  const progressBar = document.getElementById('progress-bar');
  const progressDetail = document.getElementById('progress-detail');
  loading.classList.add('show');
  loadingText.textContent = `Uploading ${file.name}...`;
  progressBar.style.width = '0%';
  progressDetail.textContent = '';

  const form = new FormData();
  form.append('file', file);

  try {
    // Step 1: Upload and preview the raw mesh
    const previewRes = await new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open('POST', '/api/preview');
      xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
          progressBar.style.width = Math.round((e.loaded / e.total) * 50) + '%';
          progressDetail.textContent = 'Uploading...';
        }
      };
      xhr.onload = () => {
        try { resolve(JSON.parse(xhr.responseText)); }
        catch (parseErr) { console.error('[upload] JSON parse error from preview response', parseErr); reject(parseErr); }
      };
      xhr.onerror = () => { console.error('[upload] XHR network error for /api/preview'); reject(new Error('Upload failed')); };
      xhr.send(form);
    });

    loading.classList.remove('show');

    if (previewRes.error) {
      console.error('[upload] preview returned error:', previewRes.error);
      alert(previewRes.error);
      return;
    }

    console.debug('[upload] preview received:', previewRes.faces, 'faces,', previewRes.vertices, 'verts, complexity:', previewRes.complexity, 'score:', previewRes.complexity_score);

    // Show filename under the viewer
    const fnLabel = document.getElementById('viewer-filename');
    if (fnLabel) fnLabel.textContent = previewRes.filename || pendingFile?.name || '';
    // Now that the current mesh name has changed, relabel the Onshape button:
    // a new (un-analyzed) mesh no longer matches the bound doc, so it shows
    // "Open in Onshape" (not "Reopen"). Must run AFTER viewer-filename updates.
    try { updateOnshapeOpenButton && updateOnshapeOpenButton(); } catch (e) {}

    // AI thinking messages about the uploaded mesh
    addChatMessage('ai', `Loaded ${previewRes.filename || 'mesh'}: ${previewRes.faces.toLocaleString()} faces, ${previewRes.vertices.toLocaleString()} vertices`, true);
    addChatMessage('ai', `Dimensions: ${previewRes.dimensions.map(d => d + 'mm').join(' × ')}. ${previewRes.watertight ? 'Mesh is watertight.' : 'Mesh has gaps — will repair.'}`, true);
    addChatMessage('ai', `Complexity: ${previewRes.complexity}. This will use 1 part conversion. Click "Analyze & Build" to proceed.`, true);

    // Save mesh preview data to sessionStorage for refresh persistence
    try {
      const previewData = {
        mesh_vertices: Array.from(previewRes.mesh_vertices),
        mesh_faces: previewRes.mesh_faces,
        filename: previewRes.filename,
        faces: previewRes.faces,
        vertices: previewRes.vertices,
        temp_path: previewRes.temp_path,
      };
      const json = JSON.stringify(previewData);
      if (json.length < 5_000_000) { // 5MB limit for sessionStorage
        sessionStorage.setItem('paramesh_preview', json);
      }
    } catch (e) { console.warn('[upload] could not save preview to sessionStorage:', e); }

    // Reset UI from any previous job
    document.getElementById('feature-list').innerHTML = '';
    document.getElementById('timeline').innerHTML = '';
    resetStagesPanel();
    currentStep = -1;
    currentJobId = null;
    // If a previous analysis left the dual/expanded build pane up (live-split or
    // live-full), tear it down so the main #viewer — where _showRawMesh draws
    // this newly-picked mesh — is revealed again. Without this, clicking another
    // folder part only updates the complexity bar; the old built model stays on
    // top and the new raw mesh renders hidden behind it.
    try { LiveBuild.reset(); LiveBuild.hide(); } catch (e) {}
    // Tear down the feature-tree EDITOR (fillet/chamfer edit chips + edge-pick)
    // left over from the previous build, so it doesn't persist into this new
    // part's pipeline.
    try { if (typeof exitEdgePick === 'function') exitEdgePick(); } catch (e) {}
    window._editState = { deletes: new Set(), finishes: [] };
    try { renderEditChips(); } catch (e) {}
    const _fe = document.getElementById('feature-editor');
    if (_fe) _fe.style.display = 'none';
    const _ets = document.getElementById('et-status');
    if (_ets) _ets.textContent = '';

    // Show the raw mesh in the 3D viewer
    pendingTempPath = previewRes.temp_path;
    pendingFile = file;
    saveViewerState('preview');
    savedMeshPreview = previewRes;
    console.debug('[upload] state -> preview, showing raw mesh');

    // Display the FULL-RESOLUTION mesh parsed client-side. The server preview is
    // decimated to ~30k faces for the pipeline, which facets/bumps curved AI
    // meshes — so render the real geometry the user actually has. Keep BOTH so a
    // toggle can A/B the full mesh vs the decimated copy the pipeline works on.
    let _displayPreview = previewRes;
    window._previewVariants = null;
    try {
      const obj = await _loadObject(file);
      const arr = obj && _hasMesh(obj) ? _objToArrays(obj) : null;
      if (arr) {
        _displayPreview = { ...previewRes, mesh_vertices: arr.vertices, mesh_faces: arr.faces };
        window._previewVariants = {
          which: 'full',
          full: { vertices: arr.vertices, faces: arr.faces, n: arr.faces.length / 3 },
          decimated: { vertices: previewRes.mesh_vertices, faces: previewRes.mesh_faces,
                       n: (previewRes.mesh_faces ? previewRes.mesh_faces.length : 0) / 3 },
        };
        console.debug('[upload] full-res', arr.faces.length / 3, 'vs decimated', window._previewVariants.decimated.n);
      }
    } catch (e) {
      console.warn('[upload] full-res client parse failed; using server preview', e);
    }
    _showRawMesh(_displayPreview);
    _setupQualityToggle();

    // Show preview info bar with complexity meter
    document.getElementById('preview-bar').style.display = 'block';
    document.getElementById('preview-faces').textContent = previewRes.faces.toLocaleString() + ' faces';
    document.getElementById('preview-dims').textContent = previewRes.dimensions.map(d => d + 'mm').join(' × ');
    window._lastPreviewDims = previewRes.dimensions;
    document.getElementById('preview-tokens').textContent = '1 part conversion';

    // Complexity meter
    const score = previewRes.complexity_score || 0;
    const fill = document.getElementById('complexity-fill');
    const label = document.getElementById('preview-complexity');
    const warning = document.getElementById('preview-warning');
    fill.style.width = score + '%';

    if (score <= 25) {
      fill.style.background = '#22C55E';
      label.textContent = previewRes.complexity;
      label.style.color = '#22C55E';
    } else if (score <= 50) {
      fill.style.background = '#D29922';
      label.textContent = previewRes.complexity;
      label.style.color = '#D29922';
    } else if (score <= 75) {
      fill.style.background = '#F97316';
      label.textContent = previewRes.complexity;
      label.style.color = '#F97316';
    } else {
      fill.style.background = '#F85149';
      label.textContent = previewRes.complexity;
      label.style.color = '#F85149';
    }

    // Show warning or block if too complex
    if (!previewRes.can_process) {
      warning.style.display = 'block';
      warning.textContent = previewRes.reject_reason || 'This mesh is too complex for the current pipeline.';
      document.getElementById('btn-analyze').style.display = 'none';
      addChatMessage('ai', (previewRes.reject_reason || 'Mesh is too complex to process.'), true);
    } else {
      warning.style.display = 'none';
      document.getElementById('btn-analyze').style.display = '';
      if (score > 75) {
        addChatMessage('ai', `This mesh has high geometric complexity. Results may vary — works best on prismatic parts like brackets, plates, and mounts.`, true);
      }
    }

    // Keep upload button visible so user can upload a different mesh
    document.getElementById('empty-state').style.display = 'none';

  } catch (err) {
    console.error('[upload] failed:', err);
    alert('Upload failed: ' + err.message);
    loading.classList.remove('show');
  }
}

// A/B the full-resolution mesh against the pipeline's decimated copy so you can
// SEE what decimation does to curves. Shows the face counts on the button.
function _setupQualityToggle() {
  const btn = document.getElementById('preview-quality-toggle');
  if (!btn) return;
  const v = window._previewVariants;
  if (!v || !v.decimated.faces || v.decimated.n === 0) { btn.style.display = 'none'; return; }
  btn.style.display = '';
  const lbl = () => `Showing: ${v.which === 'full' ? 'Full' : 'Decimated'} (` +
    `${(v.which === 'full' ? v.full.n : v.decimated.n).toLocaleString()} faces)`;
  btn.textContent = lbl();
}

window.togglePreviewQuality = function () {
  const v = window._previewVariants;
  if (!v) return;
  v.which = v.which === 'full' ? 'decimated' : 'full';
  const m = v.which === 'full' ? v.full : v.decimated;
  _showRawMesh({ mesh_vertices: m.vertices, mesh_faces: m.faces }, false);
  _setupQualityToggle();
};

document.getElementById('file-input').addEventListener('change', async (e) => {
  const f = e.target.files[0];
  try { await previewFile(f); } finally { e.target.value = ''; }
});

// ─── Folder browser ──────────────────────────────────────────────────────────
// "Open Folder" lists every mesh in the chosen directory in the left sidebar
// with a client-side 3D thumbnail; clicking one loads it through previewFile().
const MESH_EXTS = ['stl', 'obj', 'ply', '3mf'];
let _thumbRenderer = null;
window._folderActive = false;

// Thumbnail cache so closing + reopening a folder (or re-picking it) doesn't
// re-render every mesh. Keyed by a STABLE file identity (path|size|mtime) since
// re-picking a folder yields fresh File objects. In-memory for the session,
// mirrored to localStorage so it also survives a page reload (size-capped).
const _thumbCache = new Map();
const _THUMB_LS_KEY = 'paramesh_thumb_cache_v1';
const _thumbKey = (f) => `${f._relPath || f.webkitRelativePath || f.name}|${f.size}|${f.lastModified}`;
try {
  const saved = JSON.parse(localStorage.getItem(_THUMB_LS_KEY) || '{}');
  for (const k in saved) _thumbCache.set(k, saved[k]);
} catch (e) { /* corrupt/oversized cache — ignore */ }
let _thumbSaveTimer = null;
function _persistThumbCache() {
  // Debounced; drop oldest entries if we blow the localStorage quota.
  clearTimeout(_thumbSaveTimer);
  _thumbSaveTimer = setTimeout(() => {
    let entries = [..._thumbCache.entries()];
    while (entries.length) {
      try {
        localStorage.setItem(_THUMB_LS_KEY, JSON.stringify(Object.fromEntries(entries)));
        break;
      } catch (e) {
        entries = entries.slice(Math.ceil(entries.length / 4));  // quota hit: shed 25%
      }
    }
  }, 800);
}

const _meshExt = (name) => (name.split('.').pop() || '').toLowerCase();
const _fmtSize = (b) => b < 1048576 ? (b / 1024).toFixed(0) + ' KB' : (b / 1048576).toFixed(1) + ' MB';

// ── Folder persistence across reloads ──
// The File System Access directory handle is structured-cloneable, so we stash
// it in IndexedDB and re-open the folder on reload. Browsers reset file
// permission to 'prompt' on a fresh page load (re-granting needs a user
// gesture), so when the grant didn't survive we show a one-click "Reopen
// <folder>" chip that re-grants WITHOUT re-navigating the picker.
function _folderIDB() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open('paramesh-folder', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('kv');
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}
async function _folderHandleSave(dir) {
  try {
    const db = await _folderIDB();
    await new Promise((res, rej) => {
      const tx = db.transaction('kv', 'readwrite');
      tx.objectStore('kv').put(dir, 'handle');
      tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error);
    });
  } catch (e) { console.warn('[folder] could not persist handle', e); }
}
async function _folderHandleLoad() {
  try {
    const db = await _folderIDB();
    return await new Promise((res, rej) => {
      const tx = db.transaction('kv', 'readonly');
      const rq = tx.objectStore('kv').get('handle');
      rq.onsuccess = () => res(rq.result || null); rq.onerror = () => rej(rq.error);
    });
  } catch (e) { return null; }
}
async function _folderHandleClear() {
  try {
    const db = await _folderIDB();
    await new Promise((res) => {
      const tx = db.transaction('kv', 'readwrite');
      tx.objectStore('kv').delete('handle');
      tx.oncomplete = () => res(); tx.onerror = () => res();
    });
  } catch (e) { /* ignore */ }
  const chip = document.getElementById('folder-reopen-chip');
  if (chip) chip.remove();
}

// Walk a directory handle and return its mesh File objects (sorted).
async function _walkDir(dir) {
  const meshes = [];
  async function walk(handle, prefix) {
    for await (const entry of handle.values()) {
      if (entry.kind === 'file' && MESH_EXTS.includes(_meshExt(entry.name))) {
        try { const f = await entry.getFile(); f._relPath = prefix + entry.name; meshes.push(f); }
        catch (e) { /* unreadable file — skip */ }
      } else if (entry.kind === 'directory') {
        await walk(entry, prefix + entry.name + '/');
      }
    }
  }
  await walk(dir, (dir.name || 'Folder') + '/');
  meshes.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  return meshes;
}

// On page load: re-open the last folder if its permission survived; otherwise
// surface a one-click reopen chip.
async function restoreFolder() {
  if (!window.showDirectoryPicker) return;      // FSA unsupported → no handle to restore
  const dir = await _folderHandleLoad();
  if (!dir) return;
  let perm = 'prompt';
  try { perm = await dir.queryPermission({ mode: 'read' }); } catch (e) { return; }
  if (perm === 'granted') {
    try {
      const meshes = await _walkDir(dir);
      if (meshes.length) openFolderPanel(meshes);
    } catch (e) { console.warn('[folder] silent restore failed', e); }
  } else {
    _showFolderReopenChip(dir);
  }
}

function _showFolderReopenChip(dir) {
  if (document.getElementById('folder-reopen-chip')) return;
  const chip = document.createElement('button');
  chip.id = 'folder-reopen-chip';
  chip.innerHTML = '<i class="ri-folder-open-line"></i>&nbsp;Reopen ' + (dir.name || 'folder');
  chip.style.cssText =
    'position:fixed;left:16px;bottom:16px;z-index:60;display:flex;align-items:center;' +
    'gap:6px;padding:9px 14px;border:1px solid #30363D;border-radius:8px;background:#161B22;' +
    'color:#E6EDF3;font-size:13px;font-weight:500;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.4)';
  chip.onclick = async () => {
    try {
      const p = await dir.requestPermission({ mode: 'read' });   // user gesture re-grants
      if (p !== 'granted') return;
      const meshes = await _walkDir(dir);
      chip.remove();
      if (meshes.length) openFolderPanel(meshes);
    } catch (e) { console.warn('[folder] reopen failed', e); }
  };
  document.body.appendChild(chip);
}

// Primary path: File System Access API — opens a normal folder picker with NO
// "upload N files to this site?" prompt, and reads files locally (nothing is
// uploaded; thumbnails render in-browser). Falls back to the webkitdirectory
// input on browsers without it (Firefox/Safari).
window.openFolderPicker = async function () {
  if (window.showDirectoryPicker) {
    let dir;
    try {
      dir = await window.showDirectoryPicker({ id: 'paramesh-meshes', mode: 'read' });
    } catch (e) { return; }   // user cancelled
    let meshes = [];
    try { meshes = await _walkDir(dir); }
    catch (e) { console.warn('[folder] walk failed', e); }
    if (!meshes.length) { alert('No mesh files (.stl, .obj, .ply, .3mf) found in that folder.'); return; }
    _folderHandleSave(dir);                                     // persist for restore-on-reload
    const chip = document.getElementById('folder-reopen-chip');  // drop any stale reopen chip
    if (chip) chip.remove();
    openFolderPanel(meshes);
  } else {
    document.getElementById('folder-input').click();   // older-browser fallback
  }
};

document.getElementById('folder-input')?.addEventListener('change', (e) => {
  const meshes = Array.from(e.target.files || []).filter(f => MESH_EXTS.includes(_meshExt(f.name)));
  e.target.value = '';
  if (!meshes.length) { alert('No mesh files (.stl, .obj, .ply, .3mf) found in that folder.'); return; }
  meshes.sort((a, b) => a.name.localeCompare(b.name, undefined, { numeric: true }));
  openFolderPanel(meshes);
});

function openFolderPanel(meshes) {
  window._folderActive = true;
  const panel = document.getElementById('folder-panel');
  const list = document.getElementById('folder-list');
  const rel = meshes[0]._relPath || meshes[0].webkitRelativePath || '';
  document.getElementById('folder-title').textContent =
    (rel.split('/')[0] || 'Folder') + ' · ' + meshes.length + ' meshes';
  list.innerHTML = '';
  panel.style.display = 'flex';
  panel.classList.remove('min');
  // Own left panel; the feature tree shifts to its right (see CSS .folder-open).
  document.body.classList.add('folder-open');
  document.body.classList.remove('folder-min');
  document.getElementById('folder-tab').style.display = 'none';

  meshes.forEach((file, i) => {
    const item = document.createElement('div');
    item.className = 'folder-item';
    item.dataset.idx = i;
    item.innerHTML =
      '<div class="folder-thumb"><i class="ri-loader-4-line fspin"></i></div>' +
      '<div class="folder-meta"><div class="folder-name"></div>' +
      '<div class="folder-ext"></div></div>';
    item.querySelector('.folder-name').textContent = file.name;
    item.querySelector('.folder-name').title = file.name;
    item.querySelector('.folder-ext').textContent = _meshExt(file.name).toUpperCase() + ' · ' + _fmtSize(file.size);
    item.addEventListener('click', () => {
      document.querySelectorAll('.folder-item.active').forEach(el => el.classList.remove('active'));
      item.classList.add('active');
      previewFile(file);
    });
    list.appendChild(item);
  });

  _renderThumbsSequential(meshes);
}

window.closeFolderPanel = function () {
  window._folderActive = false;
  document.getElementById('folder-panel').style.display = 'none';
  document.getElementById('folder-list').innerHTML = '';
  document.getElementById('folder-tab').style.display = 'none';
  document.body.classList.remove('folder-open', 'folder-min');
  // Explicit close = forget it, so it doesn't reopen on the next load. (Minimize
  // via toggleFolderMin keeps it persisted.)
  _folderHandleClear();
};

window.toggleFolderMin = function () {
  const panel = document.getElementById('folder-panel');
  const tab = document.getElementById('folder-tab');
  const min = !panel.classList.contains('min');
  panel.classList.toggle('min', min);
  document.body.classList.toggle('folder-min', min);
  panel.style.display = min ? 'none' : 'flex';
  tab.style.display = min ? 'flex' : 'none';
};

function _setThumb(thumb, url) {
  thumb.innerHTML = '';
  const img = new Image();
  img.src = url;
  thumb.appendChild(img);
}

async function _renderThumbsSequential(meshes) {
  let cachedAny = false;
  for (let i = 0; i < meshes.length; i++) {
    const item = document.querySelector(`.folder-item[data-idx="${i}"]`);
    if (!item || !window._folderActive) continue;
    const thumb = item.querySelector('.folder-thumb');
    const key = _thumbKey(meshes[i]);
    const hit = _thumbCache.get(key);
    if (hit) {                       // cache hit — instant, no re-render
      _setThumb(thumb, hit);
      continue;
    }
    try {
      const url = await _thumbnailFor(meshes[i]);
      if (url) {
        _setThumb(thumb, url);
        _thumbCache.set(key, url);
        cachedAny = true;
      } else {
        thumb.innerHTML = '<i class="ri-shape-line"></i>';
      }
    } catch (err) {
      console.warn('[folder] thumbnail failed for', meshes[i].name, err);
      thumb.innerHTML = '<i class="ri-error-warning-line"></i>';
    }
    await new Promise(r => setTimeout(r, 0));   // yield so the UI stays responsive
  }
  if (cachedAny) _persistThumbCache();
}

function _getThumbRenderer() {
  if (_thumbRenderer) return _thumbRenderer;
  _thumbRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, preserveDrawingBuffer: true });
  _thumbRenderer.setSize(128, 128);
  return _thumbRenderer;
}

// Load a mesh file into an Object3D. For OBJ/3MF this keeps the loader's full
// scene graph (3MF build items carry per-mesh transforms — using the first raw
// geometry, as before, dropped those and produced empty/blank thumbnails).
function _hasMesh(obj) {
  if (!obj) return false;
  let found = false;
  obj.traverse(c => {
    if (c.isMesh && c.geometry && c.geometry.attributes && c.geometry.attributes.position
        && c.geometry.attributes.position.count > 0) found = true;
  });
  return found;
}

// Flatten a parsed Object3D into { vertices: [x,y,z,...], faces: [i,j,k,...] },
// merging all meshes (with their world transforms) and generating indices for
// non-indexed geometry. Used to display the FULL-RESOLUTION uploaded mesh
// instead of the server's decimated preview.
function _objToArrays(obj) {
  obj.updateMatrixWorld(true);
  const vertices = [];
  const faces = [];
  let base = 0;
  const v = new THREE.Vector3();
  obj.traverse(c => {
    if (!c.isMesh || !c.geometry || !c.geometry.attributes.position) return;
    const pos = c.geometry.attributes.position;
    const mw = c.matrixWorld;
    for (let i = 0; i < pos.count; i++) {
      v.fromBufferAttribute(pos, i).applyMatrix4(mw);
      vertices.push(v.x, v.y, v.z);
    }
    const idx = c.geometry.index;
    if (idx) { for (let i = 0; i < idx.count; i++) faces.push(base + idx.getX(i)); }
    else { for (let i = 0; i < pos.count; i++) faces.push(base + i); }
    base += pos.count;
  });
  return faces.length >= 3 ? { vertices, faces } : null;
}

// Direct 3MF parse: a .3mf is a ZIP whose model XML holds <vertex x y z> +
// <triangle v1 v2 v3>. AI exporters (Meshy/Tripo) split the mesh into a
// SEPARATE object .model file (root 3dmodel.model only has build items) and
// ThreeMFLoader's full pipeline can blank on that — so scan ALL .model files
// and extract the core mesh data by regex (fast + low-memory on the ~10MB
// object XML; DOMParser would build 200k+ nodes). bbox-normalised later, so
// per-item transforms don't matter for a thumbnail.
const _VTX_RE = /<vertex\s+x="([-+\deE.]+)"\s+y="([-+\deE.]+)"\s+z="([-+\deE.]+)"/g;
const _TRI_RE = /<triangle\s+v1="(\d+)"\s+v2="(\d+)"\s+v3="(\d+)"/g;
function _parse3mfDirect(buf) {
  let files;
  try { files = fflate.unzipSync(new Uint8Array(buf)); }
  catch (e) { console.warn('[3mf] unzip failed', e); return null; }
  const group = new THREE.Group();
  for (const name in files) {
    if (!name.toLowerCase().endsWith('.model')) continue;
    const xml = new TextDecoder().decode(files[name]);
    if (xml.indexOf('<vertex') === -1) continue;     // root / metadata model
    const verts = [];
    let m;
    _VTX_RE.lastIndex = 0;
    while ((m = _VTX_RE.exec(xml)) !== null) verts.push(+m[1], +m[2], +m[3]);
    const idx = [];
    _TRI_RE.lastIndex = 0;
    while ((m = _TRI_RE.exec(xml)) !== null) idx.push(+m[1], +m[2], +m[3]);
    if (verts.length >= 9 && idx.length >= 3) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(verts), 3));
      g.setIndex(idx);
      group.add(new THREE.Mesh(g));
    }
  }
  return group.children.length ? group : null;
}

async function _loadObject(file) {
  const ext = _meshExt(file.name);
  const buf = await file.arrayBuffer();
  if (ext === 'stl') return new THREE.Mesh(new STLLoader().parse(buf));
  if (ext === 'ply') return new THREE.Mesh(new PLYLoader().parse(buf));
  if (ext === 'obj') return new OBJLoader().parse(new TextDecoder().decode(buf));
  if (ext === '3mf') {
    // Direct regex parse FIRST — fast and robust on the split-model AI 3MF that
    // blanks ThreeMFLoader. Fall back to the full loader only if that finds no
    // mesh (exotic 3MF: components/transforms the regex path doesn't follow).
    const direct = _parse3mfDirect(buf);
    if (_hasMesh(direct)) return direct;
    try {
      const obj = new ThreeMFLoader().parse(buf.slice(0));
      if (_hasMesh(obj)) return obj;
    } catch (e) {
      console.warn('[3mf] both parsers failed:', file.name, e);
    }
    return direct;   // may be null -> placeholder
  }
  return null;
}

function _disposeObj(obj) {
  obj.traverse(c => {
    if (c.geometry) c.geometry.dispose();
    if (c.material) { (Array.isArray(c.material) ? c.material : [c.material]).forEach(m => m.dispose()); }
  });
}

// Render any loaded Object3D to a 128px thumbnail dataURL (centred + normalised
// in world space so 3MF child transforms are honoured). Disposes the object.
function _renderObjThumb(obj) {
  obj.updateMatrixWorld(true);
  let hasMesh = false;
  obj.traverse(c => {
    if (c.isMesh && c.geometry && c.geometry.attributes.position) {
      hasMesh = true;
      if (!c.geometry.attributes.normal) c.geometry.computeVertexNormals();
      c.material = new THREE.MeshStandardMaterial({ color: 0x9AA4AE, metalness: 0.1, roughness: 0.72 });
    }
  });
  if (!hasMesh) { _disposeObj(obj); return null; }
  const box = new THREE.Box3().setFromObject(obj);
  if (box.isEmpty()) { _disposeObj(obj); return null; }
  const c = box.getCenter(new THREE.Vector3());
  const sz = box.getSize(new THREE.Vector3());
  const scl = 1.7 / (Math.max(sz.x, sz.y, sz.z) || 1);
  obj.position.sub(c);

  const grp = new THREE.Group();
  grp.add(obj);
  grp.scale.setScalar(scl);
  grp.rotation.x = -Math.PI / 2 + 0.5;
  grp.rotation.z = 0.5;

  const scene = new THREE.Scene();
  scene.add(grp);
  scene.add(new THREE.HemisphereLight(0xE0E8F0, 0x3D4450, 1.2));
  const key = new THREE.DirectionalLight(0xffffff, 1.1); key.position.set(2, 3, 2); scene.add(key);
  const cam = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
  cam.position.set(0, 0, 3); cam.lookAt(0, 0, 0);
  const r = _getThumbRenderer();
  r.render(scene, cam);
  const url = r.domElement.toDataURL('image/png');
  _disposeObj(obj);
  return url;
}

async function _thumbnailFor(file) {
  // Thumbnails are generated PURELY in the browser — the folder's meshes are
  // never uploaded. (Only the one mesh you click to actually work on is sent to
  // the server.) If a file can't be parsed client-side, a placeholder shows.
  try {
    const obj = await _loadObject(file);
    if (obj) return _renderObjThumb(obj);
  } catch (e) {
    console.warn('[folder] client-side thumbnail failed for', file.name, e);
  }
  return null;
}

// Show raw mesh triangles in the viewer (before pipeline)
function _showRawMesh(previewRes, animateToFit = true) {
  console.debug('[_showRawMesh] verts:', previewRes.mesh_vertices?.length, 'faces:', previewRes.mesh_faces?.length);
  // Hide feature tree during mesh preview — it shows after pipeline completes
  // (the folder browser is a separate left panel, unaffected by this).
  document.getElementById('sidebar').classList.add('closed');
  while (modelGroup.children.length) modelGroup.remove(modelGroup.children[0]);

  try {
    const verts = new Float32Array(previewRes.mesh_vertices);
    const faces = previewRes.mesh_faces;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    geo.setIndex(faces);
    geo.computeVertexNormals();

    // Mesh preview — wireframe only for low-poly meshes, solid for dense scans
    const numFaces = Math.floor(faces.length / 3);
    const mat = new THREE.MeshStandardMaterial({ color: 0x8B949E, metalness: 0.1, roughness: 0.8 });
    modelGroup.add(new THREE.Mesh(geo, mat));
    if (numFaces < 20000) {
      modelGroup.add(new THREE.LineSegments(
        new THREE.WireframeGeometry(geo),
        new THREE.LineBasicMaterial({ color: 0x30363D, linewidth: 1 })
      ));
    }
    console.debug('[_showRawMesh] created mesh + wireframe objects');
  } catch (e) {
    console.error('[_showRawMesh] geometry creation failed', e);
    return;
  }

  // Center model at origin — reset position first to avoid cumulative drift
  try {
    modelGroup.position.set(0, 0, 0);
    const box = new THREE.Box3().setFromObject(modelGroup);
    const center = box.getCenter(new THREE.Vector3());
    modelGroup.position.sub(center);
    const size = box.getSize(new THREE.Vector3()).length();

    // Smooth animation only on first upload, not when clicking Mesh button
    if (animateToFit) {
      const pos = bestCameraPos(box);
      console.debug('[_showRawMesh] centering, bbox size:', size.toFixed(1), 'animating camera to face largest profile');
      animateCamera(pos, { x: 0, y: 0, z: 0 }, 1.5);
    } else {
      controls.target.set(0, 0, 0);
      controls.update();
    }
  } catch (e) {
    console.error('[_showRawMesh] bounding box / camera setup failed', e);
  }
}

// ── Variant slider ──
document.getElementById('variant-slider')?.addEventListener('input', (e) => {
  document.getElementById('variant-count').textContent = e.target.value;
});

// ── Resize dimension gate ──
let _dimGateResolve = null;
window.cancelDimGate = function() {
  document.getElementById('dim-gate').classList.remove('show');
  if (_dimGateResolve) _dimGateResolve(null);
  _dimGateResolve = null;
};
window.confirmDimGate = function() {
  const w = parseFloat(document.getElementById('gate-width').value) || 0;
  const h = parseFloat(document.getElementById('gate-height').value) || 0;
  const t = parseFloat(document.getElementById('gate-thickness').value) || 0;
  document.getElementById('dim-gate').classList.remove('show');
  if (_dimGateResolve) _dimGateResolve({ width: w, height: h, thickness: t });
  _dimGateResolve = null;
};

function showDimGate(previewDims) {
  // Pre-fill with current mesh dimensions
  if (previewDims && previewDims.length === 3) {
    const sorted = [...previewDims].sort((a, b) => b - a);
    document.getElementById('gate-width').value = sorted[0].toFixed(1);
    document.getElementById('gate-height').value = sorted[1].toFixed(1);
    document.getElementById('gate-thickness').value = sorted[2].toFixed(1);
  }
  document.getElementById('dim-gate').classList.add('show');
  return new Promise(resolve => { _dimGateResolve = resolve; });
}

// Analyze & Build button handler
let _analyzeRunning = false;
document.getElementById('btn-analyze').addEventListener('click', async () => {
  if (!pendingTempPath || _analyzeRunning) return;

  // Check if resize is enabled — show dimension gate
  const resizeOn = document.getElementById('resize-toggle')?.checked;
  if (resizeOn) {
    const dims = await showDimGate(window._lastPreviewDims || []);
    if (!dims) return; // user cancelled
    window._editedDims = dims;
    console.debug('[analyze] resize dimensions:', dims);
  }

  _analyzeRunning = true;
  _lastProgressStage = '';
  window._pipelineCancelled = false;
  window._pipelineRotate = true;
  window._explodePiecesVersion = -1;
  window._lastTempPath = pendingTempPath; // save before pipeline might clear it
  console.debug('[analyze] starting pipeline, temp_path:', pendingTempPath);

  // Clear any stale mesh from a previous pipeline run (e.g. peeled +
  // decomposed bodies) and redraw the uploaded preview. Ingest can take
  // 20+ seconds due to isotropic remeshing; without this the old
  // pipeline's final geometry lingers on screen, which looks like the
  // pre-peel stage is showing a peeled mesh.
  if (savedMeshPreview) {
    _showRawMesh(savedMeshPreview, false);
  }

  // Hide UI elements during generation
  document.getElementById('actions').style.display = 'none';
  document.getElementById('timeline').style.display = 'none';
  document.getElementById('play-btn').style.display = 'none';
  document.getElementById('preview-bar').style.display = 'none';

  const loading = document.getElementById('loading');
  const loadingText = document.getElementById('loading-text');
  const progressBar = document.getElementById('progress-bar');
  const progressDetail = document.getElementById('progress-detail');
  loading.classList.add('show');
  loadingText.textContent = 'Running pipeline...';
  progressBar.style.width = '0%';
  progressDetail.textContent = '';
  document.getElementById('status-dot').className = 'status-dot processing';

  try {
    // Submit pipeline job
    const res = await fetch('/api/run-pipeline', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        temp_path: pendingTempPath,
        filename: pendingFile?.name || 'mesh',
        num_variants: parseInt(document.getElementById('variant-slider')?.value || '1'),
        target_dims: window._editedDims || null,
        // Authoritative at submit time — avoids the post-submit POST
        // racing the worker (and a stale page never toggling at all).
        step_through: document.getElementById('step-through-cb')?.checked ?? true,
      }),
    });
    if (!res.ok) {
      // Surface the server's reason (e.g. free-trial/monthly-limit on 403)
      // instead of a bare status code.
      let reason = 'HTTP ' + res.status;
      try { const ed = await res.json(); if (ed && ed.error) reason = ed.error; } catch (e) {}
      console.error('[analyze] /api/run-pipeline failed:', res.status, reason);
      alert('Pipeline request failed: ' + reason);
      loading.classList.remove('show');
      return;
    }
    let data;
    try { data = await res.json(); } catch (e) { console.error('[analyze] JSON parse failed from /api/run-pipeline', e); alert('Pipeline response invalid'); loading.classList.remove('show'); return; }
    if (data.error) {
      console.error('[analyze] pipeline error:', data.error);
      loading.classList.remove('show');
      if (data.error.includes('not found') || data.error.includes('Upload first')) {
        // Stale temp path from previous session — clear and prompt re-upload
        sessionStorage.removeItem('paramesh_preview');
        pendingTempPath = null;
        document.getElementById('btn-analyze').style.display = 'none';
        document.getElementById('btn-upload').style.display = '';
        addChatMessage('system', 'Session expired. Please re-upload your mesh file.');
      } else {
        alert(data.error);
      }
      return;
    }

    currentJobId = data.job_id;
    console.debug('[analyze] job submitted, id:', currentJobId);

    // Send step-through preference to backend
    const stepCb = document.getElementById('step-through-cb');
    const stepOn = stepCb ? stepCb.checked : true;
    fetch(`/api/job/${currentJobId}/step-through/${stepOn ? '1' : '0'}`, { method: 'POST' });

    // Send the one-shot breakpoint stage — but ONLY while its control is
    // actually active (Debug on + step mode off). The dropdown persists in
    // localStorage, so without the debug gate a breakpoint set during a
    // debug session kept silently pausing every later run with the
    // control hidden.
    const pauseSel = document.getElementById('pause-at-select');
    const debugOn = document.getElementById('debug-cb')?.checked;
    const pauseAt = (!stepOn && debugOn && pauseSel) ? pauseSel.value : 'none';
    fetch(`/api/job/${currentJobId}/pause-at/${pauseAt}`, { method: 'POST' });

    addChatMessage('system', 'Pipeline started — analyzing mesh...');
    let lastStage = '';

    // Poll progress and stream thinking to chat
    // Capture this invocation's job ID so zombie polls from previous analyze
    // calls don't interfere when this job gets cancelled/replaced
    const myJobId = currentJobId;
    // Stuck-detection state: when the progress percentage stops moving
    // for too long, surface a "Force reset queue" button so the user
    // isn't trapped staring at a frozen spinner.
    const _analyzeStartedAt = Date.now();
    let _lastProgress = -1;
    let _lastProgressAt = Date.now();
    // Fresh run — drop any stale right-pane build from a previous analyze.
    try { LiveBuild.reset(); LiveBuild.hide(); } catch (e) {}
    const statusData = await new Promise((resolve, reject) => {
      const poll = setInterval(async () => {
        try {
          // Exit if this poll's job is no longer the current one
          // (either cancelled or replaced by a newer analyze)
          if (myJobId !== currentJobId) {
            clearInterval(poll);
            reject(new Error('Cancelled'));
            return;
          }
          if (!myJobId || myJobId === 'null') {
            clearInterval(poll);
            reject(new Error('Cancelled'));
            return;
          }
          const r = await fetch(`/api/job/${myJobId}/status`);
          const d = await r.json();
          progressBar.style.width = d.progress + '%';
          loadingText.textContent = d.stage || 'Processing...';
          progressDetail.textContent = d.progress + '%';

          // Show LLM build strategy when it becomes available
          if (d.stage && d.stage !== lastStage) {
            lastStage = d.stage;
            if (d.variants && d.variants.length > 0 && lastStage === 'Building CAD model...') {
              const strategy = d.variants[0].strategy;
              if (strategy) addChatMessage('ai', 'Build strategy: ' + strategy);
            }
          }

          // Render progress geometry on the 3D model (LEFT pane = analysis).
          // Skip while the user has PINNED an older stage card (not following),
          // so their pinned visual isn't overwritten every 500ms.
          if (d.progress_geometry && stagesFollowing()) {
            _renderProgressGeometry(d.progress_geometry);
          }

          // RIGHT pane = CAD model accreting live. As soon as the build
          // subprocess streams its first construction step, split the viewer
          // and accrete features there in real time. Both panes advance on
          // their own — the left through analysis stages, the right through
          // construction steps.
          // Only split into the dual pane once there's an ACTUAL construction
          // step to show. Splitting on building_live alone (set the instant the
          // CAD subprocess gets its input, before the first step streams) left
          // a blank right pane — most visible in step mode, where the gap is
          // long. Wait for num_steps > 0 so the right pane always has content.
          if (d.status === 'processing' && (d.num_steps || 0) > 0) {
            LiveBuild.show();
            LiveBuild.update(myJobId, d.num_steps || 0, d.steps_version || 0);
          }

          // Stream construction stages into the right-sidebar Stages panel
          // (cards pop in, newest shows a loading bar). Runs every poll so the
          // final 'done' tick clears the loader and shows all ticks.
          updateStagesPanel(myJobId, d);

          // Update debug summary panel (visible only when Debug toggle is on).
          // Group lines per body so the user can scan one body's stages at a
          // time. Each line: "[Body N] Stage X: rest" → split into a tag
          // ("S2"/"S3"/"S4"/"S5") and a body string. "misfit" gets warn
          // styling so problems pop visually.
          if (Array.isArray(d.debug_summary)) {
            const container = document.getElementById('debug-lines');
            if (container) {
              const byBody = new Map();
              for (const line of d.debug_summary) {
                const m = line.match(/^\[Body (\d+)\] Stage (\d+): (.*)$/);
                if (!m) continue;
                const [, bodyN, stageN, rest] = m;
                if (!byBody.has(bodyN)) byBody.set(bodyN, []);
                byBody.get(bodyN).push({ stage: stageN, body: rest });
              }
              const escapeHtml = (s) => s
                .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
              let html = '';
              for (const [bodyN, stages] of byBody) {
                html += `<div class="dbg-body"><div class="dbg-body-title">Body ${bodyN}</div>`;
                for (const { stage, body } of stages) {
                  let bodyHtml = escapeHtml(body)
                    .replace(/\b(\d+)\b/g, '<span class="num">$1</span>')
                    .replace(/(misfit[^,)]*)/g, '<span class="warn">$1</span>');
                  html += `<div class="dbg-stage"><span class="dbg-stage-tag">S${stage}</span><span class="dbg-stage-body">${bodyHtml}</span></div>`;
                }
                html += '</div>';
              }
              container.innerHTML = html || '<div style="color:var(--text-muted);font-style:italic;">Waiting for pipeline…</div>';
            }
          }

          // Detect stuck-at-queued / no-progress and surface a
          // "Force reset queue" escape hatch. The backend has its own
          // 3-minute watchdog, but the frontend hint shows up after
          // 30 s of zero progress so the user isn't left staring at a
          // spinner.
          const stuckBtnHost = document.getElementById('btn-cancel');
          const isStuck = (
            (d.status === 'queued' && (Date.now() - _analyzeStartedAt) > 30000)
            || (d.status === 'processing' && !d.paused
                && d.progress === _lastProgress
                && (Date.now() - _lastProgressAt) > 60000)
          );
          if (d.progress !== _lastProgress) {
            _lastProgress = d.progress;
            _lastProgressAt = Date.now();
          }
          if (isStuck && stuckBtnHost) {
            if (!document.getElementById('btn-force-reset')) {
              const rBtn = document.createElement('button');
              rBtn.id = 'btn-force-reset';
              rBtn.textContent = 'Force reset queue';
              rBtn.title = 'Kill the stuck worker thread and start a fresh one';
              rBtn.style.cssText = 'margin-left:8px;padding:4px 14px;border-radius:6px;border:1px solid #f55;background:#3a0e0e;color:#ff9090;font-size:12px;font-weight:600;cursor:pointer;font-family:OpenAI Sans,sans-serif;';
              rBtn.onclick = async () => {
                rBtn.disabled = true;
                rBtn.textContent = 'Resetting...';
                try {
                  await fetch('/api/queue/reset', { method: 'POST' });
                  setTimeout(() => location.reload(), 700);
                } catch (e) {
                  console.error('[reset] failed:', e);
                  rBtn.disabled = false;
                  rBtn.textContent = 'Force reset queue';
                }
              };
              stuckBtnHost.parentElement.appendChild(rBtn);
            }
          } else {
            const rBtn = document.getElementById('btn-force-reset');
            if (rBtn) rBtn.remove();
          }

          // Show Continue button when pipeline is paused
          const cancelBtn = document.getElementById('btn-cancel');
          if (d.paused && cancelBtn) {
            if (!document.getElementById('btn-continue-pipeline')) {
              const contBtn = document.createElement('button');
              contBtn.id = 'btn-continue-pipeline';
              contBtn.textContent = 'Continue';
              contBtn.style.cssText = 'margin-left:8px;padding:4px 16px;border-radius:6px;border:none;background:var(--accent);color:#000;font-size:12px;font-weight:600;cursor:pointer;font-family:OpenAI Sans,sans-serif;';
              contBtn.onclick = () => {
                fetch(`/api/job/${currentJobId}/continue`, { method: 'POST' });
                contBtn.remove();
              };
              cancelBtn.parentElement.appendChild(contBtn);
            }
            // Hide spinner during pause
            const spinner = document.querySelector('#loading .spinner');
            if (spinner) spinner.style.display = 'none';
          } else {
            const contBtn = document.getElementById('btn-continue-pipeline');
            if (contBtn) contBtn.remove();
            const spinner = document.querySelector('#loading .spinner');
            if (spinner) spinner.style.display = '';
          }

          if (_cancelledJobId === myJobId) { clearInterval(poll); reject(new Error('Cancelled')); return; }
          if (d.status === 'done') {
            clearInterval(poll);
            // Keep the live pane: expand it to full + recenter the finished
            // model (no reload flash). Only expand if it was actually showing.
            if (LiveBuild.isActive()) LiveBuild.expandToFull();
            resolve(d);
          }
          else if (d.status === 'error') { clearInterval(poll); LiveBuild.dismiss(); reject(new Error(d.error)); }
        } catch (e) { console.warn('[analyze] poll error:', e); }
      }, 500);
    });

    progressBar.style.width = '90%';
    loadingText.textContent = 'Loading model...';

    // Hide preview bar, show upload button again
    document.getElementById('preview-bar').style.display = 'none';
    document.getElementById('btn-analyze').style.display = 'none';
    document.getElementById('btn-upload').style.display = '';
    window._lastTempPath = pendingTempPath; // keep for resize
    pendingTempPath = null;

    if (statusData.status === 'done') {
      // Summary message in chat
      const holes = statusData.steps.filter(s => s.name.includes('Hole')).length;
      const chamfers = statusData.steps.filter(s => s.name.includes('Chamfer')).length;
      const fillets = statusData.steps.filter(s => s.name.includes('Fillet')).length;
      let summary = `Reconstructed with ${statusData.num_steps} features: 1 base extrusion`;
      if (holes > 0) summary += `, ${holes} hole${holes > 1 ? 's' : ''}`;
      if (chamfers > 0) summary += `, ${chamfers} chamfer${chamfers > 1 ? 's' : ''}`;
      if (fillets > 0) summary += `, ${fillets} fillet${fillets > 1 ? 's' : ''}`;
      addChatMessage('ai', summary);

      // Populate model stats panel
      document.getElementById('stat-features').textContent = statusData.num_steps + ' features';
      document.getElementById('stat-volume').textContent = statusData.mesh_volume.toLocaleString() + ' mm³';
      document.getElementById('model-stats').style.display = '';

      // Run slicer comparison in background and show accuracy
      if (currentJobId) {
        console.debug('[analyze] requesting slicer comparison for job:', currentJobId);
        fetch(`/api/job/${currentJobId}/compare`).then(r => {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        }).then(comp => {
          if (comp.score !== undefined) {
            console.debug('[analyze] comparison score:', comp.score.toFixed(1), '%, slices:', comp.per_slice?.length);
            document.getElementById('stat-accuracy').textContent = comp.score.toFixed(0) + '% (' + (comp.per_slice?.length || 0) + ' slices)';
          }
        }).catch(e => { console.warn('[analyze] slicer comparison failed (non-critical):', e.message); });
      }

      // AI dimension analysis in background — populates panel when ready
      fetchDimensionAnalysis(statusData);

      // finishLoad will dismiss the loading overlay after the 3D model renders
      finishLoad(statusData);

      // Reveal / relabel the "Open in Onshape" button now that this mesh has
      // steps, then — if its document is already open — silently re-sync it.
      try { updateOnshapeOpenButton && updateOnshapeOpenButton(); } catch (e) {}
      try { window.maybeAutoBuildOnshape && window.maybeAutoBuildOnshape(); } catch (e) {}
    } else {
      alert('Pipeline error: ' + (data.error || 'unknown'));
      document.getElementById('status-dot').className = 'status-dot';
    }
  } catch (err) {
    if (err.message === 'Cancelled') {
      console.debug('[analyze] pipeline cancelled');
    } else {
      console.error('[analyze] pipeline error:', err);
      alert('Pipeline error: ' + err.message);
    }
    document.getElementById('status-dot').className = 'status-dot';
    loading.classList.remove('show');
  } finally {
    _analyzeRunning = false;
    // Keep _pipelineRotate true — finishLoad will stop it after model renders
    // Restore UI elements
    document.getElementById('actions').style.display = '';
    document.getElementById('timeline').style.display = '';
  }
});

// Cancel pipeline
let _cancelledJobId = null;
window._cancelPipeline = function() {
  console.debug('[cancel] starting, currentJobId:', currentJobId,
                'savedMeshPreview:', savedMeshPreview ? 'set' : 'null',
                '_lastTempPath:', window._lastTempPath);
  _cancelledJobId = currentJobId;
  window._pipelineCancelled = true;  // poll/render check this
  _analyzeRunning = false;
  window._pipelineRotate = false;
  window._idleDist = null;

  // Fire-and-forget cancel on the server
  if (currentJobId) {
    fetch(`/api/job/${currentJobId}/cancel`, { method: 'POST' }).catch(() => {});
  }

  // Reset UI
  document.getElementById('loading').classList.remove('show');
  document.getElementById('actions').style.display = '';
  document.getElementById('timeline').style.display = '';
  const timeline = document.getElementById('timeline');
  Array.from(timeline.children).forEach(ch => {
    if (ch.classList && ch.classList.contains('timeline-step')) ch.remove();
  });
  document.getElementById('status-dot').className = 'status-dot';
  const spinner = document.querySelector('#loading .spinner');
  if (spinner) spinner.style.display = '';
  const contBtn = document.getElementById('btn-continue-pipeline');
  if (contBtn) contBtn.remove();
  document.getElementById('sidebar').classList.add('closed');
  document.getElementById('feature-list').innerHTML = '';
  document.getElementById('play-btn').style.display = 'none';

  // Clear stored cad job so refresh doesn't restore old model
  localStorage.removeItem('paramesh_last_step');
  sessionStorage.removeItem('paramesh_mesh_compare');
  currentJobId = null;

  // Restore the viewer to the original uploaded mesh.
  // Priority: in-memory savedMeshPreview → sessionStorage preview.
  let preview = savedMeshPreview;
  if (!preview) {
    try {
      const saved = sessionStorage.getItem('paramesh_preview');
      if (saved) preview = JSON.parse(saved);
    } catch (e) {}
  }
  console.debug('[cancel] preview to restore:', preview ? `${preview.mesh_vertices?.length} verts` : 'NONE');

  // Clear modelGroup and rebuild the mesh
  while (modelGroup.children.length) {
    const child = modelGroup.children[0];
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else child.material.dispose();
    }
    modelGroup.remove(child);
  }

  if (preview && preview.mesh_vertices && preview.mesh_vertices.length > 0) {
    savedMeshPreview = preview;
    document.getElementById('empty-state').style.display = 'none';
    _showRawMesh(preview, true);
    pendingTempPath = window._lastTempPath || preview.temp_path || pendingTempPath;
    if (pendingTempPath) {
      document.getElementById('btn-analyze').style.display = '';
    }
    // Persist preview so hard refresh restores it
    try {
      sessionStorage.setItem('paramesh_preview', JSON.stringify(preview));
    } catch (e) {}
    console.debug('[cancel] restored raw mesh, modelGroup children:', modelGroup.children.length);
  } else {
    console.warn('[cancel] no preview available to restore, showing empty state');
    document.getElementById('empty-state').style.display = '';
  }

  addChatMessage('system', 'Pipeline cancelled.');
};

// Rerun pipeline with the same mesh
window.rerunPipeline = function() {
  // After a successful run the analyze handler clears pendingTempPath
  // but stashes the value on window._lastTempPath so the same mesh can
  // be re-processed without re-uploading.
  if (!pendingTempPath) {
    pendingTempPath = window._lastTempPath || null;
  }
  if (!pendingTempPath) return;
  document.getElementById('btn-rerun').style.display = 'none';
  document.getElementById('btn-analyze').style.display = '';
  document.getElementById('btn-analyze').click();
};

let cachedStatusData = null;

function buildDimsFromStatus(statusData) {
  const raw = [0,1,2].map(i => Math.abs(statusData.bbox_max[i] - statusData.bbox_min[i]));
  raw.sort((a,b) => b - a);
  const dims = {
    width: Math.round(raw[0] * 10) / 10,
    height: Math.round(raw[1] * 10) / 10,
    thickness: Math.round(raw[2] * 10) / 10,
  };
  // Extract hole diameters from step names like "Blind Hole (r=2.5...)"
  if (statusData.steps) {
    const radii = [];
    for (const s of statusData.steps) {
      const m = s.name.match(/[Hh]ole.*r=([\d.]+)/);
      if (m) radii.push(parseFloat(m[1]));
    }
    if (radii.length > 0) {
      dims.hole_diameters = radii.map(r => Math.round(r * 2 * 10) / 10);
    }
  }
  return { part_type: 'Detected', suggested_dimensions: dims, notes: '' };
}

function finishLoad(statusData) {
  console.debug('[finishLoad] state -> cad, steps:', statusData.steps?.length, 'volume:', statusData.mesh_volume, 'tokens:', statusData.tokens_used);
  // Show feature tree now that pipeline is done
  document.getElementById('sidebar').classList.remove('closed');
  cachedStatusData = statusData;
  saveViewerState('cad');
  sessionStorage.removeItem('paramesh_preview'); // clear preview, CAD is now primary
  // Persist mesh data for Mesh/Compare buttons across refreshes
  if (savedMeshPreview) {
    try { sessionStorage.setItem('paramesh_mesh_compare', JSON.stringify(savedMeshPreview)); } catch (e) { /* too large */ }
  }
  populateUI(statusData);

  // Mark which mesh this job built, so the Onshape button knows the job is
  // valid for the loaded mesh (vs a stale job after a new upload).
  _analyzedMeshName = (statusData && statusData.filename) || currentMeshName() || '';

  // Reveal the "Open in Onshape" button now that this mesh has steps
  // (covers restored sessions and the demo, not just a fresh analysis).
  try { updateOnshapeOpenButton && updateOnshapeOpenButton(); } catch (e) {}

  // Show dimension panel from bbox (AI analysis will upgrade it if available)
  if (statusData.bbox_min && statusData.bbox_max) {
    populateDimPanel(buildDimsFromStatus(statusData));
  }
  // Update parts counter from usage endpoint
  fetch('/api/usage').then(r => r.ok ? r.json() : null).then(u => {
    if (!u) return;
    const txt = u.parts_limit > 0 ? `${u.parts_used} / ${u.parts_limit}` : `${u.parts_used}`;
    document.getElementById('token-count').textContent = txt;
    localStorage.setItem('paramesh_parts_count', txt);
  }).catch(() => {});
  document.getElementById('status-dot').className = 'status-dot done';
  document.getElementById('btn-export').disabled = false;
  document.getElementById('btn-rerun').style.display =
    (pendingTempPath || window._lastTempPath) ? '' : 'none';
  document.getElementById('play-btn').style.display = '';

  // Enable chat and dimension editing
  document.getElementById('chat-input').disabled = false;
  document.getElementById('chat-send').disabled = false;
  const editBtn = document.getElementById('dim-edit-btn');
  if (editBtn) editBtn.style.display = '';

  // Add system message to chat
  addChatMessage('system', `Model loaded: ${statusData.num_steps} features, ${statusData.mesh_volume.toLocaleString()} mm³`);

  // Show model stats
  document.getElementById('stat-features').textContent = statusData.num_steps + ' features';
  document.getElementById('stat-volume').textContent = statusData.mesh_volume.toLocaleString() + ' mm³';
  document.getElementById('model-stats').style.display = '';

  const loading = document.getElementById('loading');
  const progressBar = document.getElementById('progress-bar');

  if (statusData.steps.length > 0) {
    loadStep(statusData.steps.length - 1).then(() => {
      // Dismiss loading overlay and stop idle rotation now that 3D model is rendered
      progressBar.style.width = '100%';
      loading.classList.remove('show');
      window._pipelineRotate = false;
      // Animate camera only when CAD model first loads — but NOT while the
      // live-build pane is showing the model: the main viewer is hidden behind
      // it and will inherit the live pane's exact camera on handoff (dismiss).
      try {
        if (!LiveBuild.isActive()) {
          const box = new THREE.Box3().setFromObject(modelGroup);
          const pos = bestCameraPos(box);
          console.debug('[finishLoad] model loaded, animating camera to face largest profile');
          animateCamera(pos, { x: 0, y: 0, z: 0 }, 1.2);
        }
        // (Removed: the ending auto-play build animation. The live pane already
        // animates the build as it streams; replaying it at the end was
        // redundant and — in live-full mode — drew into the hidden main viewer
        // so the visual never changed. The manual Play button still works.)
      } catch (e) {
        console.error('[finishLoad] camera animation failed', e);
      }
    });
  } else {
    loading.classList.remove('show');
  }
}

// ── Inline Dimension Panel ──
async function fetchDimensionAnalysis(statusData) {
  let analysis;
  try {
    console.debug('[analyze] requesting AI dimension analysis');
    const res = await fetch('/api/chat/analyze-mesh', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        bbox_min: statusData.bbox_min,
        bbox_max: statusData.bbox_max,
        volume: statusData.mesh_volume,
        num_faces: statusData.num_steps,
        features: statusData.steps.map(s => ({ type: s.name, depth: 0 })),
      }),
    });
    if (!res.ok) throw new Error('HTTP ' + res.status);
    analysis = await res.json();
    if (analysis.skip) throw new Error('skip');
  } catch (err) {
    console.warn('[analyze] AI dimension analysis unavailable, falling back to bbox:', err.message);
    if (statusData.bbox_min && statusData.bbox_max) {
      const dims = [0,1,2].map(i => Math.abs(statusData.bbox_max[i] - statusData.bbox_min[i]));
      dims.sort((a,b) => b - a);
      analysis = {
        part_type: 'Prismatic part',
        suggested_dimensions: {
          width: Math.round(dims[0] * 10) / 10,
          height: Math.round(dims[1] * 10) / 10,
          thickness: Math.round(dims[2] * 10) / 10,
        },
        notes: 'Dimensions from bounding box. AI analysis unavailable.',
      };
    } else {
      return; // No dimension data at all
    }
  }
  populateDimPanel(analysis);
}

function populateDimPanel(analysis) {
  console.debug('[dimPanel] part_type:', analysis.part_type, 'dims:', Object.keys(analysis.suggested_dimensions || {}));
  const panel = document.getElementById('dim-panel');
  const rows = document.getElementById('dim-rows');

  document.getElementById('dim-part-type').textContent = analysis.part_type || 'Unknown';

  rows.innerHTML = '';
  const dims = analysis.suggested_dimensions || {};
  for (const [key, val] of Object.entries(dims)) {
    // For hole_diameters: summarize as count × unique sizes instead of listing all
    if (key === 'hole_diameters' && Array.isArray(val)) {
      const nums = val.map(v => typeof v === 'number' ? v : parseFloat(v)).filter(v => !isNaN(v));
      if (nums.length === 0) continue;
      // Group by diameter: "4 × ø8, 2 × ø5"
      const counts = {};
      nums.forEach(d => { const k = d.toFixed(1); counts[k] = (counts[k] || 0) + 1; });
      const summary = Object.entries(counts)
        .sort((a, b) => parseFloat(b[0]) - parseFloat(a[0]))
        .map(([d, n]) => `${n}× ø${d}`)
        .join(', ');
      rows.innerHTML += `<div class="dim-row"><span class="label">holes</span><span class="value" style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--text-primary)">${summary} mm</span></div>`;
      continue;
    }
    const displayVal = Array.isArray(val) ? val.map(v => typeof v === 'number' ? v : parseFloat(v)).join(', ') : val;
    rows.innerHTML += `<div class="dim-row"><span class="label">${key}</span><span class="value"><input class="dim-input" type="text" data-key="${key}" value="${displayVal}" readonly> mm</span></div>`;
  }

  if (analysis.usage) {
    document.getElementById('token-count').textContent = analysis.usage.calls_used || 0;
  }

  panel.style.display = '';
  panel.classList.remove('collapsed');

  // Persist to localStorage
  try { localStorage.setItem('paramesh_dims', JSON.stringify(analysis)); } catch (e) {}
}

window.toggleDimPanel = function() {
  document.getElementById('dim-panel').classList.toggle('collapsed');
};

window.toggleDimEdit = async function() {
  const inputs = document.querySelectorAll('#dim-rows .dim-input');
  const btn = document.getElementById('dim-edit-btn');
  const isReadonly = inputs[0]?.readOnly;

  // When clicking Done — resize the mesh live
  if (!isReadonly) {
    const dimValues = {};
    inputs.forEach(inp => { dimValues[inp.dataset.key] = parseFloat(inp.value) || 0; });
    window._editedDims = dimValues;
    console.debug('[resize] dimValues:', dimValues, 'pendingTempPath:', pendingTempPath, '_lastTempPath:', window._lastTempPath);

    const tempPath = pendingTempPath || window._lastTempPath;
    if (tempPath && (dimValues.width || dimValues.height || dimValues.thickness)) {
      btn.innerHTML = '<i class="ri-loader-4-line"></i> ...';
      btn.disabled = true;
      try {
        const res = await fetch('/api/resize-mesh', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            temp_path: tempPath,
            width: dimValues.width || null,
            height: dimValues.height || null,
            thickness: dimValues.thickness || null,
          }),
        });
        const data = await res.json();
        console.debug('[resize] response:', res.status, data);
        if (res.ok && data.dimensions) {
          // Update dimension inputs with actual resized values
          const sorted = data.dimensions;
          const keys = ['width', 'height', 'thickness'];
          inputs.forEach(inp => {
            const ki = keys.indexOf(inp.dataset.key);
            if (ki >= 0 && ki < sorted.length) inp.value = sorted[ki];
          });
          // Apply per-axis scale to the 3D model (non-uniform)
          if (data.scale_xyz) {
            const s = data.scale_xyz;
            modelGroup.scale.set(
              modelGroup.scale.x * s[0],
              modelGroup.scale.y * s[1],
              modelGroup.scale.z * s[2]
            );
          }
          window._lastPreviewDims = sorted;
          addChatMessage('system', 'Resized to ' + sorted.map(d => d + 'mm').join(' × '));
        } else {
          addChatMessage('system', 'Resize failed: ' + (data.error || 'unknown'));
        }
      } catch (e) {
        console.error('[resize] failed:', e);
        addChatMessage('system', 'Resize failed: ' + e.message);
      }
      btn.disabled = false;
    }
  }

  // Toggle readonly state
  inputs.forEach(inp => {
    inp.readOnly = !isReadonly;
    inp.style.background = isReadonly ? 'var(--bg-tertiary)' : 'transparent';
    inp.style.borderColor = isReadonly ? 'var(--accent)' : 'var(--border)';
  });
  btn.innerHTML = isReadonly ? '<i class="ri-check-line"></i> Done' : '<i class="ri-pencil-line"></i> Edit';
  btn.style.borderColor = isReadonly ? 'var(--accent)' : 'var(--border)';
  btn.style.color = isReadonly ? 'var(--accent)' : 'var(--text-muted)';
};

// Restore dimension panel from localStorage on load
try {
  const savedDims = localStorage.getItem('paramesh_dims');
  if (savedDims) {
    const analysis = JSON.parse(savedDims);
    if (analysis.suggested_dimensions) populateDimPanel(analysis);
  }
} catch (e) {}

// Persist and restore variant slider + resize toggle
document.getElementById('variant-slider')?.addEventListener('change', (e) => {
  localStorage.setItem('paramesh_num_variants', e.target.value);
});
document.getElementById('resize-toggle')?.addEventListener('change', (e) => {
  localStorage.setItem('paramesh_resize_on', e.target.checked ? '1' : '0');
});
try {
  const savedVariants = localStorage.getItem('paramesh_num_variants');
  if (savedVariants) {
    const slider = document.getElementById('variant-slider');
    if (slider) { slider.value = savedVariants; document.getElementById('variant-count').textContent = savedVariants; }
  }
  const savedResize = localStorage.getItem('paramesh_resize_on');
  if (savedResize === '1') {
    const toggle = document.getElementById('resize-toggle');
    if (toggle) toggle.checked = true;
  }
} catch (e) {}

// ── AI Chat ──
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');

function addChatMessage(role, text, isThinking = false) {
  const msg = document.createElement('div');
  msg.className = `chat-msg ${role}${isThinking ? ' thinking' : ''}`;
  msg.textContent = text;
  chatMessages.appendChild(msg);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

async function sendChatMessage() {
  const text = chatInput.value.trim();
  if (!text) return;
  console.debug('[sendChat] sending message, length:', text.length);

  addChatMessage('user', text);
  chatInput.value = '';
  chatInput.disabled = true;
  chatSend.disabled = true;

  // Show typing indicator
  const typing = document.createElement('div');
  typing.className = 'chat-typing';
  typing.textContent = 'Thinking...';
  chatMessages.appendChild(typing);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  try {
    const statusRes = await fetch('/api/status');
    if (!statusRes.ok) console.warn('[sendChat] /api/status HTTP', statusRes.status);
    let context;
    try { context = await statusRes.json(); } catch (e) { console.error('[sendChat] failed to parse /api/status response', e); context = {}; }

    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, context }),
    });
    if (!res.ok) { console.error('[sendChat] /api/chat HTTP', res.status); throw new Error('Chat request failed: HTTP ' + res.status); }
    let data;
    try { data = await res.json(); } catch (e) { console.error('[sendChat] JSON parse failed from /api/chat', e); throw e; }

    typing.remove();
    console.debug('[sendChat] reply received, length:', data.reply?.length);
    addChatMessage('ai', data.reply);

    // Update token count
    if (data.usage) {
      document.getElementById('token-count').textContent = data.usage.calls_used || 0;
    }

    // Handle feature edits
    if (data.edit) {
      addChatMessage('system', `Applying edit: ${JSON.stringify(data.edit)}`);
      // TODO: re-run pipeline with modified params and refresh viewer
    }
  } catch (err) {
    console.error('[sendChat] error:', err);
    typing.remove();
    addChatMessage('ai', `Error: ${err.message}`);
  } finally {
    chatInput.disabled = false;
    chatSend.disabled = false;
    chatInput.focus();
  }
}

chatSend.addEventListener('click', sendChatMessage);

// ── Settings modal ──
window.openSettings = async function() {
  console.debug('[openSettings] opening settings modal');
  const modal = document.getElementById('settings-modal');
  modal.classList.add('show');

  try {
    const res = await fetch('/api/usage');
    if (!res.ok) { console.error('[openSettings] /api/usage HTTP', res.status); return; }
    const u = await res.json();
    console.debug('[openSettings] usage loaded, plan:', u.plan, 'daily:', u.daily);

    const limit = u.parts_limit || u.tokens_limit || 30;
    const partsUsed = u.parts_used || u.monthly || 0;

    // Plan badge
    document.getElementById('settings-plan-badge').textContent = u.plan.charAt(0).toUpperCase() + u.plan.slice(1);
    document.getElementById('settings-plan-name').textContent = u.plan.charAt(0).toUpperCase() + u.plan.slice(1);
    document.getElementById('settings-upload').textContent = u.upload_limit_mb + ' MB';

    // Parts this month
    fillBar('bar-month', 'num-month', 'lim-month', partsUsed, limit);
  } catch (e) { console.error('[openSettings] failed to load usage data', e); }
};

function fillBar(barId, numId, limId, spent, limit) {
  const pct = Math.min(100, (spent / limit) * 100);
  const bar = document.getElementById(barId);
  bar.style.width = pct + '%';
  bar.className = 'usage-bar-fill ' + (pct < 60 ? 'green' : pct < 85 ? 'yellow' : 'red');
  document.getElementById(numId).textContent = spent;
  document.getElementById(limId).textContent = '/ ' + limit;
}

// Close settings on backdrop click
document.getElementById('settings-modal').addEventListener('click', function(e) {
  if (e.target === this) this.classList.remove('show');
});
chatInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendChatMessage();
  }
});

// ── Drag-to-resize panels ──
function setupResize(handleId, side) {
  const handle = document.getElementById(handleId);
  if (!handle) return;
  let dragging = false;

  handle.addEventListener('mousedown', (e) => {
    dragging = true;
    handle.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
    e.preventDefault();
  });

  document.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const sidebar = document.getElementById('sidebar');
    const chat = document.getElementById('chat-panel');
    const viewer = document.getElementById('viewer');
    const resizeLeft = document.getElementById('resize-left');
    const resizeRight = document.getElementById('resize-right');

    if (side === 'left') {
      const w = Math.max(48, Math.min(400, e.clientX));
      sidebar.style.width = w + 'px';
      viewer.style.left = w + 'px';
      resizeLeft.style.left = w + 'px';
    } else {
      const w = Math.max(0, Math.min(500, window.innerWidth - e.clientX));
      chat.style.width = w + 'px';
      viewer.style.right = w + 'px';
      resizeRight.style.right = w + 'px';
    }
    handleResize();
  });

  document.addEventListener('mouseup', () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove('dragging');
    document.body.style.cursor = '';
    document.body.style.userSelect = '';
  });
}
setupResize('resize-left', 'left');
setupResize('resize-right', 'right');

// ── Resize ──
function handleResize() {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
}
window.addEventListener('resize', handleResize);

// Save/restore last viewer state across refreshes
function saveViewerState(state) {
  // state: 'preview' or 'cad'
  localStorage.setItem('paramesh_viewer_state', state);
}

// Restore parts count from localStorage, then refresh from server
try {
  const savedParts = localStorage.getItem('paramesh_parts_count');
  if (savedParts) document.getElementById('token-count').textContent = savedParts;
} catch (e) {}
fetch('/api/usage').then(r => r.ok ? r.json() : null).then(u => {
  if (!u) return;
  const txt = u.parts_limit > 0 ? `${u.parts_used} / ${u.parts_limit}` : `${u.parts_used}`;
  document.getElementById('token-count').textContent = txt;
  localStorage.setItem('paramesh_parts_count', txt);
}).catch(() => {});

async function autoLoad() {
  console.debug('[autoLoad] checking for saved state...');
  // Check sessionStorage for a saved mesh preview
  const saved = sessionStorage.getItem('paramesh_preview');
  if (saved) {
    try {
      const previewRes = JSON.parse(saved);
      if (previewRes.mesh_vertices && previewRes.mesh_vertices.length > 0 && previewRes.temp_path) {
        // Verify the temp file still exists on the server
        try {
          const check = await fetch('/api/check-file', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ temp_path: previewRes.temp_path }),
          });
          const checkData = await check.json();
          if (!checkData.exists) {
            console.warn('[autoLoad] stale preview — temp file gone, clearing');
            sessionStorage.removeItem('paramesh_preview');
            // fall through to load last CAD job
          } else {
            console.debug('[autoLoad] restoring preview from sessionStorage, verts:', previewRes.mesh_vertices.length);
            document.getElementById('empty-state').style.display = 'none';
            _showRawMesh(previewRes);
            pendingTempPath = previewRes.temp_path;
            pendingFile = null;
            document.getElementById('btn-analyze').style.display = '';
            const fnL = document.getElementById('viewer-filename');
            if (fnL) fnL.textContent = previewRes.filename || '';
            return;
          }
        } catch (e) {
          console.warn('[autoLoad] check-file failed, clearing stale preview');
          sessionStorage.removeItem('paramesh_preview');
        }
      }
    } catch (e) {
      console.error('[autoLoad] preview restore failed:', e);
    }
  }
  // Check for a demo job from the landing page (user just signed up after demo)
  const demoJobCookie = document.cookie.split(';').map(c => c.trim()).find(c => c.startsWith('paramesh_demo_job='));
  if (demoJobCookie) {
    const demoJobId = demoJobCookie.split('=')[1];
    console.debug('[autoLoad] found demo job cookie:', demoJobId);
    // Clear the cookie so it doesn't interfere next time
    document.cookie = 'paramesh_demo_job=; path=/; max-age=0';
    try {
      const demoRes = await fetch('/api/job/' + demoJobId + '/status');
      if (demoRes.ok) {
        const demoData = await demoRes.json();
        if (demoData.status === 'done' && demoData.steps && demoData.steps.length > 0) {
          console.debug('[autoLoad] loading demo job:', demoJobId);
          currentJobId = demoJobId;
          document.getElementById('empty-state').style.display = 'none';
          document.getElementById('sidebar').classList.remove('closed');
          finishLoad(demoData);
          return;
        }
      }
    } catch (e) {
      console.debug('[autoLoad] demo job load failed:', e.message);
    }
  }

  // No preview — load last completed CAD job
  console.debug('[autoLoad] no preview found, checking /api/status for last CAD job');
  try {
    const res = await fetch('/api/status');
    if (!res.ok) { console.debug('[autoLoad] /api/status HTTP', res.status, '- no previous job'); return; }
    let data;
    try { data = await res.json(); } catch (e) { console.error('[autoLoad] JSON parse failed from /api/status', e); return; }
    if (data.status === 'done' && data.steps && data.steps.length > 0) {
      currentJobId = data.job_id || null;
      console.debug('[autoLoad] restoring CAD job:', currentJobId, 'steps:', data.steps.length);
      document.getElementById('empty-state').style.display = 'none';
      // Restore mesh data for Mesh/Compare buttons
      const savedMesh = sessionStorage.getItem('paramesh_mesh_compare');
      if (savedMesh) {
        try { savedMeshPreview = JSON.parse(savedMesh); console.debug('[autoLoad] restored mesh compare data'); } catch (e) { /* ignore */ }
      }
      // Load the last viewed step (saved in localStorage)
      const lastStep = parseInt(localStorage.getItem('paramesh_last_step') || '-1');
      if (lastStep >= 0 && lastStep < data.steps.length) {
        console.debug('[autoLoad] restoring last step:', lastStep);
        document.getElementById('sidebar').classList.remove('closed');
        populateUI(data);
        cachedStatusData = data;
        document.getElementById('status-dot').className = 'status-dot done';
        document.getElementById('btn-export').disabled = false;
        document.getElementById('play-btn').style.display = '';
        document.getElementById('chat-input').disabled = false;
        document.getElementById('chat-send').disabled = false;
        // Show dimension panel from bbox on restore
        if (data.bbox_min && data.bbox_max) {
          populateDimPanel(buildDimsFromStatus(data));
        }
        loadStep(lastStep).then(() => {
          // Fit camera to loaded model using smooth animation
          try {
            const box = new THREE.Box3().setFromObject(modelGroup);
            const s = box.getSize(new THREE.Vector3());
            const size = s.length();
            if (size > 0) {
              const camY = Math.max(size * 1.2, s.y * 2.5);
              console.debug('[autoLoad] camera fit, bbox size:', size.toFixed(1));
              animateCamera(
                { x: size * 0.7, y: camY, z: size * 0.7 },
                { x: 0, y: 0, z: 0 },
                1.2,
              );
            }
          } catch (e) {
            console.error('[autoLoad] camera animation failed', e);
          }
        });
      } else {
        console.debug('[autoLoad] no valid last step, calling finishLoad');
        finishLoad(data);
      }
    } else {
      console.debug('[autoLoad] no completed job found, status:', data.status);
    }
  } catch (e) { console.debug('[autoLoad] no previous job available:', e.message); }
}
// ── Camera animation system (must be before autoLoad) ──
let cameraAnim = null;

function animateCamera(endPos, endTarget, duration = 1.5) {
  console.debug('[animateCamera] to:', endPos, 'target:', endTarget, 'duration:', duration + 's');
  cameraAnim = {
    startPos: camera.position.clone(),
    endPos: new THREE.Vector3(endPos.x, endPos.y, endPos.z),
    startTarget: controls.target.clone(),
    endTarget: new THREE.Vector3(endTarget.x, endTarget.y, endTarget.z),
    startTime: performance.now(),
    duration: duration * 1000,
  };
}

function easeOutCubic(t) { return 1 - Math.pow(1 - t, 3); }

// Compute camera position facing the largest profile of the model
function bestCameraPos(box) {
  const s = box.getSize(new THREE.Vector3());
  const size = s.length();
  // Find the two largest dimensions — the camera should face the smallest dimension (thickness)
  const dims = [{axis: 'x', val: s.x}, {axis: 'y', val: s.y}, {axis: 'z', val: s.z}];
  dims.sort((a, b) => a.val - b.val);
  const thinAxis = dims[0].axis; // smallest = thickness direction
  const dist = size * 1.8;

  // Position camera along the thin axis, offset slightly for perspective
  const pos = { x: dist * 0.2, y: dist * 0.35, z: dist * 0.2 };
  if (thinAxis === 'x') pos.x = dist;
  else if (thinAxis === 'y') pos.y = dist;
  else pos.z = dist;
  return pos;
}

// Step-through toggle — live update during pipeline
// Restore step-through preference from localStorage (default: on)
(function restoreStepThrough() {
  const cb = document.getElementById('step-through-cb');
  if (!cb) return;
  const saved = localStorage.getItem('paramesh_step_through');
  if (saved !== null) cb.checked = saved === '1';
})();

document.getElementById('step-through-cb')?.addEventListener('change', (e) => {
  localStorage.setItem('paramesh_step_through', e.target.checked ? '1' : '0');
  if (currentJobId) {
    fetch(`/api/job/${currentJobId}/step-through/${e.target.checked ? '1' : '0'}`, { method: 'POST' });
  }
  updatePauseAtVisibility();
});

// Debug toggle — show/hide the per-stage pipeline summary panel.
// Persists across reloads via localStorage. The panel itself is
// populated from job.debug_summary inside the status poll loop.
(function restoreDebug() {
  const cb = document.getElementById('debug-cb');
  if (!cb) return;
  const saved = localStorage.getItem('paramesh_debug') === '1';
  cb.checked = saved;
  const panel = document.getElementById('debug-panel');
  if (panel) panel.style.display = saved ? '' : 'none';
})();
document.getElementById('debug-cb')?.addEventListener('change', (e) => {
  localStorage.setItem('paramesh_debug', e.target.checked ? '1' : '0');
  const panel = document.getElementById('debug-panel');
  if (panel) panel.style.display = e.target.checked ? '' : 'none';
  // Turning Debug off deactivates the breakpoint: clear it on the
  // active job (server also unpauses if currently stopped at it).
  if (!e.target.checked && currentJobId) {
    fetch(`/api/job/${currentJobId}/pause-at/none`, { method: 'POST' });
  }
  updatePauseAtVisibility();
});

// "Pause at stage" breakpoint dropdown. Only shown when Debug is on and
// Step is off — it's a single-stop breakpoint for the otherwise-non-step
// pipeline. Populated from /api/pause-stages, selection persisted, and
// sent to the backend on analyze + live when changed mid-run.
function updatePauseAtVisibility() {
  const wrap = document.getElementById('pause-at-wrap');
  const debugOn = document.getElementById('debug-cb')?.checked;
  const stepOn = document.getElementById('step-through-cb')?.checked;
  if (wrap) wrap.style.display = (debugOn && !stepOn) ? 'flex' : 'none';
}
(async function initPauseAt() {
  const sel = document.getElementById('pause-at-select');
  if (!sel) return;
  try {
    const r = await fetch('/api/pause-stages');
    if (r.ok) {
      const data = await r.json();
      for (const s of (data.stages || [])) {
        const opt = document.createElement('option');
        opt.value = s.key;
        opt.textContent = s.name;
        sel.appendChild(opt);
      }
    }
  } catch (e) { /* dropdown stays at "none" only */ }
  const saved = localStorage.getItem('paramesh_pause_at');
  if (saved) sel.value = saved;
  sel.addEventListener('change', (e) => {
    localStorage.setItem('paramesh_pause_at', e.target.value);
    if (currentJobId) {
      fetch(`/api/job/${currentJobId}/pause-at/${e.target.value}`, { method: 'POST' });
    }
  });
  updatePauseAtVisibility();
})();

autoLoad();
// Independent of the viewer/preview restore above: bring back the last opened
// folder panel (or its one-click reopen chip) if there was one.
restoreFolder();

function animate() {
  requestAnimationFrame(animate);

  if (cameraAnim) {
    const elapsed = performance.now() - cameraAnim.startTime;
    const t = Math.min(1, elapsed / cameraAnim.duration);
    const e = easeOutCubic(t);
    camera.position.lerpVectors(cameraAnim.startPos, cameraAnim.endPos, e);
    controls.target.lerpVectors(cameraAnim.startTarget, cameraAnim.endTarget, e);
    if (t >= 1) cameraAnim = null;
  }

  // Smooth zoom interpolation
  if (_zoomTargetDist !== null) {
    const dir = camera.position.clone().sub(controls.target);
    const curDist = dir.length();
    const newDist = curDist + ((_zoomTargetDist - curDist) * ZOOM_LERP);
    if (Math.abs(newDist - _zoomTargetDist) < 0.5) {
      dir.normalize().multiplyScalar(_zoomTargetDist);
      _zoomTargetDist = null;
    } else {
      dir.normalize().multiplyScalar(newDist);
    }
    camera.position.copy(controls.target).add(dir);
  }

  // Idle rotation during pipeline — orbit camera around Z axis (vertical spin).
  // Stateless: all orbit parameters (distance, theta, Z) are recomputed
  // from the actual camera position each frame, so:
  //   - wheel-zoom is respected (distance updates flow through immediately)
  //   - drag-release is respected (theta and Z match wherever the user
  //     orbited to; no snap-back to a cached orientation)
  //   - we never write an orbit position that contradicts what the user
  //     just did with the mouse
  if (window._pipelineRotate && !window._userDragging) {
    const curDist = camera.position.distanceTo(controls.target);
    const curZ = camera.position.z;
    const curTheta = Math.atan2(
      camera.position.y - controls.target.y,
      camera.position.x - controls.target.x,
    );
    const newTheta = curTheta + 0.00075;
    const zOff = curZ - controls.target.z;
    const r = Math.sqrt(Math.max(0.1, curDist * curDist - zOff * zOff));
    camera.position.set(
      controls.target.x + r * Math.cos(newTheta),
      controls.target.y + r * Math.sin(newTheta),
      curZ,
    );
    camera.lookAt(controls.target);
  }

  controls.update();
  renderer.render(scene, camera);

  // Expose camera distance for debugging
  window._camDist = camera.position.distanceTo(controls.target);
  window._camPos = { x: camera.position.x.toFixed(1), y: camera.position.y.toFixed(1), z: camera.position.z.toFixed(1) };

  // Sync axis gizmo rotation with main camera
  axisCamera.position.copy(camera.position).sub(controls.target).normalize().multiplyScalar(5);
  axisCamera.lookAt(0, 0, 0);
  axisRenderer.render(axisScene, axisCamera);
}
animate();


// ──────────────────────────────────────────────────────────────────
// Onshape integration (per-user OAuth export). Backend:
// api/onshape_oauth.py. Exposed on window for the inline menu onclicks.
// ──────────────────────────────────────────────────────────────────
let _onshapeStatus = null;

async function refreshOnshapeStatus() {
  try {
    const r = await fetch('/api/onshape/status');
    _onshapeStatus = await r.json();
  } catch (e) { _onshapeStatus = null; }
  const item = document.getElementById('onshape-menu-item');
  if (item) {
    if (_onshapeStatus && _onshapeStatus.connected) {
      const who = _onshapeStatus.onshape_user_name || 'Connected';
      item.innerHTML = '<i class="ri-link"></i> Onshape: ' + who + ' · Disconnect';
    } else {
      item.innerHTML = '<i class="ri-link-unlink"></i> Connect Onshape';
    }
  }
  // Reveal the "Open in Onshape" button once we know the account is connected
  // (so an empty document can be opened before analyzing a mesh).
  try { updateOnshapeOpenButton(); } catch (e) {}
}
window.refreshOnshapeStatus = refreshOnshapeStatus;

window.onshapeConnectClick = async function () {
  // re-read live status so the message matches reality
  let st = _onshapeStatus;
  try { st = await (await fetch('/api/onshape/status')).json(); _onshapeStatus = st; } catch (e) {}
  console.log('[onshape] Connect clicked; status =', st);
  if (st && st.connected) {
    if (confirm('Disconnect your Onshape account from Paramesh?')) {
      fetch('/api/onshape/disconnect', { method: 'POST' })
        .then(() => { addChatMessage('ai', 'Onshape account disconnected.'); refreshOnshapeStatus(); })
        .catch(() => addChatMessage('ai', 'Could not disconnect Onshape.'));
    }
    return;
  }
  if (st && !st.configured) {
    addChatMessage('ai', 'Onshape is not configured on the server (missing OAuth app credentials).');
    return;
  }
  if (st && !st.eligible) {
    // Either not logged in, or not a Pro/developer account.
    addChatMessage('ai', 'To connect Onshape you must be logged in with a Pro account. Taking you to sign in…');
    window.location = '/api/onshape/connect?next=' + encodeURIComponent(location.pathname);
    return;
  }
  addChatMessage('ai', 'Opening Onshape to connect your account…');
  window.location = '/api/onshape/connect?next=' + encodeURIComponent(location.pathname);
};

function _onshapeProgressEl() {
  const d = document.createElement('div');
  d.className = 'chat-msg ai';
  chatMessages.appendChild(d);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return d;
}

// ── Onshape document binding + the unified "Open in Onshape" button ──
// ONE button drives everything. First press for a mesh creates a fresh
// Onshape document, builds the part into it, and opens the tab; after that
// it reads "Reopen in Onshape" and just re-opens that same document's tab.
// Uploading a different mesh resets it to "Open" and the next press makes a
// NEW document for that mesh (so parts never overlap). Pressing Play (or a
// Reanalyze) syncs into the already-open document WITHOUT opening a tab.
//
// _onshapeTarget = {document_id, workspace_id, element_id, url, filename,
// opened}. `filename` is which mesh the doc holds; `opened` is whether the
// user has opened its tab.
let _onshapeTarget = null;
try { _onshapeTarget = JSON.parse(localStorage.getItem('onshape_target') || 'null'); } catch (e) {}

// Filename the CURRENT job was analyzed for. After uploading a new mesh the
// old job is stale — building it would push the wrong (old) part — so the
// Open button opens an empty doc until the new mesh is analyzed.
let _analyzedMeshName = '';
function _jobMatchesCurrentMesh() {
  return !!currentJobId && !!_analyzedMeshName
    && _analyzedMeshName === currentMeshName();
}

function _saveOnshapeTarget() {
  try {
    if (_onshapeTarget) localStorage.setItem('onshape_target', JSON.stringify(_onshapeTarget));
    else localStorage.removeItem('onshape_target');
  } catch (e) {}
}

function currentMeshName() {
  const el = document.getElementById('viewer-filename');
  return (el && el.textContent)
    || (typeof pendingFile !== 'undefined' && pendingFile && pendingFile.name) || '';
}

// The bound document "belongs to" the current mesh when its filename matches
// — OR when it's an UNCLAIMED empty document (filename not set yet, e.g. one
// opened before analyzing). An unclaimed doc accepts the next mesh built, so
// "Open in Onshape" before analysis works: the pipeline builds into it.
function _boundForCurrentMesh() {
  if (!(_onshapeTarget && _onshapeTarget.document_id)) return false;
  if (!_onshapeTarget.filename) return true;          // unclaimed empty doc
  const cur = currentMeshName();
  return !!cur && _onshapeTarget.filename === cur;
}

// "Open in Onshape" until the current document has been opened, then
// "Reopen in Onshape". Shown ONLY when the Onshape account is connected
// (otherwise the button is hidden — connect via the menu first).
function updateOnshapeOpenButton() {
  const btn = document.getElementById('onshape-open-btn');
  if (!btn) return;
  const analyzed = !!currentJobId && !!(typeof steps !== 'undefined' && steps && steps.length);
  const connected = !!(_onshapeStatus && _onshapeStatus.connected);
  btn.style.display = connected ? 'flex' : 'none';
  const opened = _boundForCurrentMesh() && _onshapeTarget.opened;
  const lbl = btn.querySelector('.os-label');
  if (lbl) lbl.textContent = opened ? 'Reopen in Onshape' : 'Open in Onshape';
  btn.title = opened
    ? 'Reopen this document in your Onshape tab'
    : (analyzed ? 'Build this part into a new Onshape document and open it'
                : 'Open an empty Onshape document — the pipeline will build into it');
}
window.updateOnshapeOpenButton = updateOnshapeOpenButton;

// Kept for the menu's "Stop building" item and other callers.
function updateOnshapeBindUI() {
  const unbind = document.getElementById('onshape-unbind-item');
  if (unbind) unbind.style.display = _onshapeTarget ? '' : 'none';
  updateOnshapeOpenButton();
}

window.unbindOnshapeDocument = function () {
  _onshapeTarget = null; _saveOnshapeTarget();
  updateOnshapeBindUI();
  addChatMessage('ai', 'Stopped building to Onshape. The document stays in your account.');
};

// THE button. Popup-safe — the tab is opened synchronously inside this click.
window.openInOnshape = async function () {
  // Connection gate. We must NOT await before opening the tab (a popup only
  // survives a synchronous user gesture), so only take the async refresh
  // path when we don't already know we're connected. If a stale status was
  // hiding a live connection, ask for one more click rather than a popup.
  if (!(_onshapeStatus && _onshapeStatus.connected)) {
    try { await refreshOnshapeStatus(); } catch (e) {}
    if (_onshapeStatus && _onshapeStatus.connected) {
      addChatMessage('ai', 'Onshape connected — click "Open in Onshape" again.');
      updateOnshapeOpenButton();
      return;
    }
    addChatMessage('ai', 'Connect your Onshape account to open it in Onshape…');
    window.onshapeConnectClick();
    return;
  }
  const cur = currentMeshName();
  // Reopen: the current document is already open → just show its tab.
  if (_boundForCurrentMesh() && _onshapeTarget.opened && _onshapeTarget.url) {
    window.open(_onshapeTarget.url, '_blank');
    return;
  }
  // Open the tab now (popup-safe). Clear any binding for a DIFFERENT mesh so
  // we make a fresh document instead of overwriting that one.
  const win = window.open('', '_blank');
  if (!_boundForCurrentMesh()) { _onshapeTarget = null; _saveOnshapeTarget(); }

  // No analyzed mesh for what's loaded (nothing analyzed, or a new upload not
  // yet analyzed) → open an EMPTY document and bind it. Running the pipeline
  // then builds into this document live (the synced animation).
  if (!_jobMatchesCurrentMesh()) { await _openEmptyOnshapeDoc(win); return; }

  // Analyzed mesh → build it into a (new or unclaimed) document and open it.
  const url = await _runOnshapeBuild(win);
  if (url && _onshapeTarget) {
    _onshapeTarget.filename = cur;
    _onshapeTarget.opened = true;
    _saveOnshapeTarget();
  } else if (win) { try { win.close(); } catch (e) {} }
  updateOnshapeOpenButton();
};

// Create + open an EMPTY Onshape document and bind it (filename unset, so the
// first mesh built claims it). `win` is the pre-opened tab from the click.
async function _openEmptyOnshapeDoc(win) {
  let r;
  try { r = await fetch('/api/onshape/document/new', { method: 'POST' }); }
  catch (e) { if (win) win.close(); addChatMessage('ai', 'Could not reach the server.'); return false; }
  if (r.status === 409) { if (win) win.close(); addChatMessage('ai', 'Connect your Onshape account first.'); window.location = '/api/onshape/connect?next=' + encodeURIComponent(location.pathname); return false; }
  if (r.status === 403) { if (win) win.close(); addChatMessage('ai', 'Onshape is a Pro feature.'); return false; }
  if (r.status === 401) { if (win) win.close(); window.location = '/login?next=' + encodeURIComponent(location.pathname); return false; }
  let d = {}; try { d = await r.json(); } catch (e) {}
  if (!r.ok || !d.url) { if (win) win.close(); addChatMessage('ai', 'Could not create the document: ' + (d.detail || d.error || r.status)); return false; }
  _onshapeTarget = {
    document_id: d.document_id, workspace_id: d.workspace_id,
    element_id: d.element_id, url: d.url, filename: null, opened: true,
  };
  _saveOnshapeTarget();
  if (win) win.location = d.url;
  updateOnshapeBindUI();
  addChatMessage('ai', 'Empty Onshape document opened. Run the pipeline and it will build here live — keep that tab visible to watch.');
  return true;
}

// Silent sync used by Play and Reanalyze: rebuild the current mesh into the
// already-open bound document. NO new tab — collaborative sync updates the
// tab the user already opened via "Open in Onshape".
window.exportToOnshape = function () { return _runOnshapeBuild(null); };

// Core streamed build. `win` is a pre-opened tab to navigate to the NEW
// document (from openInOnshape), or null for a silent sync. Drives the
// build animation (advances loadStep as each Onshape feature lands) and
// returns the document URL (or null on failure).
async function _runOnshapeBuild(win) {
  console.log('[onshape] build; jobId =', currentJobId, 'target =', _onshapeTarget && _onshapeTarget.document_id, 'tab =', !!win);
  if (!currentJobId) { if (win) { try { win.close(); } catch (e) {} } return null; }
  // Sync into the bound doc only when it belongs to the current mesh;
  // otherwise build a fresh document (no target query string).
  const target = _boundForCurrentMesh() ? _onshapeTarget : null;
  let winNavigated = false;
  let qs = '';
  if (target) qs = '?document_id=' + encodeURIComponent(target.document_id) +
    '&workspace_id=' + encodeURIComponent(target.workspace_id) +
    '&element_id=' + encodeURIComponent(target.element_id);
  addChatMessage('ai', target ? 'Rebuilding in your Onshape document…' : 'Building in your Onshape account…');
  let resp;
  try {
    resp = await fetch('/api/onshape/export/' + currentJobId + '/live' + qs, { method: 'POST' });
  } catch (e) { if (win) win.close(); addChatMessage('ai', 'Onshape build failed: network error.'); return null; }
  console.log('[onshape] /live responded', resp.status);
  if (resp.status === 401) { if (win) win.close(); addChatMessage('ai', 'Log in to build in Onshape.'); window.location = '/login?next=' + encodeURIComponent(location.pathname); return null; }
  if (resp.status === 409) { if (win) win.close(); addChatMessage('ai', 'Connecting your Onshape account…'); window.location = '/api/onshape/connect?next=' + encodeURIComponent(location.pathname); return null; }
  if (resp.status === 403) { if (win) win.close(); addChatMessage('ai', 'Building in Onshape needs a Pro account.'); return null; }
  if (!resp.ok || !resp.body) {
    let d = {}; try { d = await resp.json(); } catch (e) {}
    if (win) win.close();
    addChatMessage('ai', 'Onshape build failed: ' + (d.detail || d.error || ('HTTP ' + resp.status)));
    return null;
  }

  const reader = resp.body.getReader();
  const dec = new TextDecoder();
  let buf = '', docUrl = null, total = 0, progEl = null, lastStep = -1;
  while (true) {
    let chunk;
    try { chunk = await reader.read(); } catch (e) { break; }
    if (chunk.done) break;
    buf += dec.decode(chunk.value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n\n')) >= 0) {
      const block = buf.slice(0, nl); buf = buf.slice(nl + 2);
      const dline = block.split('\n').find(l => l.startsWith('data:'));
      if (!dline) continue;
      let ev; try { ev = JSON.parse(dline.slice(5).trim()); } catch (e) { continue; }
      if (ev.type === 'document') {
        docUrl = ev.url; total = ev.total || 0;
        if (win && !winNavigated) { win.location = ev.url; winNavigated = true; }
        // Capture the document ids so Play / Reanalyze re-sync into it.
        if (ev.document_id && ev.workspace_id && ev.element_id) {
          _onshapeTarget = Object.assign({}, _onshapeTarget, {
            document_id: ev.document_id, workspace_id: ev.workspace_id,
            element_id: ev.element_id, url: ev.url,
          });
          _saveOnshapeTarget();
          updateOnshapeBindUI();
        }
      } else if (ev.type === 'feature') {
        if (steps && steps.length && total) {
          const si = Math.min(steps.length - 1, Math.max(0,
            Math.round((ev.index + 1) / total * steps.length) - 1));
          if (si !== lastStep) { lastStep = si; try { loadStep(si); } catch (e) {} }
        }
        const txt = 'Building ' + ev.n_ok + '/' + ev.total + ': ' + ev.name + (ev.ok ? '' : ' (skipped)');
        if (!progEl) progEl = _onshapeProgressEl();
        progEl.textContent = txt;
      } else if (ev.type === 'done') {
        // Tag the doc with the mesh it now holds, so uploading a DIFFERENT
        // mesh resets the button to "Open" and the next open makes a new doc.
        if (_onshapeTarget && !ev.already) {
          _onshapeTarget.filename = currentMeshName() || _onshapeTarget.filename || '';
          _saveOnshapeTarget();
        }
        addChatMessage('ai', ev.already
          ? 'Opened your Onshape document.'
          : 'Done — built ' + ev.n_ok + '/' + ev.total + ' features in Onshape.');
        refreshOnshapeStatus();
        updateOnshapeOpenButton();
      } else if (ev.type === 'error') {
        if (win && !docUrl) win.close();
        addChatMessage('ai', 'Onshape build error: ' + (ev.detail || 'unknown'));
      }
    }
  }
  return docUrl;
}

// Pipeline-complete / Reanalyze: silently re-sync into the opened doc.
// No tab — the document tab is already open from "Open in Onshape".
window.maybeAutoBuildOnshape = function () {
  if (!_boundForCurrentMesh() || !_onshapeTarget.opened) return;
  if (!(_onshapeStatus && _onshapeStatus.connected)) return;
  setTimeout(() => { try { _runOnshapeBuild(null); } catch (e) {} }, 600);
};

// Handle the ?onshape=... flag the OAuth round-trip lands back on, and
// prime the connection status for the menu label.
(function () {
  const flag = new URLSearchParams(location.search).get('onshape');
  const msgs = {
    connected: 'Onshape account connected. Use the "Open in Onshape" button to build your part.',
    denied: 'Onshape connection was cancelled.',
    upgrade: 'Onshape is a Pro feature — upgrade to connect.',
    unconfigured: 'Onshape integration is not configured on the server.',
    error: 'Onshape connection failed. Please try again.',
    badstate: 'Onshape connection expired. Please try again.',
  };
  if (flag && msgs[flag]) {
    setTimeout(() => { try { addChatMessage('ai', msgs[flag]); } catch (e) {} }, 1000);
    history.replaceState({}, '', location.pathname);
  }
  refreshOnshapeStatus();
  try { updateOnshapeBindUI(); } catch (e) {}
})();
