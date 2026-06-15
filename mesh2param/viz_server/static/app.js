import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

console.debug('[init] app.js loading, Three.js revision:', THREE.REVISION);

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

// ── Load step geometry ──
async function loadStep(index) {
  console.debug('[loadStep]', index, 'of', steps.length, 'jobId:', currentJobId);
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

  // Clear previous model — dispose geometry/materials to free GPU memory
  const disposed = modelGroup.children.length;
  while (modelGroup.children.length) {
    const child = modelGroup.children[0];
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else child.material.dispose();
    }
    modelGroup.remove(child);
  }
  if (disposed > 0) console.debug('[loadStep] disposed', disposed, 'previous objects');

  try {
    const verts = new Float32Array(data.vertices);
    const indices = data.faces;

    if (data.is_sketch && !data.sketch_overlay) {
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
      modelGroup.add(mesh);

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
}

function populateUI(statusData) {
  console.debug('[populateUI] building tree and timeline for', statusData.steps?.length, 'steps, filename:', statusData.filename);
  const list = document.getElementById('feature-list');
  const timeline = document.getElementById('timeline');
  list.innerHTML = '';
  timeline.innerHTML = '';
  steps = statusData.steps;

  // PNG icon mapping — icons from /static/icons/
  const I = (file, cls) => ({ cls, svg: `<img src="/static/icons/${file}" style="width:22px;height:22px;border-radius:50%;">` });
  const iconMap = {
    'Sketch':   I('sketch.png', 'sketch'),
    'Extrude':  I('extrude.png', 'extrude'),
    'Through':  I('hole.png', 'hole'),
    'Blind':    I('hole.png', 'hole'),
    'Chamfer':  I('chamfer.png', 'chamfer'),
    'Side Profile': I('sketch.png', 'sketch'),
    'Side Cut': I('chamfer.png', 'chamfer'),
    'Fillet':   I('fillet.png', 'fillet'),
    'Pocket':   I('shell.png', 'pocket'),
    'Boss':     I('extrude.png', 'boss'),
    'Slot':     I('shell.png', 'slot'),
    'Loft':     I('loft.png', 'extrude'),
    'Revolve':  I('revolve.png', 'extrude'),
    'Sweep':    I('sweep.png', 'extrude'),
    'Shell':    I('shell.png', 'pocket'),
    'Original': I('bodies.png', 'sketch'),
  };

  // Hide sketches folder — sketches go inline under Body
  const sketchesFolder = document.getElementById('sketches-folder');
  sketchesFolder.style.display = 'none';

  // Add "Mesh" button at the start of timeline
  const meshBtn = document.createElement('div');
  meshBtn.className = 'timeline-step timeline-special timeline-dashed';
  meshBtn.style.background = 'var(--bg-tertiary)';
  meshBtn.innerHTML = '<span style="font-size:10px;color:var(--text-secondary);">Mesh</span>';
  meshBtn.title = 'Show original mesh';
  meshBtn.onclick = () => {
    if (savedMeshPreview) {
      _showRawMesh(savedMeshPreview, false);
      document.querySelectorAll('.timeline-step').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.feature-item').forEach(el => el.classList.remove('active'));
      meshBtn.classList.add('active');
    }
  };
  timeline.appendChild(meshBtn);

  // Build feature tree items (left sidebar) — these navigate individual steps
  statusData.steps.forEach((step, i) => {
    let iconInfo = I('component.png', 'extrude');
    for (const [key, val] of Object.entries(iconMap)) {
      if (step.name.includes(key)) { iconInfo = val; break; }
    }
    // Strip LLM description after — or →
    const shortName = step.name.split(/\s*[—→]\s*/)[0].trim();
    const item = document.createElement('div');
    item.className = 'feature-item';
    item.innerHTML = `
      <div class="icon ${iconInfo.cls}">${iconInfo.svg}</div>
      <div class="texts">
        <div class="name">${shortName}</div>
        <div class="detail">${step.volume > 0 ? step.volume.toLocaleString() + ' mm³' : '2D'}</div>
      </div>
    `;
    item.onclick = () => loadStep(i);
    list.appendChild(item);
  });

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
    if (!savedMeshPreview) return;
    // Clear and show both mesh + CAD side by side
    while (modelGroup.children.length) {
      const child = modelGroup.children[0];
      if (child.geometry) child.geometry.dispose();
      if (child.material) child.material.dispose();
      modelGroup.remove(child);
    }

    // Mesh on the left (offset -X)
    const meshVerts = new Float32Array(savedMeshPreview.mesh_vertices);
    const meshGeo = new THREE.BufferGeometry();
    meshGeo.setAttribute('position', new THREE.BufferAttribute(meshVerts, 3));
    meshGeo.setIndex(savedMeshPreview.mesh_faces);
    meshGeo.computeVertexNormals();
    const meshMat = new THREE.MeshStandardMaterial({ color: 0x8B949E, metalness: 0.1, roughness: 0.8 });
    const meshObj = new THREE.Mesh(meshGeo, meshMat);

    // CAD on the right (offset +X) — load last step
    const lastIdx = statusData.steps.length - 1;
    const cadUrl = currentJobId ? `/api/job/${currentJobId}/step/${lastIdx}` : `/api/step/${lastIdx}`;
    console.debug('[compare] fetching CAD step', lastIdx, 'from', cadUrl);
    fetch(cadUrl).then(r => {
      if (!r.ok) throw new Error('HTTP ' + r.status + ' for ' + cadUrl);
      return r.json();
    }).then(cadData => {
      if (!cadData.vertices) { console.warn('[compare] CAD step has no vertices'); return; }
      try {
        const cadVerts = new Float32Array(cadData.vertices);
        const cadGeo = new THREE.BufferGeometry();
        cadGeo.setAttribute('position', new THREE.BufferAttribute(cadVerts, 3));
        cadGeo.setIndex(cadData.faces);
        cadGeo.computeVertexNormals();
        const cadMat = new THREE.MeshPhysicalMaterial({
          color: 0xC0C8D0, metalness: 0.5, roughness: 0.28,
          clearcoat: 0.15, reflectivity: 0.6,
        });
        const cadObj = new THREE.Mesh(cadGeo, cadMat);

        // The preview mesh (raw ingest frame) and the CAD model (pipeline
        // frame, after scan-align/regularize snap the part to world axes) can
        // sit in DIFFERENT orientations — the mesh lies flat while the model
        // stands upright. Canonically orient BOTH the same way (thinnest axis
        // up, longest axis along X) so the split view compares like with like.
        const _canonicalOrient = (geo) => {
          geo.computeBoundingBox();
          let s = geo.boundingBox.getSize(new THREE.Vector3());
          const dims = [s.x, s.y, s.z];
          const thin = dims.indexOf(Math.min(...dims));
          if (thin === 0) geo.rotateZ(Math.PI / 2);        // X-thin -> Y-thin
          else if (thin === 2) geo.rotateX(Math.PI / 2);   // Z-thin -> Y-thin
          geo.computeBoundingBox();
          s = geo.boundingBox.getSize(new THREE.Vector3());
          if (s.z > s.x) geo.rotateY(Math.PI / 2);          // longest -> X
          geo.computeBoundingBox();
        };
        // Canonical orient leaves a 180-deg ambiguity (which face is up). The
        // mesh frame and the regularized CAD frame can disagree on it, so one
        // ends up flipped. Resolve by the vertex centroid: flip so the heavier
        // half is consistently on the same side for both.
        const _flipConsistent = (geo) => {
          geo.computeBoundingBox();
          const c = geo.boundingBox.getCenter(new THREE.Vector3());
          const pos = geo.attributes.position;
          let cy = 0, cx = 0;
          for (let i = 0; i < pos.count; i++) { cy += pos.getY(i); cx += pos.getX(i); }
          cy /= pos.count; cx /= pos.count;
          if (cy - c.y < 0) { geo.rotateX(Math.PI); }      // heavy side up
          geo.computeBoundingBox();
          const c2 = geo.boundingBox.getCenter(new THREE.Vector3());
          let cx2 = 0; for (let i = 0; i < pos.count; i++) cx2 += pos.getX(i); cx2 /= pos.count;
          if (cx2 - c2.x < 0) { geo.rotateY(Math.PI); }     // heavy side +X
          geo.computeBoundingBox();
        };
        _canonicalOrient(meshGeo);
        _canonicalOrient(cadGeo);
        _flipConsistent(meshGeo);
        _flipConsistent(cadGeo);

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
    // Render Mesh thumbnail
    if (savedMeshPreview) {
      const url = renderMeshThumbnail(savedMeshPreview.mesh_vertices, savedMeshPreview.mesh_faces);
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
    // Render Compare thumbnail showing both mesh + CAD side by side
    if (savedMeshPreview && statusData.steps.length > 0) {
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
          const mv = new Float32Array(savedMeshPreview.mesh_vertices);
          const mg = new THREE.BufferGeometry();
          mg.setAttribute('position', new THREE.BufferAttribute(mv, 3));
          mg.setIndex(savedMeshPreview.mesh_faces);
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

document.getElementById('file-input').addEventListener('change', async (e) => {
  const file = e.target.files[0];
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
    currentStep = -1;
    currentJobId = null;

    // Show the raw mesh in the 3D viewer
    pendingTempPath = previewRes.temp_path;
    pendingFile = file;
    saveViewerState('preview');
    savedMeshPreview = previewRes;
    console.debug('[upload] state -> preview, showing raw mesh');
    _showRawMesh(previewRes);

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
  } finally {
    e.target.value = '';
  }
});

// Show raw mesh triangles in the viewer (before pipeline)
function _showRawMesh(previewRes, animateToFit = true) {
  console.debug('[_showRawMesh] verts:', previewRes.mesh_vertices?.length, 'faces:', previewRes.mesh_faces?.length);
  // Hide feature tree during mesh preview — it shows after pipeline completes
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
    if (!res.ok) { console.error('[analyze] /api/run-pipeline HTTP', res.status); alert('Pipeline request failed: HTTP ' + res.status); loading.classList.remove('show'); return; }
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

          // Render progress geometry on the 3D model
          if (d.progress_geometry) {
            _renderProgressGeometry(d.progress_geometry);
          }

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
          if (d.status === 'done') { clearInterval(poll); resolve(d); }
          else if (d.status === 'error') { clearInterval(poll); reject(new Error(d.error)); }
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
      // Animate camera only when CAD model first loads
      try {
        const box = new THREE.Box3().setFromObject(modelGroup);
        const pos = bestCameraPos(box);
        console.debug('[finishLoad] model loaded, animating camera to face largest profile');
        animateCamera(pos, { x: 0, y: 0, z: 0 }, 1.2);
        // Auto-play build animation after camera settles
        setTimeout(() => { if (typeof playBuildAnimation === 'function') playBuildAnimation(); }, 1500);
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
