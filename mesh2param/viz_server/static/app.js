import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

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

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 10;
controls.maxDistance = 2000;

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
}
window._updateSceneBg = updateSceneBg;

// Small XYZ axis gizmo in the bottom-left corner (rendered separately)
const axisScene = new THREE.Scene();
const axisCamera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
axisCamera.position.set(3, 2, 3);
axisCamera.lookAt(0, 0, 0);
const axisRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
axisRenderer.setSize(120, 120);
axisRenderer.setClearColor(0x000000, 0);
const axIsDark = document.documentElement.getAttribute('data-theme') === 'dark';
axisRenderer.domElement.style.cssText = `position:fixed;bottom:90px;left:10px;z-index:50;pointer-events:none;border:1px solid ${axIsDark ? 'rgba(48,54,61,0.5)' : 'rgba(208,215,222,0.5)'};border-radius:8px;`;
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

// ── Load step geometry ──
async function loadStep(index) {
  const url = currentJobId ? `/api/job/${currentJobId}/step/${index}` : `/api/step/${index}`;
  let res;
  try {
    res = await fetch(url);
  } catch (e) {
    console.error('Failed to fetch step', index, e);
    return;
  }
  if (!res.ok) {
    console.error('Step fetch error', index, res.status);
    return;
  }
  const data = await res.json();
  if (data.error) {
    console.error('Step data error', index, data.error);
    return;
  }
  if (!data.vertices || data.vertices.length === 0) {
    console.warn('Step has no geometry', index);
    return;
  }

  // Clear previous model — dispose geometry/materials to free GPU memory
  while (modelGroup.children.length) {
    const child = modelGroup.children[0];
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      if (Array.isArray(child.material)) child.material.forEach(m => m.dispose());
      else child.material.dispose();
    }
    modelGroup.remove(child);
  }

  const verts = new Float32Array(data.vertices);
  const indices = data.faces;

  if (data.is_sketch) {
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
    // Solid mesh
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    const mat = new THREE.MeshPhysicalMaterial({
      color: 0xC0C8D0,
      metalness: 0.5,
      roughness: 0.28,
      clearcoat: 0.15,
      clearcoatRoughness: 0.3,
      reflectivity: 0.6,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    modelGroup.add(mesh);
  }

  // Center model at origin
  const box = new THREE.Box3().setFromObject(modelGroup);
  const center = box.getCenter(new THREE.Vector3());
  modelGroup.position.sub(center);

  const bsize = box.getSize(new THREE.Vector3());
  const size = bsize.length();
  const height = bsize.y;

  // For step 0 or 1 (sketch/extrude): show full model from above
  // For feature steps (holes/chamfers): zoom toward the feature area
  const isFeatureStep = index > 1 && !data.is_sketch;

  if (isFeatureStep && currentJobId) {
    // Find where the feature is by comparing with previous step's vertices
    // Use the centroid of vertices that exist in prev but not in current
    const prevUrl = `/api/job/${currentJobId}/step/${index - 1}`;
    fetch(prevUrl).then(r => r.ok ? r.json() : null).then(prevData => {
      if (!prevData || !prevData.vertices || prevData.is_sketch) return;

      // Find approximate center of removed geometry
      const prevV = new Float32Array(prevData.vertices);
      const curV = verts;
      // Simple heuristic: average the vertices from previous step
      // that are "far" from any current vertex — these are near the hole
      // For speed, just compute centroid difference
      let sx = 0, sy = 0, sz = 0, n = prevV.length / 3;
      for (let i = 0; i < prevV.length; i += 3) {
        sx += prevV[i]; sy += prevV[i+1]; sz += prevV[i+2];
      }
      let cx = 0, cy = 0, cz = 0, m = curV.length / 3;
      for (let i = 0; i < curV.length; i += 3) {
        cx += curV[i]; cy += curV[i+1]; cz += curV[i+2];
      }
      // The removed geometry's centroid is weighted toward the hole
      const featureX = (sx/n - cx/m) * n;
      const featureY = (sy/n - cy/m) * n;
      const featureZ = (sz/n - cz/m) * n;

      // Zoom toward the feature but keep some distance
      const fLen = Math.sqrt(featureX*featureX + featureY*featureY + featureZ*featureZ);
      if (fLen > 0.1) {
        const fx = featureX / fLen;
        const fy = featureY / fLen;
        const fz = featureZ / fLen;
        const zoomDist = size * 0.4;
        animateCamera(
          { x: fx * zoomDist + size * 0.2, y: Math.abs(fy * zoomDist) + size * 0.3, z: fz * zoomDist + size * 0.2 },
          { x: fx * size * 0.1, y: fy * size * 0.1, z: fz * size * 0.1 },
          0.8,
        );
      }
    }).catch(() => {});
  } else {
    // Full model view for base steps
    const camY = Math.max(size * 1.2, height * 2.5);
    animateCamera(
      { x: size * 0.7, y: camY, z: size * 0.7 },
      { x: 0, y: 0, z: 0 },
      1.0,
    );
  }

  // Update UI
  currentStep = index;
  localStorage.setItem('paramesh_last_step', index.toString());
  updateActiveStates();

  // Show step name toast briefly
  let toast = document.getElementById('step-toast');
  if (!toast) {
    toast = document.createElement('div');
    toast.id = 'step-toast';
    toast.style.cssText = 'position:fixed;top:60px;left:50%;transform:translateX(-50%);background:var(--bg-secondary);border:1px solid var(--border);border-radius:8px;padding:8px 16px;font-size:13px;color:var(--text-primary);z-index:100;opacity:0;transition:opacity 0.3s;pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,0.1);';
    document.body.appendChild(toast);
  }
  toast.textContent = `Step ${index + 1}: ${data.name}`;
  toast.style.opacity = '1';
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => { toast.style.opacity = '0'; }, 1500);
  document.getElementById('empty-state').style.display = 'none';
}

window._loadStep = loadStep;

// ── UI Updates ──
function updateActiveStates() {
  document.querySelectorAll('.feature-item').forEach((el, i) => {
    el.classList.toggle('active', i === currentStep);
  });
  document.querySelectorAll('.timeline-step').forEach((el, i) => {
    el.classList.toggle('active', i === currentStep);
  });
}

function populateUI(statusData) {
  const list = document.getElementById('feature-list');
  const timeline = document.getElementById('timeline');
  list.innerHTML = '';
  timeline.innerHTML = '';
  steps = statusData.steps;

  // Fusion 360-style inline SVG icons (18×18 viewBox, stroke-based)
  const S = (d, cls) => ({ cls, svg: `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 18 18" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round" style="width:14px;height:14px;">${d}</svg>` });
  const iconMap = {
    // Sketch: grid plane with pencil
    'Sketch': S('<rect x="2" y="5" width="10" height="10" rx="1"/><line x1="2" y1="10" x2="12" y2="10"/><line x1="7" y1="5" x2="7" y2="15"/><path d="M13 2l2.5 2.5-6 6H7V8l6-6z"/>', 'sketch'),
    // Extrude: rectangle with upward arrow
    'Extrude': S('<rect x="3" y="8" width="12" height="8" rx="1"/><line x1="9" y1="8" x2="9" y2="2"/><polyline points="6,5 9,2 12,5"/>', 'extrude'),
    // Through hole: cylinder top-down with arrow through
    'Through': S('<ellipse cx="9" cy="9" rx="5.5" ry="5.5"/><ellipse cx="9" cy="9" rx="2.5" ry="2.5"/><line x1="9" y1="2" x2="9" y2="16" stroke-dasharray="1.5 1.5"/>', 'hole'),
    // Blind hole: cylinder with depth stop
    'Blind': S('<ellipse cx="9" cy="5" rx="5" ry="2.5"/><line x1="4" y1="5" x2="4" y2="13"/><line x1="14" y1="5" x2="14" y2="13"/><ellipse cx="9" cy="13" rx="5" ry="2.5"/><line x1="9" y1="5" x2="9" y2="10"/><circle cx="9" cy="10" r="0.8" fill="currentColor" stroke="none"/>', 'hole'),
    // Chamfer: corner with angled bevel line
    'Chamfer': S('<polyline points="3,15 3,7 11,7"/><line x1="3" y1="11" x2="7" y2="7" stroke-width="2"/><line x1="3" y1="15" x2="3" y2="11"/><line x1="7" y1="7" x2="11" y2="7"/><line x1="1" y1="13" x2="5" y2="9" stroke-dasharray="1.2 1.2" stroke-width="1"/>', 'chamfer'),
    // Fillet: corner with radius arc
    'Fillet': S('<line x1="3" y1="15" x2="3" y2="9"/><line x1="9" y1="7" x2="15" y2="7"/><path d="M3 9 Q3 7 9 7" stroke-width="2"/>', 'fillet'),
    // Pocket: recessed rectangular cut (top-down view)
    'Pocket': S('<rect x="2" y="2" width="14" height="14" rx="1.5"/><rect x="5" y="5" width="8" height="8" rx="1"/><line x1="2" y1="2" x2="5" y2="5"/><line x1="16" y1="2" x2="13" y2="5"/><line x1="2" y1="16" x2="5" y2="13"/><line x1="16" y1="16" x2="13" y2="13"/>', 'pocket'),
    // Boss: raised block with upward offset
    'Boss': S('<rect x="2" y="10" width="14" height="6" rx="1"/><rect x="5" y="4" width="8" height="6" rx="1"/><line x1="5" y1="10" x2="2" y2="10"/><line x1="13" y1="10" x2="16" y2="10"/>', 'boss'),
    // Slot: elongated rounded cutout
    'Slot': S('<rect x="2" y="2" width="14" height="14" rx="1.5"/><rect x="4" y="6.5" width="10" height="5" rx="2.5"/>', 'slot'),
    // Original mesh file
    'Original': S('<path d="M5 2h6l4 4v10a1 1 0 01-1 1H5a1 1 0 01-1-1V3a1 1 0 011-1z"/><polyline points="11,2 11,6 15,6"/><path d="M7 10l2 2 3-4" stroke-width="1.2"/>', 'sketch'),
  };

  // Show sketches folder and clear it
  const sketchesFolder = document.getElementById('sketches-folder');
  const sketchesList = document.getElementById('sketches-list');
  sketchesList.innerHTML = '';
  let hasSketch = false;

  statusData.steps.forEach((step, i) => {
    // Determine icon
    let iconInfo = S('<rect x="3" y="3" width="12" height="12" rx="2"/>', 'extrude');
    for (const [key, val] of Object.entries(iconMap)) {
      if (step.name.includes(key)) { iconInfo = val; break; }
    }

    const isSketch = step.name.toLowerCase().includes('sketch');

    // Feature tree item
    const item = document.createElement('div');
    item.className = 'feature-item';
    item.innerHTML = `
      <div class="icon ${iconInfo.cls}">${iconInfo.svg}</div>
      <div class="texts">
        <div class="name">${step.name}</div>
        <div class="detail">${step.volume > 0 ? step.volume.toLocaleString() + ' mm³' : '2D'}</div>
      </div>
    `;
    item.onclick = () => loadStep(i);

    // Put sketches in the sketches folder, features in the main list
    if (isSketch) {
      sketchesList.appendChild(item);
      hasSketch = true;
    } else {
      list.appendChild(item);
    }

    // Timeline step — thumbnail rendered async (no text overlay)
    const tStep = document.createElement('div');
    tStep.className = 'timeline-step';
    tStep.innerHTML = `<span class="step-num">${i + 1}</span>`;
    tStep.onclick = () => loadStep(i);
    timeline.appendChild(tStep);

    // Queue thumbnail for sequential rendering
    thumbQueue.push({ index: i, element: tStep });
  });

  // Show sketches folder if there are sketches
  sketchesFolder.style.display = hasSketch ? '' : 'none';

  // Render thumbnails sequentially to avoid WebGL race conditions
  renderThumbQueue();
}

// ── Thumbnail renderer (offscreen, serialized) ──
const thumbRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
thumbRenderer.setSize(160, 112);
thumbRenderer.toneMapping = THREE.ACESFilmicToneMapping;
let thumbQueue = [];
let thumbRendering = false;

async function renderThumbQueue() {
  if (thumbRendering) return;
  thumbRendering = true;
  while (thumbQueue.length > 0) {
    const { index, element } = thumbQueue.shift();
    const dataUrl = await renderThumbnail(index);
    if (dataUrl && element) {
      element.style.backgroundImage = `url(${dataUrl})`;
      element.style.backgroundSize = 'cover';
      element.style.backgroundPosition = 'center';
      const num = element.querySelector('.step-num');
      if (num) num.style.display = 'none';
    }
  }
  thumbRendering = false;
}

async function renderThumbnail(stepIndex) {
  try {
    const url = currentJobId ? `/api/job/${currentJobId}/step/${stepIndex}` : `/api/step/${stepIndex}`;
    const res = await fetch(url);
    if (!res.ok) return null;
    const data = await res.json();
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

// ── File Upload: two-step flow (preview mesh → analyze & build) ──
let pendingFile = null;
let pendingTempPath = null;

document.getElementById('file-input').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

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
      xhr.onload = () => resolve(JSON.parse(xhr.responseText));
      xhr.onerror = () => reject(new Error('Upload failed'));
      xhr.send(form);
    });

    loading.classList.remove('show');

    if (previewRes.error) {
      alert(previewRes.error);
      return;
    }

    // AI thinking messages about the uploaded mesh
    addChatMessage('ai', `Loaded ${previewRes.filename || 'mesh'}: ${previewRes.faces.toLocaleString()} faces, ${previewRes.vertices.toLocaleString()} vertices`, true);
    addChatMessage('ai', `Dimensions: ${previewRes.dimensions.map(d => d + 'mm').join(' × ')}. ${previewRes.watertight ? 'Mesh is watertight.' : 'Mesh has gaps — will repair.'}`, true);
    addChatMessage('ai', `Complexity: ${previewRes.complexity}. Estimated cost: ~${previewRes.estimated_tokens} tokens. Click "Analyze & Build" to proceed.`, true);

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
    } catch (e) { console.warn('Could not save preview to sessionStorage:', e); }

    // Reset UI from any previous job
    document.getElementById('feature-list').innerHTML = '';
    document.getElementById('timeline').innerHTML = '';
    currentStep = -1;
    currentJobId = null;

    // Show the raw mesh in the 3D viewer
    pendingTempPath = previewRes.temp_path;
    pendingFile = file;
    saveViewerState('preview');
    _showRawMesh(previewRes);

    // Show preview info bar with complexity meter
    document.getElementById('preview-bar').style.display = 'block';
    document.getElementById('preview-faces').textContent = previewRes.faces.toLocaleString() + ' faces';
    document.getElementById('preview-dims').textContent = previewRes.dimensions.map(d => d + 'mm').join(' × ');
    document.getElementById('preview-tokens').textContent = '~' + previewRes.estimated_tokens + ' tokens';

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
      addChatMessage('ai', '⚠️ ' + (previewRes.reject_reason || 'Mesh is too complex to process.'), true);
    } else {
      warning.style.display = 'none';
      document.getElementById('btn-analyze').style.display = '';
      if (score > 60) {
        addChatMessage('ai', `⚠️ High complexity (${score}/100). Results may be less accurate for this mesh. Works best on prismatic parts with flat faces and cylindrical holes.`, true);
      }
    }

    document.getElementById('btn-upload').style.display = 'none';
    document.getElementById('empty-state').style.display = 'none';

  } catch (err) {
    alert('Upload failed: ' + err.message);
    loading.classList.remove('show');
  } finally {
    e.target.value = '';
  }
});

// Show raw mesh triangles in the viewer (before pipeline)
function _showRawMesh(previewRes) {
  while (modelGroup.children.length) modelGroup.remove(modelGroup.children[0]);

  const verts = new Float32Array(previewRes.mesh_vertices);
  const faces = previewRes.mesh_faces;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
  geo.setIndex(faces);
  geo.computeVertexNormals();

  // Wireframe look — show this is a raw mesh, not a CAD model
  const mat = new THREE.MeshStandardMaterial({ color: 0x8B949E, metalness: 0.1, roughness: 0.8 });
  modelGroup.add(new THREE.Mesh(geo, mat));
  modelGroup.add(new THREE.LineSegments(
    new THREE.WireframeGeometry(geo),
    new THREE.LineBasicMaterial({ color: 0x30363D, linewidth: 1 })
  ));

  // Center model at origin
  const box = new THREE.Box3().setFromObject(modelGroup);
  const center = box.getCenter(new THREE.Vector3());
  modelGroup.position.sub(center);
  const size = box.getSize(new THREE.Vector3()).length();

  // Smooth animation from current camera position to above-right angle
  const meshHeight = box.getSize(new THREE.Vector3()).y;
  const camY = Math.max(size * 1.2, meshHeight * 2.5);
  animateCamera(
    { x: size * 0.7, y: camY, z: size * 0.7 },
    { x: 0, y: 0, z: 0 },
    1.5,
  );
}

// Analyze & Build button handler
document.getElementById('btn-analyze').addEventListener('click', async () => {
  if (!pendingTempPath) return;

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
      body: JSON.stringify({ temp_path: pendingTempPath, filename: pendingFile?.name || 'mesh' }),
    });
    const data = await res.json();
    if (data.error) { alert(data.error); loading.classList.remove('show'); return; }

    currentJobId = data.job_id;

    // Show AI thinking process in chat as pipeline runs
    addChatMessage('system', 'Pipeline started — analyzing mesh...');
    let lastStage = '';

    const thinkingMessages = {
      'Ingesting mesh...': '🔍 Reading mesh file, checking topology...',
      'Segmenting surfaces...': '🧩 Identifying surface regions — planes, cylinders, cones...',
      'Fitting primitives...': '📐 Fitting analytic surfaces to each region...',
      'Building topology...': '🔗 Analyzing edge relationships — sharp, fillet, chamfer...',
      'Recognizing features...': '🏗️ Detecting features — extrusions, holes, pockets, chamfers...',
      'Building CAD model...': '⚙️ Building parametric solid: sketch → extrude → cut → chamfer...',
      'Done': '✅ CAD model built successfully!',
    };

    // Poll progress and stream thinking to chat
    const statusData = await new Promise((resolve, reject) => {
      const poll = setInterval(async () => {
        try {
          const r = await fetch(`/api/job/${currentJobId}/status`);
          const d = await r.json();
          progressBar.style.width = d.progress + '%';
          loadingText.textContent = d.stage || 'Processing...';
          progressDetail.textContent = d.progress + '%';

          // Add thinking message when stage changes
          if (d.stage && d.stage !== lastStage) {
            lastStage = d.stage;
            const thought = thinkingMessages[d.stage];
            if (thought) addChatMessage('ai', thought, true);
          }

          if (d.status === 'done') { clearInterval(poll); resolve(d); }
          else if (d.status === 'error') { clearInterval(poll); reject(new Error(d.error)); }
        } catch (e) {}
      }, 500);
    });

    progressBar.style.width = '100%';
    loadingText.textContent = 'Loading model...';

    // Hide preview bar, show upload button again
    document.getElementById('preview-bar').style.display = 'none';
    document.getElementById('btn-analyze').style.display = 'none';
    document.getElementById('btn-upload').style.display = '';
    pendingTempPath = null;

    if (statusData.status === 'done') {
      // Summary message in chat
      const holes = statusData.steps.filter(s => s.name.includes('Hole')).length;
      const chamfers = statusData.steps.filter(s => s.name.includes('Chamfer')).length;
      const fillets = statusData.steps.filter(s => s.name.includes('Fillet')).length;
      let summary = `📊 Reconstructed with ${statusData.num_steps} features: 1 base extrusion`;
      if (holes > 0) summary += `, ${holes} hole${holes > 1 ? 's' : ''}`;
      if (chamfers > 0) summary += `, ${chamfers} chamfer${chamfers > 1 ? 's' : ''}`;
      if (fillets > 0) summary += `, ${fillets} fillet${fillets > 1 ? 's' : ''}`;
      addChatMessage('ai', summary);
      addChatMessage('system', 'Click features in the tree to step through the build. Ask me about dimensions or features.');

      // Run slicer comparison in background and show accuracy
      if (currentJobId) {
        fetch(`/api/job/${currentJobId}/compare`).then(r => r.json()).then(comp => {
          if (comp.score !== undefined) {
            addChatMessage('ai', `🎯 Cross-section accuracy: ${comp.score.toFixed(0)}% (${comp.per_slice?.length || 0} slices compared)`);
          }
        }).catch(() => {}); // silently skip if comparison fails
      }

      // Try AI dimension analysis — keep loading visible with status
      loadingText.textContent = 'Analyzing dimensions...';
      progressBar.style.width = '100%';
      try {
        const analysisRes = await fetch('/api/chat/analyze-mesh', {
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
        const analysis = await analysisRes.json();

        if (!analysis.skip) {
          loading.classList.remove('show');
          showDimensionModal(analysis, statusData);
          return; // Modal will call finishLoad when confirmed
        }
      } catch (err) {
        // AI unavailable — show basic dimension review from bbox
        if (statusData.bbox_min && statusData.bbox_max) {
          const dims = [0,1,2].map(i => Math.abs(statusData.bbox_max[i] - statusData.bbox_min[i]));
          dims.sort((a,b) => b - a);
          const basicAnalysis = {
            part_type: 'Prismatic part',
            suggested_dimensions: {
              width: Math.round(dims[0] * 10) / 10,
              height: Math.round(dims[1] * 10) / 10,
              thickness: Math.round(dims[2] * 10) / 10,
            },
            notes: 'Dimensions from bounding box. AI analysis unavailable.',
          };
          loading.classList.remove('show');
          showDimensionModal(basicAnalysis, statusData);
          return;
        }
      }

      loading.classList.remove('show');
      // No dimensions available — load directly
      finishLoad(statusData);
    } else {
      alert('Pipeline error: ' + (data.error || 'unknown'));
      document.getElementById('status-dot').className = 'status-dot';
    }
  } catch (err) {
    alert('Upload failed: ' + err.message);
    document.getElementById('status-dot').className = 'status-dot';
    loading.classList.remove('show');
  } finally {
    e.target.value = '';
  }
});

let cachedStatusData = null;

function finishLoad(statusData) {
  cachedStatusData = statusData;
  saveViewerState('cad');
  sessionStorage.removeItem('paramesh_preview'); // clear preview, CAD is now primary
  populateUI(statusData);
  document.getElementById('token-count').textContent = statusData.tokens_used;
  document.getElementById('status-dot').className = 'status-dot done';
  document.getElementById('btn-export').disabled = false;

  // Enable chat
  document.getElementById('chat-input').disabled = false;
  document.getElementById('chat-send').disabled = false;

  // Add system message to chat
  addChatMessage('system', `Model loaded: ${statusData.num_steps} features, ${statusData.mesh_volume.toLocaleString()} mm³`);

  if (statusData.steps.length > 0) {
    loadStep(statusData.steps.length - 1);
  }
}

// ── Dimension Review Modal ──
function showDimensionModal(analysis, statusData) {
  cachedStatusData = statusData;
  const modal = document.getElementById('dim-modal');
  const rows = document.getElementById('dim-rows');
  const notes = document.getElementById('dim-notes');

  document.getElementById('dim-part-type').innerHTML = `
    <div style="font-size:14px;color:var(--accent);margin-bottom:12px">
      Detected: <strong>${analysis.part_type || 'Unknown'}</strong>
    </div>`;

  rows.innerHTML = '';
  const dims = analysis.suggested_dimensions || {};
  for (const [key, val] of Object.entries(dims)) {
    const displayVal = Array.isArray(val) ? val.join(', ') : val;
    rows.innerHTML += `<div class="dim-row"><span class="label">${key}</span><span class="value"><input class="dim-input" type="text" data-key="${key}" value="${displayVal}"> mm</span></div>`;
  }

  notes.textContent = analysis.notes || '';
  if (analysis.usage) {
    document.getElementById('token-count').textContent = analysis.usage.calls_used || 0;
  }

  modal.classList.add('show');
}

// Called from HTML buttons
window.skipDimensionReview = function() {
  document.getElementById('dim-modal').classList.remove('show');
  if (cachedStatusData) finishLoad(cachedStatusData);
};

window.confirmDimensionReview = function() {
  // Read edited dimension values from inputs
  const inputs = document.querySelectorAll('#dim-rows .dim-input');
  const editedDims = {};
  inputs.forEach(inp => { editedDims[inp.dataset.key] = inp.value.trim(); });
  document.getElementById('dim-modal').classList.remove('show');
  if (cachedStatusData) finishLoad(cachedStatusData);
  const dimSummary = Object.entries(editedDims).map(([k, v]) => `${k}: ${v} mm`).join(', ');
  addChatMessage('system', `Dimensions confirmed: ${dimSummary}`);
};

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
    const context = await statusRes.json();

    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, context }),
    });
    const data = await res.json();

    typing.remove();
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
  const modal = document.getElementById('settings-modal');
  modal.classList.add('show');

  try {
    const res = await fetch('/api/usage');
    const u = await res.json();

    const limit = u.tokens_limit > 0 ? u.tokens_limit : 100;

    // Plan badge
    document.getElementById('settings-plan-badge').textContent = u.plan.charAt(0).toUpperCase() + u.plan.slice(1);
    document.getElementById('settings-plan-name').textContent = u.plan.charAt(0).toUpperCase() + u.plan.slice(1);
    document.getElementById('settings-upload').textContent = u.upload_limit_mb + ' MB';
    document.getElementById('settings-jobs').textContent = u.total_jobs;

    // Session (daily)
    fillBar('bar-session', 'num-session', 'lim-session', u.daily, limit);
    // Week
    fillBar('bar-week', 'num-week', 'lim-week', u.weekly, limit);
    // Month
    fillBar('bar-month', 'num-month', 'lim-month', u.monthly, limit);
  } catch (e) {}
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

async function autoLoad() {
  // Check sessionStorage for a saved mesh preview
  const saved = sessionStorage.getItem('paramesh_preview');
  if (saved) {
    try {
      const previewRes = JSON.parse(saved);
      if (previewRes.mesh_vertices && previewRes.mesh_vertices.length > 0) {
        document.getElementById('empty-state').style.display = 'none';
        _showRawMesh(previewRes);
        pendingTempPath = previewRes.temp_path;
        pendingFile = null;
        document.getElementById('btn-analyze').style.display = '';
        document.getElementById('btn-upload').style.display = 'none';
        return; // DONE — skip CAD load entirely
      }
    } catch (e) {
      console.error('Preview restore error:', e);
    }
  }
  // No preview — load last completed CAD job
  try {
    const res = await fetch('/api/status');
    if (!res.ok) return;
    const data = await res.json();
    if (data.status === 'done' && data.steps && data.steps.length > 0) {
      currentJobId = data.job_id || null;
      document.getElementById('empty-state').style.display = 'none';
      // Load the last viewed step (saved in localStorage)
      const lastStep = parseInt(localStorage.getItem('paramesh_last_step') || '-1');
      if (lastStep >= 0 && lastStep < data.steps.length) {
        populateUI(data);
        cachedStatusData = data;
        document.getElementById('status-dot').className = 'status-dot done';
        document.getElementById('btn-export').disabled = false;
        document.getElementById('chat-input').disabled = false;
        document.getElementById('chat-send').disabled = false;
        loadStep(lastStep);
      } else {
        finishLoad(data);
      }
    }
  } catch (e) { /* no previous job */ }
}
// ── Camera animation system (must be before autoLoad) ──
let cameraAnim = null;

function animateCamera(endPos, endTarget, duration = 1.5) {
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

  controls.update();
  renderer.render(scene, camera);

  // Sync axis gizmo rotation with main camera
  axisCamera.position.copy(camera.position).sub(controls.target).normalize().multiplyScalar(5);
  axisCamera.lookAt(0, 0, 0);
  axisRenderer.render(axisScene, axisCamera);
}
animate();
