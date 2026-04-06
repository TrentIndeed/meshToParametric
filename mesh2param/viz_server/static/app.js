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
axisRenderer.setSize(120, 120);
axisRenderer.setClearColor(0x000000, 0);
console.debug('[init] Axis gizmo renderer created');
const axIsDark = document.documentElement.getAttribute('data-theme') === 'dark';
axisRenderer.domElement.style.cssText = `position:fixed;bottom:90px;left:10px;z-index:60;pointer-events:none;border:1px solid ${axIsDark ? 'rgba(48,54,61,0.5)' : 'rgba(208,215,222,0.5)'};border-radius:8px;`;
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

    if (data.is_sketch) {
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
        side: THREE.DoubleSide,
        polygonOffset: true,
        polygonOffsetFactor: 1,
        polygonOffsetUnits: 1,
      });
      const mesh = new THREE.Mesh(geo, mat);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      modelGroup.add(mesh);
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
  toast.textContent = `Step ${index + 1}/${steps.length}: ${data.name} (${Math.round(data.vertices.length/3)} verts)`;
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

  // Add "Original Mesh" button at the start of timeline
  const meshBtn = document.createElement('div');
  meshBtn.className = 'timeline-step timeline-special';
  meshBtn.innerHTML = '<span style="font-size:10px;color:var(--text-secondary);">Mesh</span>';
  meshBtn.title = 'Show original mesh';
  meshBtn.onclick = () => {
    if (savedMeshPreview) {
      _showRawMesh(savedMeshPreview, false);
      // Deactivate all steps
      document.querySelectorAll('.timeline-step').forEach(el => el.classList.remove('active'));
      document.querySelectorAll('.feature-item').forEach(el => el.classList.remove('active'));
      meshBtn.classList.add('active');
    }
  };
  timeline.appendChild(meshBtn);

  statusData.steps.forEach((step, i) => {
    // Determine icon
    let iconInfo = I('component.png', 'extrude'); // default fallback
    for (const [key, val] of Object.entries(iconMap)) {
      if (step.name.includes(key)) { iconInfo = val; break; }
    }

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

    // All features go in the body list (sketches are inline, like Onshape)
    list.appendChild(item);

    // Timeline step — thumbnail rendered async (no text overlay)
    const tStep = document.createElement('div');
    tStep.className = 'timeline-step';
    tStep.innerHTML = `<span class="step-num">${i + 1}</span>`;
    tStep.onclick = () => loadStep(i);
    timeline.appendChild(tStep);

    // Queue thumbnail for sequential rendering
    thumbQueue.push({ index: i, element: tStep });
  });

  // Add "Compare" split-screen button at the end of timeline
  const compareBtn = document.createElement('div');
  compareBtn.className = 'timeline-step timeline-special';
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

  // Render thumbnails sequentially to avoid WebGL race conditions
  console.debug('[populateUI] queued', thumbQueue.length, 'thumbnails for rendering');
  renderThumbQueue().then(() => {
    // Render Mesh thumbnail from saved preview data
    if (savedMeshPreview) {
      const url = renderMeshThumbnail(savedMeshPreview.mesh_vertices, savedMeshPreview.mesh_faces);
      if (url) {
        meshBtn.style.backgroundImage = `url(${url})`;
        meshBtn.style.backgroundSize = 'cover';
        meshBtn.style.backgroundPosition = 'center';
        meshBtn.querySelector('span').style.display = 'none';
      }
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
        const num = element.querySelector('.step-num');
        if (num) num.style.display = 'none';
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
      addChatMessage('ai', '⚠️ ' + (previewRes.reject_reason || 'Mesh is too complex to process.'), true);
    } else {
      warning.style.display = 'none';
      document.getElementById('btn-analyze').style.display = '';
      if (score > 75) {
        addChatMessage('ai', `This mesh has high geometric complexity. Results may vary — works best on prismatic parts like brackets, plates, and mounts.`, true);
      }
    }

    document.getElementById('btn-upload').style.display = 'none';
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

    // Wireframe look — show this is a raw mesh, not a CAD model
    const mat = new THREE.MeshStandardMaterial({ color: 0x8B949E, metalness: 0.1, roughness: 0.8 });
    modelGroup.add(new THREE.Mesh(geo, mat));
    modelGroup.add(new THREE.LineSegments(
      new THREE.WireframeGeometry(geo),
      new THREE.LineBasicMaterial({ color: 0x30363D, linewidth: 1 })
    ));
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
      const meshHeight = box.getSize(new THREE.Vector3()).y;
      const camY = Math.max(size * 1.2, meshHeight * 2.5);
      console.debug('[_showRawMesh] centering, bbox size:', size.toFixed(1), 'animating camera');
      animateCamera(
        { x: size * 0.7, y: camY, z: size * 0.7 },
        { x: 0, y: 0, z: 0 },
        1.5,
      );
    } else {
      controls.target.set(0, 0, 0);
      controls.update();
    }
  } catch (e) {
    console.error('[_showRawMesh] bounding box / camera setup failed', e);
  }
}

// Analyze & Build button handler
document.getElementById('btn-analyze').addEventListener('click', async () => {
  if (!pendingTempPath) return;
  console.debug('[analyze] starting pipeline, temp_path:', pendingTempPath);

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
    if (!res.ok) { console.error('[analyze] /api/run-pipeline HTTP', res.status); alert('Pipeline request failed: HTTP ' + res.status); loading.classList.remove('show'); return; }
    let data;
    try { data = await res.json(); } catch (e) { console.error('[analyze] JSON parse failed from /api/run-pipeline', e); alert('Pipeline response invalid'); loading.classList.remove('show'); return; }
    if (data.error) { console.error('[analyze] pipeline error:', data.error); alert(data.error); loading.classList.remove('show'); return; }

    currentJobId = data.job_id;
    console.debug('[analyze] job submitted, id:', currentJobId);

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
        } catch (e) { console.warn('[analyze] poll error:', e); }
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
        console.debug('[analyze] requesting slicer comparison for job:', currentJobId);
        fetch(`/api/job/${currentJobId}/compare`).then(r => {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        }).then(comp => {
          if (comp.score !== undefined) {
            console.debug('[analyze] comparison score:', comp.score.toFixed(1), '%, slices:', comp.per_slice?.length);
            addChatMessage('ai', `🎯 Cross-section accuracy: ${comp.score.toFixed(0)}% (${comp.per_slice?.length || 0} slices compared)`);
          }
        }).catch(e => { console.warn('[analyze] slicer comparison failed (non-critical):', e.message); });
      }

      // Try AI dimension analysis — keep loading visible with status
      loadingText.textContent = 'Analyzing dimensions...';
      progressBar.style.width = '100%';
      try {
        console.debug('[analyze] requesting AI dimension analysis');
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
        if (!analysisRes.ok) { console.warn('[analyze] dimension analysis HTTP', analysisRes.status); throw new Error('HTTP ' + analysisRes.status); }
        const analysis = await analysisRes.json();
        console.debug('[analyze] dimension analysis received, skip:', analysis.skip);

        if (!analysis.skip) {
          loading.classList.remove('show');
          showDimensionModal(analysis, statusData);
          return; // Modal will call finishLoad when confirmed
        }
      } catch (err) {
        console.warn('[analyze] AI dimension analysis unavailable, falling back to bbox:', err.message);
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
    console.error('[analyze] pipeline error:', err);
    alert('Pipeline error: ' + err.message);
    document.getElementById('status-dot').className = 'status-dot';
    loading.classList.remove('show');
  }
});

// Rerun pipeline with the same mesh
window.rerunPipeline = function() {
  if (!pendingTempPath) return;
  document.getElementById('btn-rerun').style.display = 'none';
  document.getElementById('btn-analyze').click();
};

let cachedStatusData = null;

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
  document.getElementById('token-count').textContent = statusData.parts_used || statusData.tokens_used || 0;
  document.getElementById('status-dot').className = 'status-dot done';
  document.getElementById('btn-export').disabled = false;
  document.getElementById('btn-rerun').style.display = pendingTempPath ? '' : 'none';

  // Enable chat
  document.getElementById('chat-input').disabled = false;
  document.getElementById('chat-send').disabled = false;

  // Add system message to chat
  addChatMessage('system', `Model loaded: ${statusData.num_steps} features, ${statusData.mesh_volume.toLocaleString()} mm³`);

  if (statusData.steps.length > 0) {
    loadStep(statusData.steps.length - 1).then(() => {
      // Animate camera only when CAD model first loads
      try {
        const box = new THREE.Box3().setFromObject(modelGroup);
        const s = box.getSize(new THREE.Vector3());
        const size = s.length();
        const camY = Math.max(size * 1.2, s.y * 2.5);
        console.debug('[finishLoad] model loaded, bbox size:', size.toFixed(1), 'animating camera');
        animateCamera(
          { x: size * 0.7, y: camY, z: size * 0.7 },
          { x: 0, y: 0, z: 0 },
          1.2,
        );
      } catch (e) {
        console.error('[finishLoad] camera animation failed', e);
      }
    });
  }
}

// ── Dimension Review Modal ──
function showDimensionModal(analysis, statusData) {
  console.debug('[showDimensionModal] part_type:', analysis.part_type, 'dims:', Object.keys(analysis.suggested_dimensions || {}));
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

async function autoLoad() {
  console.debug('[autoLoad] checking for saved state...');
  // Check sessionStorage for a saved mesh preview
  const saved = sessionStorage.getItem('paramesh_preview');
  if (saved) {
    try {
      const previewRes = JSON.parse(saved);
      if (previewRes.mesh_vertices && previewRes.mesh_vertices.length > 0) {
        console.debug('[autoLoad] restoring preview from sessionStorage, verts:', previewRes.mesh_vertices.length);
        document.getElementById('empty-state').style.display = 'none';
        _showRawMesh(previewRes);
        pendingTempPath = previewRes.temp_path;
        pendingFile = null;
        document.getElementById('btn-analyze').style.display = '';
        document.getElementById('btn-upload').style.display = 'none';
        return; // DONE — skip CAD load entirely
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
        document.getElementById('chat-input').disabled = false;
        document.getElementById('chat-send').disabled = false;
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
