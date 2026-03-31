import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

// ── State ──
let currentStep = -1;
let steps = [];

// ── Three.js Setup ──
const container = document.getElementById('viewer');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0D1117);

const camera = new THREE.PerspectiveCamera(
  45, container.clientWidth / container.clientHeight, 0.1, 5000
);
camera.position.set(200, 150, 200);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.minDistance = 10;
controls.maxDistance = 2000;

// ── Lighting ──
const ambient = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambient);

const key = new THREE.DirectionalLight(0xffffff, 0.9);
key.position.set(150, 200, 100);
scene.add(key);

const fill = new THREE.DirectionalLight(0x8888ff, 0.3);
fill.position.set(-100, 50, -100);
scene.add(fill);

const rim = new THREE.DirectionalLight(0xffffff, 0.2);
rim.position.set(0, -100, 0);
scene.add(rim);

// ── Grid + Axes ──
function createGrid() {
  const gridSize = 300;
  const divisions = 30;
  const grid = new THREE.GridHelper(gridSize, divisions, 0x1E293B, 0x161B22);
  grid.position.y = -0.1;
  scene.add(grid);

  // XYZ axes
  const axLen = gridSize * 0.6;
  const axes = [
    { dir: [axLen, 0, 0], color: 0xF85149 },   // X red
    { dir: [0, axLen, 0], color: 0x3FB950 },    // Y green
    { dir: [0, 0, axLen], color: 0x58A6FF },    // Z blue
  ];
  axes.forEach(({ dir, color }) => {
    const geo = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(...dir),
    ]);
    const mat = new THREE.LineBasicMaterial({ color, linewidth: 2, transparent: true, opacity: 0.6 });
    scene.add(new THREE.Line(geo, mat));
  });
}
createGrid();

// ── Model group ──
let modelGroup = new THREE.Group();
scene.add(modelGroup);

// ── Load step geometry ──
async function loadStep(index) {
  const res = await fetch(`/api/step/${index}`);
  if (!res.ok) return;
  const data = await res.json();

  // Clear previous model
  while (modelGroup.children.length) modelGroup.remove(modelGroup.children[0]);

  const verts = new Float32Array(data.vertices);
  const indices = data.faces;

  if (data.is_sketch) {
    // Draw as lines
    const points = [];
    const numPts = verts.length / 3;
    for (let i = 0; i < numPts; i++) {
      const next = (i + 1) % numPts;
      points.push(
        new THREE.Vector3(verts[i*3], verts[i*3+1], verts[i*3+2]),
        new THREE.Vector3(verts[next*3], verts[next*3+1], verts[next*3+2]),
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

    const mat = new THREE.MeshStandardMaterial({
      color: 0xB0B8C0,
      metalness: 0.4,
      roughness: 0.35,
      envMapIntensity: 0.5,
    });
    const mesh = new THREE.Mesh(geo, mat);
    modelGroup.add(mesh);

    // Feature edges only (hide internal triangle diagonals)
    const edgeGeo = new THREE.EdgesGeometry(geo, 25);
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x30363D, linewidth: 1 });
    modelGroup.add(new THREE.LineSegments(edgeGeo, edgeMat));
  }

  // Center and fit camera on first load
  const box = new THREE.Box3().setFromObject(modelGroup);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();

  controls.target.copy(center);
  if (currentStep <= 0) {
    camera.position.set(center.x + size, center.y + size * 0.7, center.z + size);
  }
  controls.update();

  // Update UI
  currentStep = index;
  updateActiveStates();
  document.getElementById('empty-state').style.display = 'none';
}

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

  const iconMap = {
    'Sketch': { cls: 'sketch', icon: 'ri-pencil-line' },
    'Extrude': { cls: 'extrude', icon: 'ri-box-3-line' },
    'Through': { cls: 'hole', icon: 'ri-record-circle-line' },
    'Blind': { cls: 'hole', icon: 'ri-focus-3-line' },
    'Original': { cls: 'sketch', icon: 'ri-file-3-line' },
  };

  statusData.steps.forEach((step, i) => {
    // Determine icon
    let iconInfo = { cls: 'extrude', icon: 'ri-shape-line' };
    for (const [key, val] of Object.entries(iconMap)) {
      if (step.name.includes(key)) { iconInfo = val; break; }
    }

    // Feature tree item
    const item = document.createElement('div');
    item.className = 'feature-item';
    item.innerHTML = `
      <div class="icon ${iconInfo.cls}"><i class="${iconInfo.icon}"></i></div>
      <div>
        <div class="name">${step.name}</div>
        <div class="detail">${step.volume > 0 ? step.volume.toLocaleString() + ' mm³' : '2D'}</div>
      </div>
    `;
    item.onclick = () => loadStep(i);
    list.appendChild(item);

    // Timeline step
    const tStep = document.createElement('div');
    tStep.className = 'timeline-step';
    tStep.innerHTML = `
      <span class="step-num">${i + 1}</span>
      <span class="label">${step.name.split('(')[0].trim()}</span>
    `;
    tStep.onclick = () => loadStep(i);
    timeline.appendChild(tStep);
  });
}

// ── File Upload ──
document.getElementById('file-input').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  const loading = document.getElementById('loading');
  const loadingText = document.getElementById('loading-text');
  loading.classList.add('show');
  loadingText.textContent = `Processing ${file.name}...`;
  document.getElementById('status-dot').className = 'status-dot processing';

  const form = new FormData();
  form.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    const data = await res.json();

    if (data.status === 'done') {
      // Fetch status and populate UI
      const statusRes = await fetch('/api/status');
      const statusData = await statusRes.json();
      populateUI(statusData);
      document.getElementById('token-count').textContent = statusData.tokens_used;
      document.getElementById('status-dot').className = 'status-dot done';
      document.getElementById('btn-onshape').disabled = false;

      // Load the last construction step (final model)
      if (statusData.steps.length > 0) {
        await loadStep(statusData.steps.length - 1);
      }
    } else {
      alert('Pipeline error: ' + (data.error || 'unknown'));
      document.getElementById('status-dot').className = 'status-dot';
    }
  } catch (err) {
    alert('Upload failed: ' + err.message);
    document.getElementById('status-dot').className = 'status-dot';
  } finally {
    loading.classList.remove('show');
    e.target.value = '';
  }
});

// ── Resize ──
window.addEventListener('resize', () => {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

// ── Auto-load if pipeline already ran ──
async function autoLoad() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.status === 'done' && data.steps.length > 0) {
      populateUI(data);
      document.getElementById('token-count').textContent = data.tokens_used;
      document.getElementById('status-dot').className = 'status-dot done';
      document.getElementById('btn-onshape').disabled = false;
      await loadStep(data.steps.length - 1);
    }
  } catch (e) { /* server not ready yet */ }
}
autoLoad();

// ── Animate ──
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
