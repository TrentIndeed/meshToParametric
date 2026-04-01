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

// ── Studio Lighting (3-point + hemisphere) ──
const hemi = new THREE.HemisphereLight(0xC8D8E8, 0x2D3436, 0.6);
scene.add(hemi);

const key = new THREE.DirectionalLight(0xFFFFFF, 1.2);
key.position.set(150, 250, 120);
key.castShadow = true;
key.shadow.mapSize.width = 1024;
key.shadow.mapSize.height = 1024;
scene.add(key);

const fill = new THREE.DirectionalLight(0x8899BB, 0.5);
fill.position.set(-120, 80, -80);
scene.add(fill);

const rim = new THREE.DirectionalLight(0xFFFFFF, 0.3);
rim.position.set(-50, -80, 150);
scene.add(rim);

const bottom = new THREE.DirectionalLight(0x445566, 0.15);
bottom.position.set(0, -200, 0);
scene.add(bottom);

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

    // Feature edges only (hide internal triangle diagonals)
    const edgeGeo = new THREE.EdgesGeometry(geo, 25);
    const edgeMat = new THREE.LineBasicMaterial({ color: 0x2D3436, linewidth: 1.5 });
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
      <div class="texts">
        <div class="name">${step.name}</div>
        <div class="detail">${step.volume > 0 ? step.volume.toLocaleString() + ' mm³' : '2D'}</div>
      </div>
    `;
    item.onclick = () => loadStep(i);
    list.appendChild(item);

    // Timeline step — thumbnail rendered async (no text overlay)
    const tStep = document.createElement('div');
    tStep.className = 'timeline-step';
    tStep.innerHTML = `<span class="step-num">${i + 1}</span>`;
    tStep.onclick = () => loadStep(i);
    timeline.appendChild(tStep);

    // Generate thumbnail in background
    renderThumbnail(i).then(dataUrl => {
      if (dataUrl) {
        tStep.style.backgroundImage = `url(${dataUrl})`;
        tStep.style.backgroundSize = 'cover';
        tStep.style.backgroundPosition = 'center';
        tStep.querySelector('.step-num').style.display = 'none';
      }
    });
  });
}

// ── Thumbnail renderer (offscreen) ──
const thumbRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
thumbRenderer.setSize(160, 112);
thumbRenderer.toneMapping = THREE.ACESFilmicToneMapping;

async function renderThumbnail(stepIndex) {
  try {
    const res = await fetch(`/api/step/${stepIndex}`);
    if (!res.ok) return null;
    const data = await res.json();

    const thumbScene = new THREE.Scene();
    thumbScene.background = new THREE.Color(0x161B22);

    // Lights
    thumbScene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const light = new THREE.DirectionalLight(0xffffff, 0.9);
    light.position.set(150, 200, 100);
    thumbScene.add(light);

    const verts = new Float32Array(data.vertices);

    if (data.is_sketch) {
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
      thumbScene.add(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x58A6FF })));
    } else {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
      geo.setIndex(data.faces);
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
    const thumbCam = new THREE.PerspectiveCamera(45, 160 / 112, 0.1, 5000);
    thumbCam.position.set(center.x + size * 0.8, center.y + size * 0.6, center.z + size * 0.8);
    thumbCam.lookAt(center);

    thumbRenderer.render(thumbScene, thumbCam);
    return thumbRenderer.domElement.toDataURL('image/png');
  } catch (e) {
    return null;
  }
}

// ── File Upload (with dimension analysis before pipeline) ──
let pendingFile = null;

document.getElementById('file-input').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  pendingFile = file;

  const loading = document.getElementById('loading');
  const loadingText = document.getElementById('loading-text');
  const progressBar = document.getElementById('progress-bar');
  const progressDetail = document.getElementById('progress-detail');
  loading.classList.add('show');
  loadingText.textContent = `Uploading ${file.name}...`;
  progressBar.style.width = '0%';
  progressDetail.textContent = '';
  document.getElementById('status-dot').className = 'status-dot processing';

  const form = new FormData();
  form.append('file', file);

  // Poll progress while pipeline runs
  let pollInterval = null;
  function startPolling() {
    pollInterval = setInterval(async () => {
      try {
        const pr = await fetch('/api/progress');
        const pd = await pr.json();
        progressBar.style.width = pd.progress + '%';
        if (pd.stage) {
          loadingText.textContent = pd.stage;
          progressDetail.textContent = pd.progress + '%';
        }
      } catch (e) {}
    }, 500);
  }

  try {
    startPolling();
    const res = await fetch('/api/upload', { method: 'POST', body: form });
    clearInterval(pollInterval);
    progressBar.style.width = '100%';
    loadingText.textContent = 'Loading viewer...';
    const data = await res.json();

    if (data.status === 'done') {
      const statusRes = await fetch('/api/status');
      const statusData = await statusRes.json();

      // Try AI dimension analysis
      loading.classList.remove('show');
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
          showDimensionModal(analysis, statusData);
          return; // Modal will call finishLoad when confirmed
        }
      } catch (err) {
        // AI unavailable, skip dimension review
      }

      // No AI or skipped — load directly
      finishLoad(statusData);
    } else {
      alert('Pipeline error: ' + (data.error || 'unknown'));
      document.getElementById('status-dot').className = 'status-dot';
    }
  } catch (err) {
    clearInterval(pollInterval);
    alert('Upload failed: ' + err.message);
    document.getElementById('status-dot').className = 'status-dot';
    loading.classList.remove('show');
  } finally {
    clearInterval(pollInterval);
    e.target.value = '';
  }
});

let cachedStatusData = null;

function finishLoad(statusData) {
  cachedStatusData = statusData;
  populateUI(statusData);
  document.getElementById('token-count').textContent = statusData.tokens_used;
  document.getElementById('status-dot').className = 'status-dot done';
  document.getElementById('btn-onshape').disabled = false;

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
    if (Array.isArray(val)) {
      rows.innerHTML += `<div class="dim-row"><span class="label">${key}</span><span class="value">[${val.join(', ')}] mm</span></div>`;
    } else {
      rows.innerHTML += `<div class="dim-row"><span class="label">${key}</span><span class="value">${val} mm</span></div>`;
    }
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
  document.getElementById('dim-modal').classList.remove('show');
  if (cachedStatusData) finishLoad(cachedStatusData);
  addChatMessage('system', 'Dimensions confirmed by AI analysis');
};

// ── AI Chat ──
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSend = document.getElementById('chat-send');

function addChatMessage(role, text) {
  const msg = document.createElement('div');
  msg.className = `chat-msg ${role}`;
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

// ── Auto-load if pipeline already ran ──
async function autoLoad() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.status === 'done' && data.steps.length > 0) {
      finishLoad(data);
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
