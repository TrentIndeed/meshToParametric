import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';

// ── State ──
let currentStep = -1;
let steps = [];
let currentJobId = null;

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
  const url = currentJobId ? `/api/job/${currentJobId}/step/${index}` : `/api/step/${index}`;
  const res = await fetch(url);
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
    const url = currentJobId ? `/api/job/${currentJobId}/step/${stepIndex}` : `/api/step/${stepIndex}`;
    const res = await fetch(url);
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

    // Show the raw mesh in the 3D viewer
    pendingTempPath = previewRes.temp_path;
    pendingFile = file;
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

  const box = new THREE.Box3().setFromObject(modelGroup);
  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3()).length();
  camera.position.set(center.x + size, center.y + size * 0.7, center.z + size);
  controls.target.copy(center);
  controls.update();
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
      'Ingesting mesh...': '🔍 Reading mesh file, checking topology, counting faces...',
      'Segmenting surfaces...': '🧩 Clustering faces into surface groups using normal angles and curvature...',
      'Fitting primitives...': '📐 Fitting planes, cylinders, and cones to each surface region via RANSAC...',
      'Building topology...': '🔗 Building adjacency graph, classifying edges as sharp/fillet/chamfer...',
      'Recognizing features...': '🏗️ Identifying base extrusion, holes, pockets, chamfers, fillets...',
      'Building CAD model...': '⚙️ Constructing CadQuery solid: sketch → extrude → cut holes → chamfer...',
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
      let summary = `📊 Found ${statusData.num_steps} construction steps: 1 base extrusion`;
      if (holes > 0) summary += `, ${holes} holes`;
      if (chamfers > 0) summary += `, ${chamfers} chamfer groups`;
      if (fillets > 0) summary += `, ${fillets} fillet groups`;
      summary += `. Volume error: ${((statusData.cad_volume - statusData.mesh_volume) / statusData.mesh_volume * 100).toFixed(1)}%`;
      addChatMessage('ai', summary);
      addChatMessage('system', 'Ask me to describe features, resize holes, or modify the model.');

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

// ── Auto-load if pipeline already ran ──
async function autoLoad() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.status === 'done' && data.steps.length > 0) {
      currentJobId = data.job_id || null;
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
