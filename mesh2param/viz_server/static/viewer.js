import * as THREE from 'https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js';
import { OrbitControls } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

const container = document.getElementById('canvas-container');

// Scene setup
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0F172A);

const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.set(50, 50, 50);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight - 48);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;

// Lights
const ambient = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambient);
const directional = new THREE.DirectionalLight(0xffffff, 0.8);
directional.position.set(50, 100, 50);
scene.add(directional);

// Grid
const grid = new THREE.GridHelper(100, 20, 0x1E293B, 0x1E293B);
scene.add(grid);

// Load model from query param
const params = new URLSearchParams(window.location.search);
const modelPath = params.get('model');

if (modelPath) {
    const loader = new GLTFLoader();
    loader.load(modelPath, (gltf) => {
        const model = gltf.scene;
        // Center the model
        const box = new THREE.Box3().setFromObject(model);
        const center = box.getCenter(new THREE.Vector3());
        model.position.sub(center);
        scene.add(model);

        // Fit camera
        const size = box.getSize(new THREE.Vector3()).length();
        camera.position.set(size, size, size);
        controls.target.set(0, 0, 0);
        controls.update();
    });
} else {
    // Show a default box
    const geometry = new THREE.BoxGeometry(10, 20, 30);
    const material = new THREE.MeshStandardMaterial({ color: 0x6366F1, metalness: 0.3, roughness: 0.7 });
    const mesh = new THREE.Mesh(geometry, material);
    scene.add(mesh);
}

// Resize handler
window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / (window.innerHeight - 48);
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight - 48);
});

// Animate
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();
