import { chromium } from 'playwright';

const AUTH_TOKEN = process.env.MESH2PARAM_AUTH_TOKEN;
if (!AUTH_TOKEN) {
  throw new Error('Set MESH2PARAM_AUTH_TOKEN to a disposable local test token.');
}
const OUT = 'mesh2param/viz_server/static';

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
await context.addCookies([{ name: 'paramesh_token', value: AUTH_TOKEN, domain: '127.0.0.1', path: '/' }]);
const page = await context.newPage();

await page.goto('http://127.0.0.1:8080/app');
await page.waitForSelector('#viewer canvas', { timeout: 10000 });
await page.waitForTimeout(5000);

// Step 1: Upload view — show the mesh in the viewer (already loaded from autoLoad)
// Click "Mesh" button to show the original mesh
const meshBtn = await page.$('.timeline-special');
if (meshBtn) {
  await meshBtn.click();
  await page.waitForTimeout(2000);
}
// Crop to just the viewer area
const viewer = await page.$('#viewer');
await viewer.screenshot({ path: `${OUT}/step1-upload.jpg`, type: 'jpeg', quality: 90 });
console.log('Step 1: Upload mesh screenshot saved');

// Step 2: Feature tree — click first step to show the sketch/extrude, capture sidebar + viewer
const firstStep = await page.$('.timeline-step:not(.timeline-special)');
if (firstStep) {
  await firstStep.click();
  await page.waitForTimeout(2000);
}
await page.screenshot({ path: `${OUT}/step2-features.jpg`, type: 'jpeg', quality: 90 });
console.log('Step 2: Feature recognition screenshot saved');

// Step 3: Final CAD model — click last real step
const allSteps = await page.$$('.timeline-step:not(.timeline-special)');
if (allSteps.length > 0) {
  await allSteps[allSteps.length - 1].click();
  await page.waitForTimeout(2000);
}
await viewer.screenshot({ path: `${OUT}/step3-cad.jpg`, type: 'jpeg', quality: 90 });
console.log('Step 3: CAD model screenshot saved');

// Step 4: Compare view — click Compare button
const compareBtn = await page.$('.timeline-special:last-child');
if (compareBtn) {
  const text = await compareBtn.textContent();
  if (text.includes('Compare')) {
    await compareBtn.click();
    await page.waitForTimeout(3000);
  }
}
await viewer.screenshot({ path: `${OUT}/step4-export.jpg`, type: 'jpeg', quality: 90 });
console.log('Step 4: Export/compare screenshot saved');

console.log('Done! Screenshots at', OUT);
await browser.close();
