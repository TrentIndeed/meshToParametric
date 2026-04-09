import { chromium } from 'playwright';
import { readFileSync } from 'fs';
import { resolve } from 'path';

const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
const page = await context.newPage();

page.on('console', msg => {
  if (msg.type() === 'error') console.log('ERR:', msg.text());
});

await page.goto('http://127.0.0.1:8080/');
await page.waitForTimeout(2000);

// Scroll to demo section
await page.evaluate(() => document.getElementById('demo')?.scrollIntoView({ behavior: 'instant' }));
await page.waitForTimeout(500);

// Upload Mesh1
const fileInput = await page.$('#demo-file-input');
await fileInput.setInputFiles('models/Mesh1/Body1Mesh.3mf');
await page.waitForTimeout(1000);

// Screenshot 1: File selected, ready to convert
await page.screenshot({ path: 'demo_1_upload.png' });
console.log('Saved demo_1_upload.png');

// Click Convert
const convertBtn = await page.$('#demo-convert-btn');
await convertBtn.click();
await page.waitForTimeout(3000);

// Screenshot 2: Processing/progress
await page.evaluate(() => document.getElementById('demo')?.scrollIntoView({ behavior: 'instant' }));
await page.waitForTimeout(500);
await page.screenshot({ path: 'demo_2_processing.png' });
console.log('Saved demo_2_processing.png');

// Wait for pipeline to finish (poll for up to 120 seconds)
for (let i = 0; i < 60; i++) {
  const done = await page.evaluate(() => {
    const preview = document.getElementById('demo-preview');
    return preview && preview.style.display !== 'none';
  });
  if (done) break;
  await page.waitForTimeout(2000);

  // Take a mid-processing screenshot at ~10s
  if (i === 5) {
    await page.evaluate(() => document.getElementById('demo')?.scrollIntoView({ behavior: 'instant' }));
    await page.waitForTimeout(200);
    await page.screenshot({ path: 'demo_2b_midprocess.png' });
    console.log('Saved demo_2b_midprocess.png');
  }
}

// Screenshot 3: Preview result
await page.evaluate(() => document.getElementById('demo')?.scrollIntoView({ behavior: 'instant' }));
await page.waitForTimeout(1000);
await page.screenshot({ path: 'demo_3_preview.png' });
console.log('Saved demo_3_preview.png');

// Scroll down to see the feature tree and gate
await page.evaluate(() => {
  const preview = document.getElementById('demo-preview');
  if (preview) preview.scrollIntoView({ behavior: 'instant', block: 'end' });
});
await page.waitForTimeout(500);
await page.screenshot({ path: 'demo_3b_gate.png' });
console.log('Saved demo_3b_gate.png');

console.log('Done!');
await browser.close();
