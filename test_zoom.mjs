import { chromium } from 'playwright';

const AUTH_TOKEN = process.env.MESH2PARAM_AUTH_TOKEN;
if (!AUTH_TOKEN) {
  throw new Error('Set MESH2PARAM_AUTH_TOKEN to a disposable local test token.');
}
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
await context.addCookies([{ name: 'paramesh_token', value: AUTH_TOKEN, domain: '127.0.0.1', path: '/' }]);

const page = await context.newPage();
page.on('console', msg => {
  if (msg.text().includes('[wheel]')) console.log('  ' + msg.text());
});

await page.goto('http://127.0.0.1:8080/app');
await page.waitForSelector('#viewer canvas', { timeout: 10000 });
await page.waitForTimeout(5000);

const viewer = await page.$('#viewer');
const box = await viewer.boundingBox();
const cx = box.x + box.width / 2;
const cy = box.y + box.height / 2;

await page.mouse.move(cx, cy);

// Simulate rapid scrolling like a real user (deltaY=111 like user's mouse)
console.log('--- Rapid zoom IN (10 ticks, no pause) ---');
for (let i = 0; i < 10; i++) {
  await page.mouse.wheel(0, -111);
  await page.waitForTimeout(50); // rapid fire
}
await page.waitForTimeout(500);
await page.screenshot({ path: 'zfix_in.png' });

console.log('\n--- Rapid zoom OUT (15 ticks, no pause) ---');
for (let i = 0; i < 15; i++) {
  await page.mouse.wheel(0, 111);
  await page.waitForTimeout(50);
}
await page.waitForTimeout(500);
await page.screenshot({ path: 'zfix_out.png' });

console.log('\nDone!');
await browser.close();
