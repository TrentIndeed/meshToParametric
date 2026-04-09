import { chromium } from 'playwright';
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
await page.goto('http://127.0.0.1:8080/');
await page.waitForTimeout(2000);

// Full page scroll screenshots
for (let i = 0; i < 8; i++) {
  await page.evaluate((y) => window.scrollTo(0, y), i * 800);
  await page.waitForTimeout(500);
  await page.screenshot({ path: `landing_scroll_${i}.png` });
  console.log(`Saved landing_scroll_${i}.png (scroll=${i * 800})`);
}
await browser.close();
