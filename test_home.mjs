import { chromium } from 'playwright';
const AUTH_TOKEN = process.env.MESH2PARAM_AUTH_TOKEN;
if (!AUTH_TOKEN) {
  throw new Error('Set MESH2PARAM_AUTH_TOKEN to a disposable local test token.');
}
const browser = await chromium.launch({ headless: true });
const context = await browser.newContext({ viewport: { width: 1400, height: 900 } });
await context.addCookies([{ name: 'paramesh_token', value: AUTH_TOKEN, domain: '127.0.0.1', path: '/' },
  { name: 'paramesh_user', value: 'testbot', domain: '127.0.0.1', path: '/' }]);
const page = await context.newPage();
page.on('console', msg => { if (msg.type() === 'error') console.log('ERR:', msg.text()); });
await page.goto('http://127.0.0.1:8080/home');
await page.waitForTimeout(2000);
await page.screenshot({ path: 'home_new.png' });
console.log('Saved home_new.png');
await browser.close();
