# Secret configuration

Keep actual credentials in an ignored local `.env` file or the deployment's
secret store. Never commit session cookies, API keys, or encryption secrets.

Docker Compose requires `AUTH_SECRET` explicitly. Generate a strong random
value and set it privately before starting the service. There is no default
production signing secret.

Onshape token encryption requires `ONSHAPE_TOKEN_KEY` or `AUTH_SECRET`, loaded
from the environment or local dotenv configuration. Keep this value stable
across restarts. Missing configuration raises an error instead of using a
shared development key. Changing an existing key makes previously stored
tokens unreadable, so plan for users to reconnect their Onshape accounts.

The local Playwright scripts `gen_screenshots.mjs`, `test_home.mjs`, and
`test_zoom.mjs` require `MESH2PARAM_AUTH_TOKEN`. Supply a fresh, disposable
local session token through your process environment, then clear it when
finished. The scripts reject a missing value before launching a browser.

If a credential was previously committed, invalidate it wherever it was
accepted before considering history cleanup. Removing the current literal
does not erase older commits or other branches. Do not reuse the old browser
token or development encryption fallback.
