# Onshape Integration Plan — seamless round-trip

Decisions locked (2026-06-13): **OAuth 2.0** per-user auth; **full
bidirectional loop** (export, re-sync, import); **design for the
Onshape App Store from the start** (launch Paramesh from inside
Onshape). Pro-tier feature. Billing stays in Stripe; Onshape is the
integration surface, not the biller.

This supersedes the placeholder items in TODO.md's Onshape section.
The live export engine itself (`onshape/feature_builder.py`,
`onshape/client.py`, `_live_export`) is built and proven on RackMount;
what remains is the multi-user account plumbing and the round-trip UX.

---

## 0. Guiding principle

Two facts make "switch back and forth" work; everything else is detail:

1. **Each user exports to their OWN Onshape account** (not the server
   owner's `.env` keys). → per-user OAuth tokens.
2. **Each `MeshJob` is LINKED to one Onshape document** (did/wid/eid).
   → re-export updates that doc instead of spawning duplicates, and we
   can deep-link both directions.

Today only the server-owner path exists (Basic auth, one account). The
whole plan is replacing that with per-user identity + a persistent
job↔document link.

---

## 1. Onshape app registration (do first — gates everything)

Register Paramesh as an OAuth application in the Onshape Developer
Portal (dev-portal.onshape.com). Capture:

- **client_id / client_secret** → `.env` as `ONSHAPE_OAUTH_CLIENT_ID` /
  `ONSHAPE_OAUTH_CLIENT_SECRET` (gitignored, like the existing keys).
- **Redirect URIs** (HTTPS, exact match): prod
  `https://<host>/api/onshape/callback` + a localhost entry for dev.
- **Scopes** (least-privilege): `OAuth2Read`, `OAuth2Write`,
  `OAuth2Profile` (user identity). Add `OAuth2Delete` only if we let
  users delete docs from Paramesh.
- **Store extensions** (the App-Store hooks — configurable now, the
  public listing comes later after review):
  - **Export action** on Part Studio + Assembly: "Convert / clean with
    Paramesh." Onshape opens our URL with the doc context → import flow.
  - **Element right-panel tab**: embeds Paramesh in an iframe inside
    Onshape (the in-Onshape entry point).
  - **Uninstall/lifecycle webhook**: purge stored tokens when a user
    removes the app.

OAuth endpoints (v6): authorize `https://oauth.onshape.com/oauth/authorize`,
token `https://oauth.onshape.com/oauth/token`. API calls stay on
`https://cad.onshape.com/api/v6/...` but with `Authorization: Bearer`.

---

## 2. Data model

`User` (add, all nullable; tokens ENCRYPTED at rest):
- `onshape_access_token_enc`, `onshape_refresh_token_enc` (Fernet;
  key from new `ONSHAPE_TOKEN_KEY` env, NOT reusing AUTH_SECRET).
- `onshape_token_expiry` (DateTime, UTC).
- `onshape_user_id`, `onshape_user_name` (display "Connected as …").
- `onshape_connected_at`.

`MeshJob` (add — the job↔doc link):
- `onshape_document_id`, `onshape_workspace_id`, `onshape_element_id`.
- `onshape_url`, `onshape_exported_at`, `onshape_export_status`.

Migrations go in `db/database.py::_migrate_db` (idempotent `_add_col`,
same pattern already there).

---

## 3. OAuth flow (Phase 1 core)

New router `api/onshape_oauth.py`:

- `GET /api/onshape/connect` → build authorize URL with a signed,
  expiring `state` (CSRF; reuse the `_sign` helper from auth.py), set
  it in a short-lived cookie, 302 to Onshape.
- `GET /api/onshape/callback?code&state` → verify `state`, exchange
  `code` for tokens at the token endpoint, fetch `/users/sessioninfo`
  for identity, encrypt + store on the `User`, redirect back to the
  app with a success toast.
- `POST /api/onshape/disconnect` → clear stored tokens (and call the
  Onshape token-revoke endpoint).
- `GET /api/onshape/status` → `{connected, onshape_user_name}` for the
  UI to show connect/disconnect state.

Token lifecycle: a `get_onshape_client_for_user(user, db)` helper
decrypts the token, refreshes via the refresh_token when within ~60s of
expiry (or on a 401), persists the rotated tokens, and returns an
authenticated `OnshapeClient`. If refresh fails → mark disconnected,
surface "reconnect Onshape."

---

## 4. Client changes (`onshape/client.py`)

`OnshapeClient` currently does Basic auth with an API key pair. Add a
**Bearer** path without disturbing the existing one:
- `OnshapeClient.from_oauth_token(access_token, base_url, timeout)`.
- Internally just swaps the `Authorization` header
  (`Bearer <token>` vs `Basic <b64>`); every endpoint method is
  unchanged.
- `_live_export` takes an optional `client` arg; when present it uses
  it (per-user), else falls back to `from_env()` (owner account / CLI).

This keeps the CLI + single-user testing path working while the web
path goes per-user.

---

## 5. Phase 1 — connect + one-click export to the user's account

- Settings page (`/settings` or `/home`): "Connect Onshape" button →
  `/api/onshape/connect`; shows "Connected as <name> · Disconnect"
  when linked. Gate behind `plan == "pro"` (developer accounts exempt).
- Job result UI: **"Export to Onshape"** → server runs `_live_export`
  with the user's client, stores did/wid/eid + url on the `MeshJob`,
  returns the url → **"Open in Onshape"** (new tab for now).
- Per-user OAuth also removes the public-docs-only limit for users with
  a paid Onshape plan (that limit is the SERVER account being free).

Done = a logged-in Pro user converts a mesh and lands the parametric
model in their own Onshape in one click.

---

## 6. Phase 2 — linked docs, re-sync, embed (the "switch back and forth")

- **Idempotent re-export**: if the job already has a linked doc and it
  still exists in the user's account, UPDATE it rather than create a
  new one. Options, simplest first: (a) delete the Part Studio's
  features and rebuild (clean, loses user edits); (b) create a new
  VERSION then rebuild on a branch (preserves history); (c) diff and
  patch (hard). Start with (a) + a warning, move to (b).
- **Embed**: render the linked doc inline in Paramesh via Onshape's
  embedded-view / share-link iframe, so "switching" is a tab, not a
  context loss. "Open in Onshape" / "Back to Paramesh" cross-links
  (the Paramesh job URL is the back-link).
- Surface the link on the job in history so users return to the same
  doc later.

---

## 7. Phase 3 — import from Onshape + in-Onshape launch (full loop)

**Import (Paramesh ← Onshape):**
- "Import from Onshape" → list the user's documents
  (`/documents`), pick doc → element → pull geometry. Easiest source is
  tessellation/STL (`GET /partstudios/.../stl` or the translation API to
  STEP). Feed the mesh into the existing pipeline as input — this is the
  big win for SCANS (clean/re-parametrize a part that already lives in
  Onshape).

**Launch from inside Onshape (App Store extensions):**
- **Export action** on a Part Studio/Assembly → Onshape opens our URL
  with `documentId/workspaceId(or versionId)/elementId` as params + the
  authorized session → Paramesh pulls that element (import flow above)
  → converts → can push back to the same doc.
- **Right-panel tab** → Paramesh embedded in an iframe inside Onshape
  for the in-context experience.
- iframe app must complete OAuth from within the frame and read the doc
  context from the launch URL.

**Store listing** (after the above work): submit for Onshape review
with description/screenshots/support URL. Install is free; USE requires
a Paramesh Pro account (Stripe), checked server-side. (Alternative:
bill through Onshape's store with `OAuth2Purchase` — deferred; keep
Stripe to avoid a second billing system.)

---

## 8. Security & edge cases

- Encrypt tokens at rest (Fernet, dedicated `ONSHAPE_TOKEN_KEY`).
- Signed/expiring `state` on the OAuth round-trip (CSRF).
- Least-privilege scopes; only request Delete if used.
- Refresh-on-401 with token rotation persisted; graceful "reconnect"
  on refresh failure.
- Uninstall webhook → purge tokens.
- Free Onshape accounts can only create PUBLIC documents (Onshape
  limitation; surface it: "your Onshape plan only allows public docs").
- Multi-workspace / version vs workspace: store the workspace we wrote
  to; handle the doc being deleted out from under us (re-create).
- Rate limits: Onshape throttles; the per-feature `rate_limit_delay`
  already exists — keep it modest and back off on 429.

---

## 9. File-by-file work breakdown

- `db/database.py` — User + MeshJob columns; `_migrate_db` entries.
- `onshape/client.py` — `from_oauth_token`; Bearer header path.
- `onshape/oauth.py` (new) — authorize-URL build, code exchange, refresh,
  revoke, `get_onshape_client_for_user`.
- `api/onshape_oauth.py` (new) — connect/callback/disconnect/status routes.
- `pipeline/export_onshape.py` — `_live_export(..., client=None)`; persist
  did/wid/eid back onto the job.
- `viz_server/app.py` — `/api/export-onshape/{job}` (per-user),
  `/api/import-onshape` (Phase 3); wire the OAuth router; Pro gating.
- `viz_server/static/` — Connect button + status, "Export/Open in
  Onshape" on results, embed iframe (Phase 2), import browser (Phase 3).
- `config.py` — OAuth client id/secret/redirect, token key.
- `tests/` — OAuth state/exchange (mocked), client Bearer path, job-link
  persistence, gating; live tests stay behind `ONSHAPE_LIVE_TESTS=1`.

---

## 10. Suggested sequencing

1. App registration + `.env` secrets + data-model migrations.
2. OAuth flow + client Bearer path + `get_onshape_client_for_user`
   (verify with a throwaway script like the existing `onshape_*` probes).
3. Phase 1 UI (connect + export + open). Ship to Pro users.
4. Phase 2 (re-sync + embed).
5. Phase 3 import + App Store extensions, then submit for review.

Reuse the probe-script pattern (`scripts/onshape_*.py`) to validate each
live behavior before wiring UI.
