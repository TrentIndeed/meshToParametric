# Onshape Integration Plan — seamless round-trip

Decisions locked (2026-06-13): **OAuth 2.0** per-user auth; **full
bidirectional loop** (export, re-sync, import); **goal: a live
incremental build a user can watch inside Onshape**; design for the
Onshape App Store but as an **async track that does NOT gate launch**.
Pro-tier feature. Billing stays in Stripe; Onshape is the integration
surface, not the biller.

**Two delivery tracks (read this first):**
1. **Core OAuth product — store-INDEPENDENT, this is the launch path
   and the DEFAULT exporter.** A registered OAuth app works for our own
   users immediately, with no Store review. Connect → export to the
   user's account → open/import/re-sync all work from parameshai.com.
   The live incremental build is achievable here too (paced feature
   adds into a doc the user keeps open in a second Onshape tab; their
   viewport updates live via Onshape's collaborative sync).
2. **App Store listing — SEPARATE, async, enhances but never blocks.**
   Store review takes an unknown amount of time, so the in-Onshape
   embedded panel (Paramesh *inside* Onshape) and the in-Onshape launch
   action are a later upgrade, NOT the default exporter. Everything in
   track 1 must stand on its own without it.

Build status (2026-06-13): the OAuth FOUNDATION is shipped — per-user
token storage (encrypted), `OnshapeClient` Bearer path,
connect/callback/disconnect/status routes, and
`POST /api/onshape/export/{job_uuid}` (export a finished job to the
user's own account, link the doc). The live export engine
(`feature_builder.py` / `_live_export`) was already proven on
RackMount. Remaining: the UI (connect + export buttons), then re-sync,
live incremental build, import, and the Store track.

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

## 6. Phase 2 — linked docs, re-sync, LIVE BUILD, embed (store-independent)

All of this works on track 1 (no Store listing needed).

- **Idempotent re-export**: if the job already has a linked doc and it
  still exists in the user's account, UPDATE it rather than create a
  new one. Options, simplest first: (a) delete the Part Studio's
  features and rebuild (clean, loses user edits); (b) create a new
  VERSION then rebuild on a branch (preserves history); (c) diff and
  patch (hard). Start with (a) + a warning, move to (b).
- **Live incremental build the user can watch in Onshape (goal).** Why
  it works: we already add features one at a time (base → joins →
  holes → each fillet). Each POST creates a new microversion, and
  Onshape's real-time collaborative sync pushes every change to ALL
  open sessions of that document — so a part studio the user has open
  updates feature-by-feature, live, as we build. Make it watchable:
    - an **incremental, paced export mode** (space the feature POSTs;
      reuse `ExportConfig.rate_limit_delay`) targeting an EXISTING
      document/element instead of only create-new-doc-all-at-once;
    - emit per-feature progress so the Paramesh side animates in step
      with the Onshape viewport (the "two synced views" effect);
    - likely flip the OAuth app's **"Supports collaboration"** ON (it
      was left off in registration) — verify the live-propagation
      latency with a throwaway probe before relying on it.
  Pre-Store form: the user keeps the doc open in a second Onshape tab
  and watches it build there. Post-Store form (track 2): the same build
  shown inside the embedded Paramesh panel + Onshape's main viewport
  side by side. NOTE: our own 3D animation lives in the Paramesh
  panel/page — Onshape can't play it; the magic is the synced native
  viewport, which is exactly the bar competitors (e.g. Backflip) hit.
- **Embed Onshape IN Paramesh** (store-independent): render the linked
  doc inline in our app via Onshape's embedded-view / share-link
  iframe, so switching is a tab, not a context loss. Cross-links "Open
  in Onshape" / "Back to Paramesh" (the job URL is the back-link).
- Surface the link on the job in history so users return to the same
  doc later.

---

## 7. Phase 3a — import from Onshape (store-independent, full loop)

- "Import from Onshape" → list the user's documents (`/documents`),
  pick doc → element → pull geometry. Easiest source is tessellation/STL
  (`GET /partstudios/.../stl` or the translation API to STEP). Feed the
  mesh into the existing pipeline as input — the big win for SCANS
  (clean/re-parametrize a part that already lives in Onshape). Needs no
  Store listing; it's just authenticated API calls.

## 7b. App Store track — Paramesh INSIDE Onshape (async, NOT the default)

Separate, parallel, does not block launch. Adds the "mini window inside
Onshape" + launch-from-Onshape experience on top of the working track-1
product.

- **Export/convert action** on a Part Studio/Assembly → Onshape opens
  our URL with `documentId/workspaceId(or versionId)/elementId` + the
  authorized session → Paramesh pulls that element (import flow) →
  converts → pushes back to the same doc (with the live build above).
- **Right-panel tab** → Paramesh embedded in an iframe inside Onshape;
  the panel completes OAuth from within the frame and reads the doc
  context from the launch URL. This is the literal "Paramesh in a mini
  window inside Onshape" the user asked about.
- **Store listing**: submit for Onshape review (description /
  screenshots / support URL). Install free; USE requires a Paramesh Pro
  account (Stripe), checked server-side. (Billing-through-Onshape via
  `OAuth2Purchase` is deferred — keep Stripe.) Because review time is
  unknown, NONE of track 1 may depend on this being live.

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

Track 1 (store-independent — this is the launch product):
1. [x] App registration + `.env` secrets + data-model migrations.
2. [x] OAuth flow + client Bearer path + `get_onshape_client_for_user`
   + `POST /api/onshape/export/{job}`. (Done 2026-06-13.)
3. Phase 1 UI: Connect button (status) + "Export to Onshape" / "Open in
   Onshape" on results. Ship to Pro users — this is the DEFAULT exporter.
4. Phase 2: idempotent re-sync; live incremental/paced build into an
   open doc (second tab); embed Onshape in Paramesh.
5. Phase 3a: import from Onshape.

Track 2 (App Store — parallel, async, never blocks track 1):
6. Register the Store extensions (right-panel tab + convert action),
   move OAuth into the embedded frame, wire the launch context, port the
   live build into the embedded panel.
7. Submit for review; ship when approved.

Reuse the probe-script pattern (`scripts/onshape_*.py`) to validate each
live behavior before wiring UI.
