"""Thin Onshape REST client (API key pair, HTTP Basic auth).

Onshape API keys work with plain Basic authentication — no HMAC
request signing needed. Credentials come from ONSHAPE_ACCESS_KEY /
ONSHAPE_SECRET_KEY environment variables, with a fallback that reads
the repo's .env file (matching how the rest of the app is configured).

Endpoints wrapped here are exactly the set the live export needs:
documents, elements, part-studio features, mass properties, bounding
boxes. Everything returns parsed JSON; HTTP errors raise
OnshapeApiError with the response body attached (Onshape's error
messages — e.g. "Parameter offset ... does not match its feature
spec" — are the only way to debug a rejected feature).
"""
from __future__ import annotations

import base64
import logging
import os
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://cad.onshape.com"
API_VERSION = "v6"


class OnshapeApiError(RuntimeError):
    """An Onshape API call failed; carries status code and body."""

    def __init__(self, status: int, body: str, url: str):
        super().__init__(f"Onshape API {status} on {url}: {body[:500]}")
        self.status = status
        self.body = body
        self.url = url


def _load_dotenv_fallback() -> None:
    """Populate os.environ from a .env at the repo root (no override)."""
    here = Path(__file__).resolve()
    for parent in (Path.cwd(), *here.parents):
        env = parent / ".env"
        if env.exists():
            try:
                for line in env.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, _, v = line.partition("=")
                        os.environ.setdefault(k.strip(), v.strip())
            except Exception:
                logger.debug("Failed reading %s", env, exc_info=True)
            return


class OnshapeClient:
    """Minimal Onshape REST API wrapper used by the live export."""

    def __init__(
        self,
        access_key: str,
        secret_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api = f"{self.base_url}/api/{API_VERSION}"
        self.timeout = timeout
        self._session = requests.Session()
        token = base64.b64encode(
            f"{access_key}:{secret_key}".encode()
        ).decode()
        self._session.headers.update({
            "Authorization": f"Basic {token}",
            "Accept": "application/json;charset=UTF-8; qs=0.09",
            "Content-Type": "application/json",
        })

    # -- construction -----------------------------------------------------

    @classmethod
    def from_env(cls, base_url: str = DEFAULT_BASE_URL,
                 timeout: float = 30.0) -> "OnshapeClient | None":
        """Build a client from env vars (with .env fallback); None if
        no credentials are configured."""
        if not (os.environ.get("ONSHAPE_ACCESS_KEY")
                and os.environ.get("ONSHAPE_SECRET_KEY")):
            _load_dotenv_fallback()
        ak = os.environ.get("ONSHAPE_ACCESS_KEY", "").strip()
        sk = os.environ.get("ONSHAPE_SECRET_KEY", "").strip()
        if not ak or not sk:
            return None
        return cls(ak, sk, base_url=base_url, timeout=timeout)

    # -- plumbing ----------------------------------------------------------

    def _request(self, method: str, path: str, **kw):
        url = f"{self.api}{path}"
        r = self._session.request(method, url, timeout=self.timeout, **kw)
        if r.status_code >= 400:
            raise OnshapeApiError(r.status_code, r.text, url)
        if not r.text:
            return {}
        try:
            return r.json()
        except ValueError:
            return {"raw": r.text}

    def get(self, path: str, **kw):
        return self._request("GET", path, **kw)

    def post(self, path: str, **kw):
        return self._request("POST", path, **kw)

    def delete(self, path: str, **kw):
        return self._request("DELETE", path, **kw)

    # -- API surface --------------------------------------------------------

    def sessioninfo(self) -> dict:
        return self.get("/users/sessioninfo")

    def create_document(self, name: str) -> tuple[str, str]:
        """Create a document; returns (document_id, workspace_id).

        Tries private first; free-tier accounts can only create PUBLIC
        documents (private creation returns 409), so falls back.
        """
        try:
            doc = self.post("/documents", json={
                "name": name, "isPublic": False,
            })
        except OnshapeApiError as exc:
            if exc.status != 409:
                raise
            logger.info(
                "Private document creation refused (free tier); "
                "creating PUBLIC document %r", name,
            )
            doc = self.post("/documents", json={
                "name": name, "isPublic": True,
            })
        return doc["id"], doc["defaultWorkspace"]["id"]

    def delete_document(self, did: str) -> None:
        self.delete(f"/documents/{did}")

    def elements(self, did: str, wid: str) -> list[dict]:
        return self.get(f"/documents/d/{did}/w/{wid}/elements")

    def partstudio_id(self, did: str, wid: str) -> str:
        for e in self.elements(did, wid):
            if e.get("elementType") == "PARTSTUDIO":
                return e["id"]
        raise OnshapeApiError(404, "no PARTSTUDIO element", did)

    def add_feature(self, did: str, wid: str, eid: str,
                    feature: dict) -> tuple[str, str]:
        """POST one feature; returns (feature_id, feature_status)."""
        d = self.post(
            f"/partstudios/d/{did}/w/{wid}/e/{eid}/features",
            json={"feature": feature},
        )
        f = d.get("feature") or {}
        fid = f.get("featureId") or (f.get("message") or {}).get("featureId")
        status = (d.get("featureState") or {}).get("featureStatus", "?")
        return fid or "", status

    def features(self, did: str, wid: str, eid: str) -> dict:
        return self.get(f"/partstudios/d/{did}/w/{wid}/e/{eid}/features")

    def delete_feature(self, did: str, wid: str, eid: str,
                       fid: str) -> None:
        self.delete(
            f"/partstudios/d/{did}/w/{wid}/e/{eid}/features/featureid/{fid}",
        )

    def volume_mm3(self, did: str, wid: str, eid: str) -> float | None:
        """Combined solid volume in mm^3 (None if no parts yet)."""
        try:
            d = self.get(
                f"/partstudios/d/{did}/w/{wid}/e/{eid}/massproperties",
                params={"massAsComputed": "true"},
            )
            return float(d["bodies"]["-all-"]["volume"][0]) * 1e9
        except (OnshapeApiError, KeyError, IndexError, TypeError, ValueError):
            return None

    def bounding_box_mm(self, did: str, wid: str,
                        eid: str) -> tuple[list, list] | None:
        try:
            b = self.get(
                f"/partstudios/d/{did}/w/{wid}/e/{eid}/boundingboxes",
            )
            lo = [b["lowX"] * 1000, b["lowY"] * 1000, b["lowZ"] * 1000]
            hi = [b["highX"] * 1000, b["highY"] * 1000, b["highZ"] * 1000]
            return lo, hi
        except (OnshapeApiError, KeyError, TypeError):
            return None

    def document_url(self, did: str, wid: str, eid: str | None = None) -> str:
        url = f"{self.base_url}/documents/{did}/w/{wid}"
        return f"{url}/e/{eid}" if eid else url
