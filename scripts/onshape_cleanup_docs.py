"""Delete intermediate mesh2param test documents, keeping the ones
named on the command line (by document id)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mesh2param.onshape.client import OnshapeClient  # noqa: E402


def main() -> int:
    keep = set(sys.argv[1:])
    c = OnshapeClient.from_env()
    docs = c.get("/documents", params={"filter": 0, "limit": 20})
    for d in docs.get("items", []):
        name = d.get("name", "")
        if not name.startswith("mesh2param"):
            continue
        if d["id"] in keep:
            print(f"keep   {d['id']} {name}")
            continue
        c.delete_document(d["id"])
        print(f"delete {d['id']} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
