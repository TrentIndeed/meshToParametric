"""FastAPI server serving a Three.js viewer for mesh visualization."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="mesh2param Viewer")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(model: str | None = None):
    """Serve the Three.js viewer page."""
    return FileResponse(str(STATIC_DIR / "viewer.html"))


def run(host: str = "127.0.0.1", port: int = 8080):
    """Run the viewer server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
