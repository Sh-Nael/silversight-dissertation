"""FastAPI application factory.

Two jobs:
  1. JSON endpoints under ``/api`` (see the routers imported below).
  2. Serving the compiled React frontend from ``web/dist`` for everything else, with a
     single-page-app fallback: any non-API path returns ``index.html`` so client-side
     routes such as ``/compare`` work on reload. Viewers therefore need Python only.

In development the React dev server (Vite, port 5173) serves the frontend with hot
reload and proxies ``/api`` to this app on port 8000.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from xaglab import __version__
from xaglab.api.routes import compare, data, folds, runs, status
from xaglab.paths import RUNS_DIR, WEB_DIST

_NO_BUILD = """<!doctype html><meta charset="utf-8"><title>SilverSight</title>
<body style="font:15px/1.6 system-ui;background:#0D1117;color:#E6EDF3;padding:40px;max-width:70ch">
<h1 style="font-weight:600">SilverSight API is running</h1>
<p>The web frontend has not been built yet, so there is nothing to show here.</p>
<p>Developers: <code>cd web && npm install && npm run build</code>, then reload this page,
or run the dev server with <code>npm run dev</code> and open <code>http://localhost:5173</code>.</p>
<p>The API itself is live: <a style="color:#4CC2D6" href="/api/status">/api/status</a> ·
<a style="color:#4CC2D6" href="/docs">/docs</a> (interactive API documentation).</p></body>"""


def create_app(dist: Path = WEB_DIST, runs_dir: Path = RUNS_DIR) -> FastAPI:
    """Build the app. `dist` is the built frontend; `runs_dir` the folder of run results
    (tests point both at temporary folders)."""
    app = FastAPI(
        title="SilverSight API",
        version=__version__,
        description="Read-only access to SilverSight data, folds and runs (xaglab).",
    )
    app.state.runs_dir = runs_dir
    for module in (status, runs, compare, folds, data):
        app.include_router(module.router, prefix="/api")

    @app.get("/api", include_in_schema=False)
    @app.get("/api/{rest:path}", include_in_schema=False)
    def api_not_found(rest: str = "") -> JSONResponse:
        # Anything under /api that no router handled is an API error, never the frontend.
        return JSONResponse({"detail": f"unknown API endpoint: /api/{rest}"}, status_code=404)

    index = dist / "index.html"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        target = (dist / path).resolve()
        # Real files in the build (favicon, images) are served as they are; guard
        # against paths escaping the build folder.
        if path and target.is_file() and dist.resolve() in target.parents:
            return FileResponse(target)
        if index.is_file():
            return FileResponse(index)
        return HTMLResponse(_NO_BUILD)

    return app
