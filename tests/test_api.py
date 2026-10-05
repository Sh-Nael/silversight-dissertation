"""Web API tests (Track U): endpoints answer correctly and the frontend is served safely."""

from __future__ import annotations

from fastapi.testclient import TestClient

from xaglab.api import create_app


def test_health_and_status():
    c = TestClient(create_app())
    assert c.get("/api/health").json() == {"ok": True}
    s = c.get("/api/status").json()
    assert s["ok"] is True
    names = {x["name"]: x for x in s["snapshots"]}
    assert names["dissertation_v1"]["verified"] and names["dissertation_v1"]["files"] == 7
    assert names["calendar_v1"]["verified"] and names["calendar_v1"]["snapshot_end"]
    assert len(s["code"]["commit"]) >= 7 and s["code"]["version"]


def test_unknown_api_path_is_a_json_404_not_the_frontend():
    r = TestClient(create_app()).get("/api/does-not-exist")
    assert r.status_code == 404 and "unknown API endpoint" in r.json()["detail"]


def test_without_a_build_the_root_explains_what_to_do(tmp_path):
    r = TestClient(create_app(dist=tmp_path / "missing")).get("/")
    assert r.status_code == 200 and "npm run build" in r.text


def test_single_page_app_fallback_and_no_path_escape(tmp_path):
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<div id=root>INDEX</div>")
    (dist / "favicon.svg").write_text("<svg/>")
    (tmp_path / "secret.txt").write_text("SECRET")
    c = TestClient(create_app(dist=dist))
    assert "INDEX" in c.get("/compare").text            # client-side route → index.html
    assert c.get("/favicon.svg").text == "<svg/>"        # real build files are served
    assert "SECRET" not in c.get("/../secret.txt").text  # cannot escape the build folder
    assert "SECRET" not in c.get("/%2e%2e/secret.txt").text


# ------------------------------------------------------------------ API v1 (U1.3)
import json

import pytest

from xaglab.eval.folds import make_calendar
from xaglab.eval.harness import ModelSpec, run, save_run
from xaglab.models.baselines import EWMA, Climatology
from xaglab.paths import EXPERIMENTS_DIR


@pytest.fixture(scope="module")
def client(features, tmp_path_factory):
    """An app whose runs folder holds two fresh, fast runs (independent of local runs)."""
    base = tmp_path_factory.mktemp("runs")
    cal = make_calendar(features.index, features.labels, train_start=features.warmup_end())
    ids = {}
    for spec in (ModelSpec(Climatology, {}, "climatology"), ModelSpec(EWMA, {"lam": 0.94}, "ewma")):
        res = run(spec, features, cal, n_jobs=4, save=False)
        save_run(res, base)
        ids[spec.name] = res.run_id
    c = TestClient(create_app(runs_dir=base))
    c.ids = ids
    return c


def test_runs_list_detail_and_predictions(client):
    runs = client.get("/api/runs").json()
    assert {r["model"] for r in runs} == {"climatology", "ewma"}
    clim = next(r for r in runs if r["model"] == "climatology")
    assert clim["tag"] is None and clim["n_refits"] == 200
    assert 0.24 < clim["headline"]["brier1"] < 0.26
    d = client.get(f"/api/runs/{clim['id']}").json()
    assert len(d["timings"]) == 200 and d["tuning"] is None and d["manifest"]["model"] == "climatology"
    p = client.get(f"/api/runs/{clim['id']}/predictions").json()
    assert len(p["date"]) == len(p["p_up1"]) == len(p["up1"]) == 4141


def test_unknown_or_malicious_run_ids_are_404(client):
    for bad in ["nope-20260101-000000", "..", "..%2Fsecret", "%2Fetc"]:
        assert client.get(f"/api/runs/{bad}").status_code == 404


def test_calibration_and_volatility(client):
    rid = client.ids["ewma"]
    bins = client.get(f"/api/runs/{rid}/calibration?h=1&bins=10").json()
    assert sum(b["n"] for b in bins) == 4141
    vol = client.get(f"/api/runs/{rid}/volatility?h=5").json()
    assert set(vol) == {"date", "forecast", "realised"} and len(vol["date"]) > 800
    assert client.get(f"/api/runs/{rid}/volatility?freq=bogus").status_code == 422
    sel = client.get(f"/api/runs/{rid}/selective?h=1&mode=ranked").json()
    assert [r["coverage_target"] for r in sel] == [0.1, 0.2, 0.3, 0.5, 1.0] and sel[-1]["n"] == 4141
    assert client.get(f"/api/runs/{rid}/selective?mode=bogus").status_code == 422
    sig = client.get(f"/api/runs/{rid}/signals?h=1").json()
    assert len(sig) == 15 and {r["side"] for r in sig} == {"both", "buy", "sell"}   # 5 coverages x 3 sides
    eq = client.get(f"/api/runs/{rid}/equity?h=5&coverage=0.3&side=buy").json()
    assert set(eq) == {"date", "equity", "drawdown"} and len(eq["date"]) > 800
    assert client.get(f"/api/runs/{rid}/equity?side=short").status_code == 422
    assert client.get(f"/api/runs/{rid}/edges").status_code == 422      # a baseline selects no edges


def test_compare(client):
    body = {"runs": [client.ids["climatology"], client.ids["ewma"]], "reference": "climatology"}
    r = client.post("/api/compare", json=body).json()
    assert r["models"] == ["climatology", "ewma"] and len(r["scoreboard"]) == 4
    sig = {(s["loss"], s["h"]): s for s in r["significance"]}
    assert sig[("brier", 1)]["wilcoxon_p"] == 1.0          # ewma direction = climatology
    assert sig[("qlike", 1)]["mean_diff"] < 0               # EWMA volatility beats climatology
    assert len(r["per_fold"]["qlike_h1"]["ewma"]) == 8
    assert set(r["cumulative"]["qlike_h1"]) == {"date", "ewma"}
    bad = client.post("/api/compare", json={**body, "reference": "gbm"})
    assert bad.status_code == 422
    dup = client.post("/api/compare", json={**body, "runs": [client.ids["ewma"]] * 2, "reference": "ewma"})
    assert dup.status_code == 422


def test_folds_endpoint_matches_the_frozen_calendar(client):
    f = client.get("/api/folds").json()
    assert f["calendar"] == json.loads((EXPERIMENTS_DIR / "folds_v1.json").read_text())
    assert len(f["refits"]) == 200 and len(f["tuning_points"]) == 17
    assert f["protocol"]["purge"] == 5


def test_data_endpoints(client):
    nodes = {n["key"]: n for n in client.get("/api/data/nodes").json()}
    assert len(nodes) == 6 and len(nodes["silver"]["features"]) == 19 and not nodes["vix"]["has_volume"]
    w = client.get("/api/data/panel?node=silver&freq=W-FRI").json()
    assert {"date", "close", "bar_ok", "filled"} <= set(w) and 1000 < len(w["date"]) < 1100
    assert client.get("/api/data/panel?node=platinum").status_code == 404
    assert client.get("/api/data/panel?node=silver&freq=hourly").status_code == 422
    f = client.get("/api/data/features?node=silver&names=r1,gsr_ect").json()
    assert set(f) == {"date", "r1", "gsr_ect"} and len(f["date"]) == 5154
    assert client.get("/api/data/features?node=silver&names=nope").status_code == 404
    assert "fomc_days_to" in client.get("/api/data/features?node=events").json()
    assert len(client.get("/api/data/events").json()) == 709
    assert len(client.get("/api/data/regimes").json()) == 5
    assert len(client.get("/api/data/quality").json()) == 6
