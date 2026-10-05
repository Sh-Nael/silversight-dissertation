"""Command line: ``python -m xaglab.cli <command>`` (or ``xag <command>`` once installed).

  verify                     check every snapshot's checksums
  folds                      print the frozen fold calendar
  run MODEL [MODEL ...]      walk-forward run(s); results in experiments/runs/<run_id>/
  report RUN [RUN ...]       scoreboard + significance tables for finished runs
  tables                     the table pack for the report, from the committed report files
  ui                         start the web app on http://localhost:8000
  models                     list runnable model names
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


def _data():
    from xaglab.data.panel import load_default_panel
    from xaglab.eval.folds import make_calendar
    from xaglab.features.build import build_features

    fs = build_features(load_default_panel())
    return fs, make_calendar(fs.index, fs.labels, train_start=fs.warmup_end())


@app.command()
def verify() -> None:
    """Verify every snapshot against its manifest."""
    from xaglab.data.store import Snapshot
    from xaglab.paths import SNAPSHOT_DIR

    for m in sorted(SNAPSHOT_DIR.glob("*/MANIFEST.json")):
        snap = Snapshot.open(m.parent.name)
        console.print(f"[green]ok[/] {snap.name}: {len(snap.files)} files verified")


@app.command()
def folds() -> None:
    """Show the frozen walk-forward calendar."""
    _, cal = _data()
    console.print_json(json.dumps(cal.describe()))


@app.command()
def models() -> None:
    """List runnable models."""
    from xaglab.models import registry

    console.print(", ".join(registry.names()))


@app.command()
def run(
    names: list[str],
    jobs: int = typer.Option(16, help="parallel processes for stateless models"),
    every: int = typer.Option(0, help="refit interval in origins (0 = calendar default 21)"),
    seed: int = typer.Option(0),
    tag: str = typer.Option("", help="suffix for the run id"),
    tuning_from: list[str] = typer.Option(  # noqa: B008
        [], help="development run whose tuned settings to reuse (one per tuned model)"),
) -> None:
    """Run one or more models through the full walk-forward protocol."""
    from xaglab.eval.harness import load_dev
    from xaglab.eval.harness import run as run_model
    from xaglab.eval.metrics import summarise
    from xaglab.models import registry

    dev = {load_dev(d).manifest["model"]: d for d in tuning_from}
    unused = set(dev) - set(names)
    if unused:
        raise typer.BadParameter(f"--tuning-from given for models not being run: {sorted(unused)}")
    fs, cal = _data()
    for name in names:
        res = run_model(registry.get(name), fs, cal, every=every or None, seed=seed,
                        n_jobs=jobs, tag=tag, tuning_from=dev.get(name))
        s = summarise(res.predictions)[["h", "brier", "logloss", "accuracy", "auc", "vol_qlike"]]
        console.print(f"[bold]{res.run_id}[/]  wall {res.manifest['wall_seconds']}s  -> {res.path}")
        console.print(s.round(4).to_string(index=False))


@app.command()
def dev(
    names: list[str],
    jobs: int = typer.Option(6, help="parallel processes (tuning points)"),
    seed: int = typer.Option(0),
    tag: str = typer.Option("", help="suffix for the run id"),
    points: str = typer.Option("", help="tuning-point numbers to run, e.g. 2,5,8,11,14 (a screen)"),
) -> None:
    """Development run: the tuning stage only (no test row touched) -> experiments/dev/."""
    from xaglab.eval.harness import develop
    from xaglab.eval.report import dev_scoreboard
    from xaglab.models import registry

    fs, cal = _data()
    for name in names:
        only = [int(p) for p in points.split(",")] if points else None
        res = develop(registry.get(name), fs, cal, seed=seed, n_jobs=jobs, tag=tag, points=only)
        console.print(f"[bold]{res.run_id}[/]  wall {res.manifest['wall_seconds']}s  -> {res.path}")
        console.print(dev_scoreboard([res]).round(4).to_string(index=False))


@app.command()
def devreport(
    runs: list[str],
    reference: str = typer.Option("linear", help="model the others are paired against"),
    out: Path | None = typer.Option(None, help="write the tables as CSV into this folder"),  # noqa: B008
) -> None:
    """Development scores of development runs and/or full runs, side by side."""
    from xaglab.eval.report import dev_paired, dev_scoreboard

    loaded = [_load_any(r) for r in runs]
    names = [r.manifest["model"] for r in loaded]
    for r in loaded:  # two runs of one model: label them by run id without the timestamp
        if names.count(r.manifest["model"]) > 1:
            r.manifest["label"] = r.run_id[: -len("-YYYYMMDD-HHMMSS")]
    board = dev_scoreboard(loaded)
    wide = board.pivot(index="task", columns="model", values="mean")
    console.print("[bold]Development scores[/] (mean out-of-fold loss over tuning points; lower is better)")
    console.print(wide.round(4).to_string())
    paired = dev_paired(loaded, reference)
    if len(paired):
        console.print(f"\n[bold]Paired with {reference}[/] (difference < 0 = better; points won of compared)")
        console.print(paired.round(4).to_string(index=False))
    if out:
        out.mkdir(parents=True, exist_ok=True)
        board.to_csv(out / "dev_scoreboard.csv", index=False)
        if len(paired):
            paired.to_csv(out / "dev_paired.csv", index=False)


def _load_any(run_id: str):
    """A development run (experiments/dev) or a full run (experiments/runs), by id or path."""
    from xaglab.eval.harness import load_dev, load_run
    from xaglab.paths import DEV_DIR, RUNS_DIR

    path = Path(run_id)
    if not path.exists():
        path = DEV_DIR / run_id if (DEV_DIR / run_id).exists() else RUNS_DIR / run_id
    return load_run(path) if (path / "predictions.parquet").exists() else load_dev(path)


@app.command()
def edges(
    run: str,
    out: Path | None = typer.Option(None, help="write the tables as CSV into this folder"),  # noqa: B008
) -> None:
    """Edge dynamics of an AIM-DG run: persistence and the regime tests (Phase 6.3, D-40)."""
    from xaglab.eval import edges as E
    from xaglab.eval.harness import load_run

    tl = E.timeline(load_run(run))
    per, overall = E.persistence(tl)
    tests = E.regime_tests(tl)
    console.print(f"[bold]{run}[/]: {overall['months']} months, {overall['mean_edges']:.2f} edges active on average, "
                  f"{overall['mean_edges_changed']:.2f} changed per month, Jaccard {overall['mean_jaccard']:.2f}, "
                  f"{overall['months_unchanged']} months unchanged")
    console.print(per.round(3).to_string(index=False))
    console.print("\n[bold]Stress months against the rest[/] (one-sided circular-shift test; primary)")
    console.print(tests[tests.primary].drop(columns="primary").round(4).to_string(index=False))
    console.print("\n[bold]Per regime[/] (descriptive)")
    console.print(tests[~tests.primary].drop(columns="primary").round(4).to_string(index=False))
    if out:
        out.mkdir(parents=True, exist_ok=True)
        tl.to_csv(out / "timeline.csv")
        per.to_csv(out / "persistence.csv", index=False)
        tests.to_csv(out / "regime_tests.csv", index=False)


@app.command()
def ensemble(
    runs: list[str],
    name: str = typer.Option("ensemble", help="model name of the new run (e.g. ens_lin_gbm)"),
) -> None:
    """Equal-weight ensemble of finished runs, saved as a run of its own (experiments/runs/)."""
    from xaglab.eval.ensemble import ensemble as make
    from xaglab.eval.harness import load_run, save_run
    from xaglab.eval.metrics import summarise

    res = make([load_run(r) for r in runs], name)
    path = save_run(res)
    s = summarise(res.predictions)[["h", "brier", "logloss", "accuracy", "auc", "vol_qlike"]]
    console.print(f"[bold]{res.run_id}[/]  members {runs}  -> {path}")
    console.print(s.round(4).to_string(index=False))


@app.command()
def signals(
    runs: list[str],
    cost_bps: float = typer.Option(10.0, help="round-trip trading cost in basis points"),
    out: Path | None = typer.Option(None, help="write the tables as CSV into this folder"),  # noqa: B008
) -> None:
    """Signal and risk layer: BUY/SELL on confident days with a 1σ stop and a 1.5σ target,
    backtested on daily bars; buy and sell trades reported separately."""
    from xaglab.data.panel import load_default_panel
    from xaglab.eval.harness import load_run
    from xaglab.eval.signals import Rules, evaluate

    bars = load_default_panel().nodes["silver"][["open", "high", "low", "close"]]
    rules = Rules(cost_bps=cost_bps)
    tables = []
    for r in runs:
        run = load_run(r)
        for h in (1, 5):
            t = evaluate(run.predictions, bars, h, rules=rules).assign(model=run.manifest["model"], h=h)
            tables.append(t)
    all_t = pd.concat(tables, ignore_index=True)
    cols = ["model", "h", "coverage_target", "side", "n", "win_rate", "profit_factor", "mean_net_bps",
            "p_positive", "target_rate", "stop_rate", "total_return", "max_drawdown"]
    console.print(f"[bold]Signal and risk layer[/] (stop 1σ, target 1.5σ, {cost_bps:g} bps round trip, "
                  "1% risk per trade, one position at a time; p_positive: mean net return > 0, one-sided)")
    for h in (1, 5):
        console.print(f"[bold]h={h}[/]")
        console.print(all_t[all_t.h == h][cols].round(4).to_string(index=False))
    if out:
        out.mkdir(parents=True, exist_ok=True)
        all_t.to_csv(out / "signals.csv", index=False)


@app.command()
def report(
    runs: list[str],
    reference: str = typer.Option("climatology", help="model every other model is tested against"),
    out: Path | None = typer.Option(None, help="write the tables as CSV into this folder"),  # noqa: B008
) -> None:
    """Scoreboard and significance tables for finished runs."""
    from xaglab.eval.harness import load_run
    from xaglab.eval.report import dev_scoreboard, scoreboard, selective, significance

    loaded = [load_run(r) for r in runs]
    board = scoreboard(loaded)
    cols = ["model", "h", "brier", "logloss", "accuracy", "balanced_accuracy", "auc",
            "mean_p", "up_calls", "up_precision", "down_precision", "vol_qlike", "vol_rmse_vol", "vol_mz_r2"]
    console.print(board[cols].round(4).to_string(index=False))
    dev = dev_scoreboard(loaded)
    if len(dev):
        console.print("\n[bold]Development scores[/] (out-of-fold, tuning points; for design choices only)")
        console.print(dev.round(4).to_string(index=False))
    tables = [board]
    for loss, hs in (("brier", (1, 5)), ("logloss", (1, 5)), ("qlike", (1, 5))):
        for h in hs:
            sig = significance(loaded, reference, loss, h)
            if len(sig):
                console.print(f"\n[bold]{loss}, h={h}[/] vs {reference}")
                console.print(sig.drop(columns=["vs", "loss", "h"]).round(4).to_string(index=False))
                tables.append(sig)
    sel = []
    for r in loaded:
        for h in (1, 5):
            for mode in ("causal", "ranked"):
                t = selective(r, h, mode=mode)
                if len(t) and t["n"].sum():
                    sel.append(t.assign(model=r.manifest["model"], h=h, mode=mode))
    if sel:
        sel = pd.concat(sel, ignore_index=True)
        console.print("\n[bold]Accuracy vs coverage[/] (causal: confident = top share of the model's last 252 "
                      "forecasts; p_edge: one-sided, vs always-up on the same days)")
        cols = ["model", "coverage_target", "n", "accuracy", "always_up", "edge", "p_edge", "p_coin"]
        for h in (1, 5):
            console.print(f"[bold]h={h}[/]")
            console.print(sel[(sel.h == h) & (sel["mode"] == "causal")][cols].round(4).to_string(index=False))
        console.print("\n[bold]Two-sided scorecard[/] (causal; each side vs the stricter of 50% and its base "
                      "rate on the selected days; goal: both sides >= 65% and significant)")
        sides = ["model", "coverage_target", "up_n", "up_right", "p_up", "down_n", "down_right", "p_down"]
        for h in (1, 5):
            console.print(f"[bold]h={h}[/]")
            console.print(sel[(sel.h == h) & (sel["mode"] == "causal")][sides].round(4).to_string(index=False))
    if out:
        out.mkdir(parents=True, exist_ok=True)
        board.to_csv(out / "scoreboard.csv", index=False)
        if len(dev):
            dev.to_csv(out / "dev_scoreboard.csv", index=False)
        pd.concat(tables[1:]).to_csv(out / "significance.csv", index=False)
        if len(sel):
            sel.to_csv(out / "selective.csv", index=False)


@app.command()
def tables(
    reports: Path | None = typer.Option(None, help="folder with the report files (default experiments/reports)"),  # noqa: B008
    out: Path | None = typer.Option(None, help="output folder (default experiments/tables)"),  # noqa: B008
    docx: bool = typer.Option(True, help="also write tables.docx (Word tables)"),
) -> None:
    """The table pack for the report: formats the committed report files, computes nothing."""
    from xaglab.eval.tables import build, write_pack
    from xaglab.paths import REPORTS_DIR, TABLES_DIR

    pack = build(reports or REPORTS_DIR)
    written = write_pack(pack, out or TABLES_DIR, docx=docx)
    for t in pack:
        console.print(f"{t.key:<42} {len(t.frame):>3} rows  ch. {t.chapter:<9} {t.title}")
    console.print(f"[bold]{len(pack)} tables[/], {len(written)} files -> {out or TABLES_DIR}")


@app.command()
def ui(
    port: int = typer.Option(8000, help="port for the web app"),
    host: str = typer.Option("127.0.0.1", help="interface; the default keeps it on this machine"),
    reload: bool = typer.Option(False, help="restart on code changes (development)"),
    open_browser: bool = typer.Option(False, "--open", help="open the browser when ready"),
) -> None:
    """Start the SilverSight web app (API + built frontend) on http://host:port."""
    import threading
    import webbrowser

    import uvicorn

    url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
    console.print(f"SilverSight web app → [bold]{url}[/]   (API docs: {url}/docs, stop: Ctrl+C)")
    if open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("xaglab.api:create_app", factory=True, host=host, port=port, reload=reload,
                reload_dirs=["xaglab"] if reload else None, log_level="info")


if __name__ == "__main__":
    app()
