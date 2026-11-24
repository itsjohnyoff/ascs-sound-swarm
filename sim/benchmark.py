"""Offline benchmark: python -m sim.benchmark [--seed N] [--occlude p2] [--stubs]

Runs judge_scenario through SyncEngine + the CONTRACTS 4 event finder + LocalizationPipeline,
prints a Markdown table, saves analysis/benchmark-<timestamp>.md. Paste it in Agent 2 PRs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.dsp.replay import _parse_phone_positions, _parse_room
from backend.server.events import find_events_offline
from backend.server.stubs import StubLocalizationPipeline, StubSyncEngine
from shared.protocol import PhoneAudioSlice, PhoneConfig
from sim.room import judge_scenario, true_time

MATCH_WINDOW_S = 0.5


@dataclass
class Row:
    gt_x_m: float
    gt_y_m: float
    expected: str
    inside_hull: bool
    est_x_m: float | None = None
    est_y_m: float | None = None
    error_m: float | None = None
    kind: str | None = None
    sound_class: str | None = None

    @property
    def missed(self) -> bool:
        return self.error_m is None


@dataclass
class BenchmarkResult:
    rows: list[Row]
    runtime_ms_per_event: float

    @property
    def errors(self) -> list[float]:
        return [r.error_m for r in self.rows if r.error_m is not None]

    @property
    def missed(self) -> int:
        return sum(r.missed for r in self.rows)

    @property
    def mean_error_m(self) -> float | None:
        return float(np.mean(self.errors)) if self.errors else None


def load_slices(rec: dict) -> tuple[dict, dict[str, PhoneConfig], dict[str, PhoneAudioSlice]]:
    meta = json.loads(str(rec["meta"]))
    phones = {pid: PhoneConfig(phone_id=pid, x_m=p["x_m"], y_m=p["y_m"], sample_rate=int(rec[f"{pid}_fs"]))
              for pid, p in _parse_phone_positions(meta).items()}
    slices = {pid: PhoneAudioSlice(pid, rec[f"{pid}_pcm"], int(rec[f"{pid}_start_index"]), int(rec[f"{pid}_fs"]),
                                   rec[f"{pid}_valid"]) for pid in phones}
    return meta, phones, slices


def run_benchmark(rec: dict, *, stubs: bool = False) -> BenchmarkResult:
    meta, phones, slices = load_slices(rec)
    room = _parse_room(meta)
    truth = meta["truth"]
    targets = [e for e in truth["events"] if e["kind"] != "chirp"]
    rows = [Row(e["x_m"], e["y_m"], e["kind"], e["inside_hull"]) for e in targets]

    kw = {"engine": StubSyncEngine(), "localizer": StubLocalizationPipeline()} if stubs else {}
    started = time.perf_counter()
    detections = find_events_offline(slices, phones, room, **kw)
    runtime = (time.perf_counter() - started) * 1e3 / max(1, len(detections))
    if not detections:
        return BenchmarkResult(rows, runtime)

    engine = kw.get("engine")
    if engine is None:  # same engine the finder built; rebuild to read the reference clock
        from backend.dsp.pipeline import SyncEngine
        engine = SyncEngine()
        engine.add_chirp_observation(slices, phones, room, "lfm_single")
    ref = engine.ref_phone_id
    for det in detections:
        t_true = true_time(truth["phones"][ref], engine.to_local(ref, det.t0))
        idx = min(range(len(targets)), key=lambda i: abs(targets[i]["t_s"] - t_true))
        row = rows[idx]
        if abs(targets[idx]["t_s"] - t_true) > MATCH_WINDOW_S or det.result is None or not row.missed:
            continue
        r = det.result
        row.est_x_m, row.est_y_m, row.kind, row.sound_class = r.x_m, r.y_m, det.kind, r.sound_class
        row.error_m = math.hypot(r.x_m - row.gt_x_m, r.y_m - row.gt_y_m)
    return BenchmarkResult(rows, runtime)


def format_markdown(result: BenchmarkResult) -> str:
    def f(v, nd=2):
        return "-" if v is None else f"{v:.{nd}f}"

    lines = ["| # | gt | est | error_m | expected | kind | class | inside_hull |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, r in enumerate(result.rows, 1):
        est = "MISSED" if r.missed else f"({f(r.est_x_m)}, {f(r.est_y_m)})"
        lines.append(f"| {i} | ({r.gt_x_m:.2f}, {r.gt_y_m:.2f}) | {est} | {f(r.error_m, 3)} | {r.expected} | "
                     f"{r.kind or '-'} | {r.sound_class or '-'} | {r.inside_hull} |")
    e = result.errors
    stats = (f"mean {f(np.mean(e) if e else None, 3)} m, median {f(np.median(e) if e else None, 3)} m, "
             f"p90 {f(np.percentile(e, 90) if e else None, 3)} m, missed {result.missed}/{len(result.rows)}, "
             f"offline runtime {result.runtime_ms_per_event:.0f} ms/event incl. full-recording onset scan")
    return "\n".join(lines + ["", stats])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Offline judge-scenario benchmark")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--occlude", help="phone id whose direct path is blocked for target sounds")
    ap.add_argument("--stubs", action="store_true", help="run with stub facades")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args(argv)
    occluded = (args.occlude, 0.15, 0.0025, 1.0) if args.occlude else None
    result = run_benchmark(judge_scenario(seed=args.seed, occluded=occluded), stubs=args.stubs)
    md = format_markdown(result)
    print(md)
    out = Path(args.out) / f"benchmark-{time.strftime('%Y%m%d-%H%M%S')}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Benchmark seed={args.seed} occlude={args.occlude} stubs={args.stubs}\n\n{md}\n")
    print(f"\nsaved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
