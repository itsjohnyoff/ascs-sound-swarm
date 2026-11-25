#!/usr/bin/env python3
"""Soak test (Gate 3 prep): python scripts/soak.py [--minutes 10] [--phones 4] [--speed 1]

Starts the real server in a subprocess, streams a long simulated session as virtual
phones over WS, and plays the dashboard's auto-chirp role (arm, chirp, solve every
30 s). Reports crashes, server memory over time, dropped frames, per-chirp residual,
detection rate, error and clap-to-detection latency. Exit code 1 if a MUST fails.
"""

from __future__ import annotations

import argparse
import asyncio
import ctypes
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.protocol import AUTO_CHIRP_PERIOD_S  # noqa: E402
from sim.room import DEFAULT_ROOM, JUDGE_POINTS, SimEvent, default_phones, simulate  # noqa: E402

INSIDE_POINTS = JUDGE_POINTS[:8]  # the last two judge points are outside the hull
MATCH_WINDOW_S = 3.0              # a detection belongs to an event emitted at most this long before it
MAX_MEMORY_GROWTH_MB = 50.0
MIN_DETECTION_RATE = 0.9
LATENCY_TARGET_S = 0.6


def soak_events(duration_s: float) -> list[SimEvent]:
    """A chirp every AUTO_CHIRP_PERIOD_S from t=1 s, and two target sounds between chirps."""
    events, k, t = [], 0, 1.0
    while t + AUTO_CHIRP_PERIOD_S <= duration_s:
        events.append(SimEvent(t, DEFAULT_ROOM.speaker_x_m, DEFAULT_ROOM.speaker_y_m, "chirp"))
        for offset in (10.0, 20.0):
            x, y, kind = INSIDE_POINTS[k % len(INSIDE_POINTS)]
            events.append(SimEvent(t + offset, x, y, kind, duration_s=0.9 if kind == "drone" else None))
            k += 1
        t += AUTO_CHIRP_PERIOD_S
    return events


def match_detections(detections: list[dict], targets: list[SimEvent], t_origin: float, speed: float) -> dict:
    """First emission per event_id, matched to the latest unmatched target emitted <= MATCH_WINDOW_S before it."""
    first: dict[str, dict] = {}
    for d in detections:
        first.setdefault(d["event_id"], d)
    hits: dict[int, tuple[float, float]] = {}  # target index -> (error_m, latency_s in sim time)
    false = 0
    for d in sorted(first.values(), key=lambda d: d["recv_s"]):
        lags = [(d["recv_s"] - (t_origin + e.t_s / speed), i) for i, e in enumerate(targets)]
        lags = [(lag, i) for lag, i in lags if 0 <= lag <= MATCH_WINDOW_S and i not in hits]
        if not lags:
            false += 1
            continue
        lag, i = min(lags)
        hits[i] = (math.hypot(d["x_m"] - targets[i].x_m, d["y_m"] - targets[i].y_m), lag * speed)
    return {"hits": hits, "false": false, "targets": len(targets), "kinds": {i: e.kind for i, e in enumerate(targets)}}


def listener_pid(port: int, fallback: int) -> int:
    """PID that owns the listening socket. On Windows a venv python.exe is a launcher whose child runs the server."""
    if sys.platform != "win32":
        return fallback
    out = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[1].endswith(f":{port}") and parts[3] == "LISTENING":
            return int(parts[4])
    return fallback


def rss_mb(pid: int) -> float | None:
    """Resident memory of another process without psutil."""
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return None
        try:
            c = Counters(cb=ctypes.sizeof(Counters))
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(c), c.cb)
            return c.WorkingSetSize / 2**20 if ok else None
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    if sys.platform == "darwin":
        try:
            process = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=True)
            return int(process.stdout.strip()) / 1024
        except (OSError, ValueError, subprocess.CalledProcessError):
            return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


async def run(args) -> int:
    import httpx
    import websockets

    from sim.virtual_phones import stream

    duration = args.minutes * 60.0
    events = soak_events(duration)
    chirps = [e for e in events if e.kind == "chirp"]
    targets = [e for e in events if e.kind != "chirp"]
    print(f"simulating {duration:.0f} s, {args.phones} phones, {len(chirps)} chirps, {len(targets)} targets ...",
          flush=True)
    rec = simulate(DEFAULT_ROOM, default_phones(np.random.default_rng(args.seed), args.phones),
                   events, duration + 2.0, reflection_gain=0.3, seed=args.seed, notes="soak")

    base = f"127.0.0.1:{args.port}"
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.server.app:app", "--host", "127.0.0.1",
                               "--port", str(args.port), "--log-level", "info"], cwd=ROOT)
    detections: list[dict] = []
    calibrations: list[dict] = []
    memory: list[float | None] = []
    probes: list[tuple[float, float, float]] = []  # (t since origin, health rtt s, client loop lag s)
    stream_error = None
    try:
        async with httpx.AsyncClient(base_url=f"http://{base}", timeout=10) as http:
            for _ in range(100):
                try:
                    if (await http.get("/api/health")).status_code == 200:
                        break
                except httpx.TransportError:
                    await asyncio.sleep(0.2)
            else:
                print("FAIL: server never became healthy")
                return 1
            server_pid = listener_pid(args.port, server.pid)
            print(f"server up (pid {server_pid}), streaming ...", flush=True)

            async def dashboard():
                async with websockets.connect(f"ws://{base}/ws/dashboard", max_size=None) as ws:
                    async for raw in ws:
                        msg = json.loads(raw)
                        if msg.get("type") == "detection":
                            detections.append({**msg, "recv_s": time.monotonic()})

            async def sleep_until(t):
                await asyncio.sleep(max(0.0, t - time.monotonic()))

            async def auto_chirp(t_origin):
                # Same flow as the dashboard's auto-chirp: arm, (chirp plays), solve with a 4 s window.
                for e in chirps:
                    await sleep_until(t_origin + (e.t_s - 0.3) / args.speed)
                    armed = (await http.post("/api/calibrate/arm", json={"mode": "lfm_single"})).json()
                    await sleep_until(t_origin + (e.t_s + 1.0) / args.speed)
                    r = await http.post("/api/calibrate/solve", json={
                        "calibration_id": armed["calibration_id"], "mode": "lfm_single", "window_s": 4.0})
                    calibrations.append(r.json() if r.status_code == 200 else {"success": False, "error": r.text})

            async def sample_memory():
                while True:
                    memory.append(rss_mb(server_pid))
                    await asyncio.sleep(10.0)

            async def probe(t_origin):
                # Server responsiveness (health round trip) and our own loop lag, every 2 s.
                while True:
                    t = time.monotonic()
                    try:
                        await http.get("/api/health")
                        rtt = time.monotonic() - t
                    except httpx.HTTPError:
                        rtt = float("inf")
                    before = time.monotonic()
                    await asyncio.sleep(2.0)
                    probes.append((t - t_origin, rtt, time.monotonic() - before - 2.0))

            t_origin = time.monotonic() + 0.5  # stream() starts its clock 0.5 s after it is called
            side = [asyncio.create_task(c) for c in (dashboard(), auto_chirp(t_origin), sample_memory(),
                                                     probe(t_origin))]
            try:
                await stream(f"ws://{base}", rec, speed=args.speed)
            except Exception as exc:  # a phone socket dying is a soak finding, not a script crash
                stream_error = f"{type(exc).__name__}: {exc} at t=+{time.monotonic() - t_origin:.0f} s"
            await asyncio.sleep(MATCH_WINDOW_S)
            for task in side:
                task.cancel()
            alive = server.poll() is None
            session = (await http.get("/api/session")).json() if alive else {"phones": []}
    finally:
        server.terminate()
        server.wait(timeout=10)

    matched = match_detections(detections, targets, t_origin, args.speed)
    return report(args, alive, session, calibrations, memory, matched, probes, stream_error)


def report(args, alive: bool, session: dict, calibrations: list[dict], memory: list, matched: dict,
           probes: list = (), stream_error: str | None = None) -> int:
    nan = float("nan")
    hits = matched["hits"]
    errors = sorted(e for e, _ in hits.values())
    kinds = matched.get("kinds", {})
    latency = {k: sorted(lat for i, (_, lat) in hits.items() if kinds.get(i) == k) for k in ("clap", "drone")}
    mem = [m for m in memory if m is not None]
    warm = mem[len(mem) // 10:] or mem  # skip the first 10% (imports, first buffers filling)
    growth = statistics.median(warm[-3:]) - statistics.median(warm[:3]) if len(warm) >= 6 else nan
    residuals = [c["max_residual_ms"] for c in calibrations if c.get("success") and c.get("max_residual_ms") is not None]
    dropped = {p["phone_id"]: p.get("dropped_frames", 0) for p in session["phones"]}
    rate = len(hits) / max(1, matched["targets"])

    def pct(xs, q):
        return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else nan

    lines = [
        f"# Soak {args.minutes:g} min, {args.phones} virtual phones, speed {args.speed:g}x, seed {args.seed}",
        "",
        f"- server alive at end: {alive}",
        f"- server memory MB: start {warm[0] if warm else nan:.0f}, end {warm[-1] if warm else nan:.0f}, "
        f"peak {max(mem) if mem else nan:.0f}, growth {growth:+.1f} (limit {MAX_MEMORY_GROWTH_MB:.0f}), "
        f"{len(mem)} samples",
        f"- dropped frames: {dropped}",
        f"- calibrations: {sum(bool(c.get('success')) for c in calibrations)}/{len(calibrations)} ok; "
        f"max residual per chirp ms: median {statistics.median(residuals) if residuals else nan:.3f}, "
        f"worst {max(residuals) if residuals else nan:.3f}",
        f"- detections: {len(hits)}/{matched['targets']} ({rate:.0%}), unmatched {matched['false']}",
        f"- error m: mean {statistics.fmean(errors) if errors else nan:.3f}, p90 {pct(errors, 0.9):.3f}, "
        f"max {errors[-1] if errors else nan:.3f}",
        *(f"- {k} latency s, emission to dashboard: median {pct(v, 0.5):.3f}, p90 {pct(v, 0.9):.3f}"
          + (f" (TARGET < {LATENCY_TARGET_S})" if k == "clap" else "") for k, v in latency.items()),
    ]
    if probes:
        worst = max(probes, key=lambda p: p[1])
        lag = max(probes, key=lambda p: p[2])
        stalls = [f"+{t:.0f}s {rtt:.1f}s" for t, rtt, _ in probes if rtt > 1.0]
        lines.append(f"- server health rtt: worst {worst[1]:.3f} s at +{worst[0]:.0f} s; stalls > 1 s: "
                     f"{', '.join(stalls[:20]) or 'none'}; client loop lag worst {lag[2]:.3f} s at +{lag[0]:.0f} s")
    lines.append(f"- phone streams: {stream_error or 'all finished'}")
    musts = {
        "no crash": alive,
        "no phone disconnects": stream_error is None,
        "no memory growth": not growth > MAX_MEMORY_GROWTH_MB,  # nan (run too short to tell) passes
        "every chirp after the first calibrates": bool(calibrations) and all(c.get("success") for c in calibrations[1:]),
        f"detection rate >= {MIN_DETECTION_RATE:.0%}": rate >= MIN_DETECTION_RATE,
    }
    lines += ["", *(f"- MUST {name}: {'PASS' if ok else 'FAIL'}" for name, ok in musts.items())]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0 if all(musts.values()) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--phones", type=int, default=4)
    ap.add_argument("--speed", type=float, default=1.0, help=">1 streams faster than real time (smoke runs only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--out", help="also write the report to this Markdown file")
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
