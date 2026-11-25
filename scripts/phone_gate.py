#!/usr/bin/env python3
"""Check the live swarm or read timing evidence from a saved gate capture."""

from __future__ import annotations

import argparse
import json
import ssl
import urllib.request
from pathlib import Path

import numpy as np


def preflight(base_url: str, *, min_phones: int, require_sync: bool, insecure: bool) -> bool:
    base_url = base_url.rstrip("/")
    context = ssl._create_unverified_context() if insecure else None

    def get(path: str) -> dict:
        with urllib.request.urlopen(base_url + path, context=context, timeout=5) as response:
            return json.load(response)

    join_info = get("/api/join-info")
    session = get("/api/session")
    phones = [phone for phone in session["phones"] if phone["connected"]]
    synced = sum(phone["sync_state"] == "synced" for phone in phones)
    print(f"Join URL: {join_info['join_url']}")
    print(f"Phones connected: {len(phones)} | synced: {synced} | required: {min_phones}")
    print(f"Calibration: {session['calibration_status']}")
    for phone in sorted(phones, key=lambda p: p["phone_id"]):
        residual = phone["residual_ms"]
        fit = "—" if residual is None else f"{residual:.4f} ms"
        print(f"  {phone['phone_id']}: {phone['sync_state']}, {phone['sample_rate']} Hz, "
              f"residual {fit}, {phone['dropped_frames']} dropped frames in current clock epoch")

    warning = session["gdop_warning"]
    if warning:
        print(f"Geometry: {warning}")
    ready = len(phones) >= min_phones and (not require_sync or synced == len(phones) and not warning)
    print(f"Preflight: {'READY' if ready else 'HOLD'}")
    return ready


def timing(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(archive["meta"].item())

    start, stop = meta.get("started_at_s"), meta.get("stopped_at_s")
    if start is not None and stop is not None:
        print(f"Capture duration: {stop - start:.1f} s")
    calibrations = meta.get("calibrations", [])
    print(f"Live calibration results: {len(calibrations)}")
    for result in calibrations:
        phones = ", ".join(
            f"{pid}={report['residual_ms']:.4f} ms"
            if report["synced"] and report["residual_ms"] is not None else f"{pid}=unavailable"
            for pid, report in sorted(result["per_phone"].items())
        )
        print(f"  +{result['offset_s']:.3f} s {result['mode']} "
              f"{'solved' if result['success'] else 'failed'} ({phones})")

    events = meta.get("event_timings", [])
    print(f"Transient broadcasts: {len(events)}")
    print("Onset estimates use server receive time minus the remaining samples in the PCM batch.")
    print("Measure acoustic onset to painted pin separately with a video of the source and map.")
    latencies = []
    for event in events:
        onset = event["onset_offset_s"]
        broadcast = event["broadcast_offset_s"]
        latency = broadcast - onset
        latencies.append(latency)
        print(f"  {event['event_id']} ({event['sound_class']}): onset +{onset:.3f} s, "
              f"broadcast +{broadcast:.3f} s, elapsed {latency:.3f} s")
    if latencies:
        p50, p90 = np.percentile(latencies, [50, 90])
        print(f"Transient onset-to-broadcast estimate: p50 {p50:.3f} s, p90 {p90:.3f} s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("preflight", help="read the live join URL and per-phone state")
    status.add_argument("base_url", help="server origin, for example https://localhost:8443")
    status.add_argument("--min-phones", type=int, default=4)
    status.add_argument("--require-sync", action="store_true", help="require every connected phone synced and good geometry")
    status.add_argument("--insecure", action="store_true", help="accept the demo's self-signed LAN certificate")
    evidence = subparsers.add_parser("timing", help="read calibration and broadcast timing from an NPZ")
    evidence.add_argument("recording", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            if args.min_phones < 1:
                parser.error("--min-phones must be positive")
            return 0 if preflight(args.base_url, min_phones=args.min_phones,
                                  require_sync=args.require_sync, insecure=args.insecure) else 1
        timing(args.recording)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
