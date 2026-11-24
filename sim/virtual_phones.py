"""Stream a simulated scenario as N phones over the real WS protocol.

python -m sim.virtual_phones --server wss://localhost:8443 --phones 4 --scenario judge [--fast]

Each phone sends phone_hello, then 2048-sample binary frames paced by its own sample
clock (real time by default). Lets the whole live stack run on one laptop.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import ssl
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from shared.protocol import encode_audio_frame
from sim.room import judge_scenario, sync_scenario, true_time

FRAME = 2048


@dataclass
class Frame:
    at_s: float      # true time at which the phone would have this frame ready
    payload: bytes


def frames_for(rec: dict, phone_id: str) -> Iterator[Frame]:
    meta = json.loads(str(rec["meta"]))
    tp = meta["truth"]["phones"][phone_id]
    pcm, valid = rec[f"{phone_id}_pcm"], rec[f"{phone_id}_valid"]
    for seq, k in enumerate(range(0, len(pcm), FRAME)):
        if not valid[k:k + FRAME].all():
            continue  # a dropped frame never reaches the server
        chunk = pcm[k:k + FRAME]
        at = true_time(tp, tp["start_index"] + k + len(chunk))  # ready once the last sample exists
        yield Frame(at, encode_audio_frame(chunk, seq, tp["fs"], tp["start_index"] + k, client_time_ms=at * 1e3))


async def _phone(url: str, rec: dict, pid: str, t_origin: float, speed: float, ctx) -> None:
    import websockets

    meta = json.loads(str(rec["meta"]))
    p = meta["phones"][pid]
    async with websockets.connect(f"{url}/ws/phone/{pid}", ssl=ctx, max_size=None) as ws:
        await ws.send(json.dumps({"type": "phone_hello", "phone_id": pid, "label": pid, "x_m": p["x_m"],
                                  "y_m": p["y_m"], "sample_rate": int(rec[f"{pid}_fs"]),
                                  "device_info": "virtual phone"}))
        for frame in frames_for(rec, pid):
            delay = t_origin + frame.at_s / speed - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            await ws.send(frame.payload)


async def stream(url: str, rec: dict, speed: float = 1.0, insecure: bool = True) -> None:
    ctx = None
    if url.startswith("wss://"):
        ctx = ssl.create_default_context()
        if insecure:  # self-signed LAN cert from scripts/start_https.py
            ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
    pids = sorted(json.loads(str(rec["meta"]))["phones"])
    t_origin = time.monotonic() + 0.5
    await asyncio.gather(*(_phone(url, rec, pid, t_origin, speed, ctx) for pid in pids))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--server", default="wss://localhost:8443")
    ap.add_argument("--phones", type=int, default=4)
    ap.add_argument("--scenario", choices=["judge", "sync"], default="judge")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="stream 4x faster than real time")
    args = ap.parse_args(argv)
    make = judge_scenario if args.scenario == "judge" else sync_scenario
    rec = make(n_phones=args.phones, seed=args.seed)
    print(f"streaming {args.scenario} scenario as {args.phones} phones to {args.server}")
    print("calibrate from the dashboard before the first chirp lands (t = 1 s), or use the chirps in the scenario")
    asyncio.run(stream(args.server.rstrip("/"), rec, speed=4.0 if args.fast else 1.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
