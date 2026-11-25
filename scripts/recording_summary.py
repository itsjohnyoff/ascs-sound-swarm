"""Report capture-scoped frame loss and optionally log a completed session."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from urllib.parse import unquote
from zipfile import BadZipFile

import numpy as np


def read_frame_report(path: Path) -> list[str]:
    # Read only metadata: ten-minute captures can contain hundreds of MB of PCM.
    with np.load(path, allow_pickle=False) as archive:
        stats = json.loads(archive["meta"].item())["frame_stats"]
    if not isinstance(stats, dict) or not stats:
        raise ValueError("Recording does not contain frame counts")

    report = []
    for phone_id, values in sorted(stats.items()):
        received = values["received_frames"]
        dropped = values["dropped_frames"]
        if not isinstance(received, int) or received <= 0 or not isinstance(dropped, int) or dropped < 0:
            raise ValueError(f"Invalid frame counts for {phone_id}")
        total = received + dropped
        label = json.dumps(unquote(phone_id), ensure_ascii=False)[1:-1]
        report.append(
            f"{label}: {dropped}/{total} dropped ({100 * dropped / total:.3f}%), "
            f"{values['sample_gaps']} sample gaps ({values['missing_samples']} samples), "
            f"{values['segments']} segments, {values['disconnects']} disconnects, "
            f"{values['clock_jumps']} clock jumps"
        )
    return report


def markdown_cell(value: str) -> str:
    return " ".join(value.splitlines()).replace("\\", "\\\\").replace("|", r"\|")


def append_log(path: Path, capture_date: date, recording: Path, what: str, result: str) -> None:
    cells = [capture_date.isoformat(), recording.name, what, result]
    prefix = "| " + " | ".join(map(markdown_cell, cells[:2])) + " | "
    row = prefix + " | ".join(map(markdown_cell, cells[2:])) + " |\n"
    with path.open("a+", encoding="utf-8") as log:
        log.seek(0)
        previous = log.read()
        if any(line.startswith(prefix) for line in previous.splitlines()):
            raise ValueError(f"{path} already has a row for {recording.name} on {capture_date}")
        log.write(("\n" if previous and not previous.endswith("\n") else "") + row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--append-log", nargs="?", type=Path, const=Path("analysis/LOG.md"),
                        metavar="PATH", help="Append one row (default path: analysis/LOG.md)")
    parser.add_argument("--date", type=date.fromisoformat, metavar="YYYY-MM-DD")
    parser.add_argument("--what", help="Gate or session description for the log")
    parser.add_argument("--result", default="", help="Other evidence to include in the log row")
    args = parser.parse_args()
    if args.append_log is not None and (args.date is None or not args.what):
        parser.error("--append-log requires --date and --what")

    try:
        report = read_frame_report(args.recording)
        if args.append_log is not None:
            cells = [args.result, *report] if args.result else report
            append_log(args.append_log, args.date, args.recording, args.what, ". ".join(cells))
    except KeyError as exc:
        if exc.args[0] == "frame_stats":
            parser.error("Recording does not contain frame counts. Use a capture from the updated server.")
        parser.error(f"Recording metadata is missing {exc}")
    except (OSError, EOFError, ValueError, TypeError, BadZipFile) as exc:
        parser.error(str(exc))

    print(*report, sep="\n")
    print("Loss uses sequence and sample-index gaps within captured segments. Check disconnects and clock jumps separately.")


if __name__ == "__main__":
    main()
