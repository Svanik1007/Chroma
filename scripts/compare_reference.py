"""Compare a recorded session with readings from a smartwatch / pulse oximeter.

While testing, note the session clock shown on the dashboard ("t = 42 s") and
the reference device's BPM every ~10 s in a CSV file:

    time_s,bpm
    15,74
    25,76
    ...

Then run:
    python scripts/compare_reference.py outputs/webcam_YYYYMMDD_HHMMSS reference.csv

Each reference reading is matched to the nearest dashboard estimate within
+/-2 s. Missing or rejected estimates count against coverage, never as zero.
"""
from pathlib import Path
import argparse
import csv
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swcf.benchmark import metrics


def load(path, value):
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            t = float(row["time_s"])
            v = row.get(value)
            rows.append((t, float(v) if v not in (None, "", "None") else None))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("session", type=Path, help="Output folder containing estimates.csv")
    p.add_argument("reference", type=Path, help="CSV with time_s,bpm from the reference device")
    p.add_argument("--column", default="bpm_display",
                   help="estimates.csv column to score (bpm_display = what the dashboard shows; bpm = raw)")
    p.add_argument("--tolerance", type=float, default=2.0, help="Max time difference in seconds")
    args = p.parse_args(argv)
    estimates = load(args.session / "estimates.csv", args.column)
    if not estimates:
        raise SystemExit("estimates.csv has no rows")
    rows = []
    for t, ref in load(args.reference, "bpm"):
        nearest = min(estimates, key=lambda e: abs(e[0] - t))
        pred = nearest[1] if abs(nearest[0] - t) <= args.tolerance else None
        rows.append({"time_s": t, "reference_bpm": ref, "prediction_bpm": pred})
        shown = f"{pred:6.1f}" if pred is not None else "    --"
        print(f"t={t:6.1f} s   reference {ref:6.1f}   ours {shown}")
    result = metrics(rows)
    print(json.dumps(result, indent=2))
    (args.session / "reference_comparison.json").write_text(
        json.dumps({"rows": rows, "metrics": result, "column": args.column}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
