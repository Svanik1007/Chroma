"""UBFC-rPPG Dataset 2 evaluator. No bundled human data or invented scores."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np
from .config import Config
from .engine import Engine
from .vision import FaceTracker


def read_ground_truth(path):
    rows = [np.fromstring(line, sep=" ") for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows) != 3 or len({len(r) for r in rows}) != 1 or len(rows[0]) < 3:
        raise ValueError("Expected UBFC Dataset 2: three equal rows (PPG, BPM, timestamps)")
    ppg, bpm, time = rows
    if not all(np.isfinite(r).all() for r in rows) or np.any(np.diff(time) <= 0):
        raise ValueError("Ground-truth values must be finite and timestamps increasing")
    return time-time[0], bpm, ppg


def reference_bpm(t, bpm, start, end):
    # Do not extrapolate the reference outside its synchronized time range.
    if start < t[0]-1e-6 or end > t[-1]+1e-6:
        return None
    inside = (t >= start) & (t <= end) & (bpm > 0)
    if inside.sum() < 2:
        return None
    return float(np.mean(bpm[inside]))


def metrics(rows):
    valid = [r for r in rows if r["prediction_bpm"] is not None and r["reference_bpm"] is not None]
    eligible = sum(r["reference_bpm"] is not None for r in rows)
    errors = np.array([r["prediction_bpm"]-r["reference_bpm"] for r in valid])
    return {"eligible_windows": eligible, "accepted_windows": len(valid),
            "coverage": len(valid)/eligible if eligible else 0.0,
            "mae_bpm": float(np.abs(errors).mean()) if len(errors) else None,
            "rmse_bpm": float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
            "bias_bpm": float(errors.mean()) if len(errors) else None}


def benchmark(args):
    videos = sorted(args.root.glob("**/vid.avi"))
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise ValueError("No vid.avi found. Expected root/subjectN/{vid.avi,ground_truth.txt}")
    args.output.mkdir(parents=True, exist_ok=True)
    cfg = Config()
    rows, failures, subject_results = [], [], []
    for video_path in videos:
        subject = str(video_path.parent.relative_to(args.root))
        cap = tracker = None
        subject_rows = []
        try:
            gt_t, gt_bpm, _ = read_ground_truth(video_path.parent/"ground_truth.txt")
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not cap.isOpened() or not 8 <= fps <= 240:
                raise ValueError("Unreadable video or invalid FPS")
            engines = {m: Engine(cfg, mode=m) for m in dict.fromkeys(args.modes)}
            tracker = FaceTracker(args.model)
            i, next_eval = 0, cfg.window
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                t = i/fps
                data = tracker.extract(frame, t)
                for e in engines.values():
                    if data is None:
                        e.face_lost(t)
                    else:
                        e.push(t, data[0], data[1], data[3])
                # Fixed evaluation schedule includes missing/rejected predictions.
                if t >= next_eval - 1e-6:
                    for mode, e in engines.items():
                        a = e.state.analysis
                        fresh = a is not None and t-e.last_update <= cfg.update + 1/fps
                        # Score what the user sees: the smoothed value, only when
                        # the latest window was accepted (held values are not counted).
                        pred = e.state.bpm_display if fresh and a.bpm is not None else None
                        span = a.window_s if fresh else cfg.window
                        ref = reference_bpm(gt_t, gt_bpm, t-span, t)
                        subject_rows.append({"subject": subject, "mode": mode, "end_s": t,
                            "prediction_bpm": pred, "reference_bpm": ref,
                            "quality": a.quality if fresh else None})
                    next_eval += cfg.update
                i += 1
            if i < fps*cfg.window:
                raise ValueError("Video is shorter than one analysis window")
            rows.extend(subject_rows)
            for mode in engines:
                result = {"subject": subject, "mode": mode, **metrics([r for r in subject_rows if r["mode"]==mode])}
                subject_results.append(result)
            print(f"Evaluated {subject}: {i} frames")
        except (ValueError, RuntimeError, OSError) as error:
            failures.append({"subject": subject, "error": str(error)})
            print(f"Skipped {subject}: {error}")
        finally:
            if cap is not None:
                cap.release()
            if tracker is not None:
                tracker.close()
    fields = ["subject","mode","end_s","prediction_bpm","reference_bpm","quality"]
    with (args.output/"windows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    summary = {"dataset": "UBFC-rPPG Dataset 2", "attempted_subjects":len(videos),
               "completed_subjects":len(videos)-len(failures), "failures":failures,
               "reference": "Mean provided oximeter BPM over the same trailing analysis window (6-12 s)",
               "timing": "Constant FPS video time; ground-truth first timestamp subtracted; no optimized lag",
               "modes": {m: metrics([r for r in rows if r["mode"]==m]) for m in dict.fromkeys(args.modes)},
               "subjects": subject_results, "config": vars(cfg)}
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary["modes"], indent=2))
    if failures:
        print(f"See summary.json: {len(failures)} subject(s) failed; no scores invented for them.")
