import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
import sys
import time
import cv2
from .config import Config
from .engine import Engine
from .vision import DEFAULT_MODEL, FaceTracker, download_model
from .capture import Camera
from .dashboard import render
from .synthetic import generate, demo_frame
from .io import Recorder


def parser():
    p = argparse.ArgumentParser(description="SW-CF educational contactless pulse prototype")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("download-model", help="Download Google's face landmark model")
    sub.add_parser("doctor", help="Print dependencies and model status (does not access camera)")
    for command in ("demo", "webcam", "video"):
        s = sub.add_parser(command)
        if command == "video":
            s.add_argument("path", type=Path)
        s.add_argument("--duration", type=float, default=0, help="Seconds; 0 means until quit/EOF (demo: 70)")
        s.add_argument("--headless", action="store_true", help="No GUI; save final dashboard PNG")
        s.add_argument("--output", type=Path, default=None)
        s.add_argument("--model", type=Path, default=DEFAULT_MODEL)
        s.add_argument("--mode", choices=["weighted", "equal", "green"], default="weighted")
        s.add_argument("--window", type=float, default=6)
        if command == "demo":
            s.add_argument("--bpm", type=float, default=72)
            s.add_argument("--seed", type=int, default=7)
            s.add_argument("--clean", action="store_true")
        if command == "webcam":
            s.add_argument("--camera", type=int, default=0)
            s.add_argument("--width", type=int, default=640)
            s.add_argument("--height", type=int, default=480)
            s.add_argument("--backend", choices=["auto","dshow","msmf","v4l2","any"], default="auto")
    b = sub.add_parser("benchmark", help="Evaluate local UBFC-rPPG subject folders")
    b.add_argument("root", type=Path)
    b.add_argument("--output", type=Path, default=Path("outputs/ubfc"))
    b.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    b.add_argument("--modes", nargs="+", choices=["weighted","equal","green"], default=["weighted","equal","green"])
    b.add_argument("--limit", type=int, default=0)
    return p


def doctor():
    print(f"Python {sys.version.split()[0]}")
    for name in ("numpy", "scipy", "opencv-contrib-python", "mediapipe", "matplotlib"):
        try:
            print(f"{name}: {importlib.metadata.version(name)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{name}: MISSING")
    print(f"Face model: {DEFAULT_MODEL} ({'ready' if DEFAULT_MODEL.exists() else 'missing'})")


def run(args):
    cfg = Config(window=args.window)
    engine = Engine(cfg, mode=args.mode)
    out = args.output or Path("outputs") / (args.command + "_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    recorder = Recorder(out)
    tracker = camera = video = None
    image = None
    meta = {"source": args.command, "mode": args.mode, "config": vars(cfg),
            "simulated": args.command == "demo", "medical_device": False}
    try:
        if args.command == "demo":
            if not 45 <= args.bpm <= 150:
                raise ValueError("Demo BPM must be within 45-150")
            duration = args.duration or 70
            stamps, colors, qualities = generate(duration, bpm=args.bpm, seed=args.seed, corrupted=not args.clean)
            meta.update(target_bpm=args.bpm, seed=args.seed, quality_source="synthetic oracle annotations")
        else:
            tracker = FaceTracker(args.model)
            if args.command == "webcam":
                camera = Camera(args.camera, args.width, args.height, backend=args.backend)
                print(f"Capture backend: {camera.backend}; driver buffer request accepted: {camera.buffer_accepted}")
                meta.update(backend=camera.backend, driver_buffer_request_accepted=camera.buffer_accepted)
            else:
                video = cv2.VideoCapture(str(args.path))
                if not video.isOpened():
                    raise ValueError(f"Cannot open video: {args.path}")
                video_fps = video.get(cv2.CAP_PROP_FPS)
                if video_fps < 8 or video_fps > 240:
                    raise ValueError("Video must have reliable 8-240 FPS metadata")
                meta.update(video=str(args.path), video_fps=video_fps)
        print(f"Output folder: {out.resolve()}")
        if not args.headless:
            cv2.namedWindow("SW-CF", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("SW-CF", 1280, 820)
        index, seq, origin = 0, 0, None
        started = time.perf_counter()
        while True:
            if args.command == "demo":
                if index >= len(stamps):
                    break
                t, rgb, q = stamps[index], colors[index], qualities[index]
                frame, polygons = demo_frame(rgb, q)
                extracted = (rgb, q, polygons)
            elif camera:
                seq, stamp, frame = camera.read(seq)
                if origin is None:
                    origin = stamp
                t = stamp - origin
                extracted = tracker.extract(frame, t)
            else:
                ok, next_frame = video.read()
                if not ok:
                    break
                frame = next_frame
                # Constant-frame-rate dataset time, independent of processing speed.
                t = index / video_fps
                extracted = tracker.extract(frame, t)
            if args.duration and t > args.duration:
                break
            updated = False
            if extracted is None:
                engine.reset("No face - center your face and improve lighting")
                polygons, q = [], [0]*5
            else:
                rgb, q, polygons = extracted
                recorder.sample(t, rgb, q)
                updated = engine.push(t, rgb, q)
                if updated:
                    recorder.estimate(engine.state)
            # Headless processing renders only at output updates, saving CPU.
            if not args.headless or updated or image is None:
                image = render(frame, polygons, q, engine.state, args.command == "demo", getattr(args,"bpm",72))
            if not args.headless:
                cv2.imshow("SW-CF", image)
                key = cv2.waitKey(1) & 255
                if key in (27, ord("q")) or cv2.getWindowProperty("SW-CF", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord("r"):
                    engine.reset("Reset - reacquiring")
                if key == ord("s"):
                    cv2.imwrite(str(out / f"dashboard_{index:06d}.png"), image)
                if not camera:
                    rate = cfg.fps if args.command == "demo" else video_fps
                    time.sleep(max(0, (index+1)/rate - (time.perf_counter()-started)))
            index += 1
        meta.update(processed_frames=index, processing_wall_seconds=time.perf_counter()-started)
        # Always render the final state, including face loss/quality rejection.
        if index:
            image = render(frame, polygons, q, engine.state, args.command == "demo", getattr(args,"bpm",72))
        if image is not None:
            cv2.imwrite(str(out / "dashboard.png"), image)
        a = engine.state.analysis
        print(f"Final status: {engine.state.status}; BPM: {a.bpm if a else None}")
        print(f"Experimental SDNN proxy: {engine.state.hrv['sdnn_ms'] if engine.state.hrv else 'unavailable'}")
    finally:
        if camera:
            camera.close()
        if video:
            video.release()
        if tracker:
            tracker.close()
        recorder.close(engine.state, meta)
        if not args.headless:
            cv2.destroyAllWindows()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "download-model":
            print(download_model())
        elif args.command == "doctor":
            doctor()
        elif args.command == "benchmark":
            from .benchmark import benchmark
            benchmark(args)
        else:
            if args.duration < 0:
                raise ValueError("Duration cannot be negative")
            run(args)
        return 0
    except KeyboardInterrupt:
        print("Stopped.")
        return 130
    except (RuntimeError, ValueError, FileNotFoundError, OSError) as error:
        print(f"SW-CF: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
