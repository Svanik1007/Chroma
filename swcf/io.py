from pathlib import Path
import csv
import json
import numpy as np


class Recorder:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.raw_file = (self.folder / "rgb.csv").open("w", newline="", encoding="utf-8")
        self.est_file = (self.folder / "estimates.csv").open("w", newline="", encoding="utf-8")
        self.raw = csv.writer(self.raw_file)
        self.est = csv.writer(self.est_file)
        self.raw.writerow(["time_s"] + [f"p{p}_{c}" for p in range(5) for c in ("r","g","b")]
                          + [f"p{p}_quality" for p in range(5)])
        self.est.writerow(["time_s", "bpm", "quality", "spectral_concentration", "fs_hz", "sdnn_proxy_ms"]
                              + [f"p{p}_weight" for p in range(5)])

    def sample(self, t, rgb, quality):
        self.raw.writerow([t, *np.asarray(rgb).ravel(), *quality])

    def estimate(self, state):
        a = state.analysis
        if a is not None:
            self.est.writerow([state.timestamp, a.bpm, a.quality, a.concentration, a.fs,
                               state.hrv["sdnn_ms"] if state.hrv else None, *a.weights])

    def close(self, state, metadata):
        self.raw_file.close(); self.est_file.close()
        a = state.analysis
        if a is not None:
            np.savetxt(self.folder/"last_pulse.csv", np.column_stack([a.times,a.pulse]),
                       delimiter=",", header="time_s,pulse_normalized", comments="")
        metadata.update(status=state.status, last_bpm=a.bpm if a else None, hrv=state.hrv)
        (self.folder/"session.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
