"""Regenerate deterministic synthetic evidence and report figures."""
from pathlib import Path
import csv
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import signal
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from swcf.config import Config
from swcf.dsp import analyze
from swcf.synthetic import generate


def main():
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    cfg, rows = Config(), []
    for bpm in (48,60,72,90,120,144):
        for corrupted in (False, True):
            for seed in (7,19,42):
                t, rgb, q = generate(10, bpm=bpm, seed=seed, corrupted=corrupted, jitter=True)
                for mode in ("weighted", "equal", "green"):
                    a = analyze(t, rgb, q, cfg, mode)
                    rows.append({"target_bpm": bpm, "corrupted": corrupted, "seed":seed,
                                 "mode":mode, "estimated_bpm":a.bpm,
                                 "absolute_error_bpm":abs(a.bpm-bpm) if a.bpm else None})
    with (out/"synthetic_results.csv").open("w", newline="", encoding="utf-8") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary = {"evidence_type":"synthetic RGB only; quality labels are supplied oracle annotations",
               "real_webcam_accuracy": "not measured", "ubfc_accuracy":"not measured", "groups":[]}
    for corrupted in (False,True):
        for mode in ("weighted","equal","green"):
            group = [r for r in rows if r["mode"]==mode and r["corrupted"]==corrupted]
            errors = [r["absolute_error_bpm"] for r in group if r["estimated_bpm"] is not None]
            summary["groups"].append({"corrupted":corrupted,"mode":mode,"cases":len(group),
                "accepted":len(errors), "mae_bpm":float(np.mean(errors)) if errors else None,
                "max_error_bpm":float(np.max(errors)) if errors else None})
    (out/"synthetic_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    t, rgb, q = generate(10, corrupted=True)
    a = analyze(t,rgb,q,cfg)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False,"axes.spines.right":False})
    fig, axes = plt.subplots(2,2,figsize=(11.7,7),layout="constrained")
    fig.suptitle("SW-CF synthetic validation | known target 72 BPM | not human measurements",fontsize=14)
    for i in range(5):
        axes[0,0].plot(t,rgb[:,i,1]-rgb[:,i,1].mean(),alpha=.7,label=f"Region {i+1}")
    axes[0,0].set(xlabel="Time (s)",ylabel="Green channel AC (levels)",title="Five input regions")
    axes[0,0].legend(ncol=3,fontsize=8)
    axes[0,1].bar(["Forehead","L cheek","R cheek","Nose","Chin"],a.weights,color=["#168a9b"]*3+["#d9952b"]*2)
    axes[0,1].set(ylabel="Normalized weight",title="Quality-weighted fusion (oracle synthetic quality)")
    axes[1,0].plot(a.times,a.pulse,color="#168a9b")
    axes[1,0].set(xlabel="Time (s)",ylabel="Normalized amplitude",title="Filtered fused pulse")
    keep=(a.frequencies>=.5)&(a.frequencies<=3)
    axes[1,1].plot(a.frequencies[keep]*60,a.power[keep],color="#168a9b")
    axes[1,1].axvline(72,color="#d9952b",linestyle="--",label="Known input")
    axes[1,1].set(xlabel="Rate (BPM)",ylabel="PSD",title=f"Welch estimate: {a.bpm:.2f} BPM")
    axes[1,1].legend()
    fig.savefig(out/"signal_pipeline.png",dpi=180); plt.close(fig)
    sos=signal.butter(4,[.75,2.5],fs=30,btype="bandpass",output="sos")
    f,h=signal.sosfreqz(sos,worN=4096,fs=30)
    fig,ax=plt.subplots(figsize=(9,4),layout="constrained")
    ax.plot(f,20*np.log10(np.maximum(abs(h)**2,1e-8)),label="Forward/backward magnitude")
    ax.axvspan(.75,2.5,alpha=.15,color="#168a9b")
    ax.set(xlim=(0,5),ylim=(-100,5),xlabel="Frequency (Hz)",ylabel="Gain (dB)",
           title="Butterworth bandpass: order-4 prototype; zero-phase forward/backward")
    ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/"filter_response.png",dpi=180); plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__ == "__main__":
    main()
