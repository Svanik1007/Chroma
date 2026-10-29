# Demonstration and viva guide

## Before entering the room

1. Use `run_demo.bat` once and confirm the artificial 72 BPM result.
2. Use `run_webcam.bat` in the actual room lighting. Rehearse six-second warm-up.
3. Keep the report PDF, `results/dashboard_demo.png`, and this guide available.
4. Add team/course details in `docs/REPORT.md`; do not add invented experiments.
5. If permitted, run UBFC evaluation on a downloaded subject and include the
   resulting summary with coverage. Real dataset results are not currently included.

## Five-minute demonstration

**0:00-0:40 - State the objective.**
"We use weak periodic color variations in facial regions to estimate a pulse
signal. We combine five patches according to quality because movement and
illumination can affect different patches differently."

**0:40-1:30 - Explain the diagram.**
Camera, timestamps, landmarks, RGB averaging, POS projection, filtering,
quality-weighted fusion, and PSD. Point to the weights and the waveform.

**1:30-2:30 - Webcam.**
Start `run_webcam.bat`, face diffuse front light, and hold still for six seconds.
Explain acquisition and rejection states. Move slightly, then stop and allow
reacquisition. The heuristic does not guarantee suppression of every artifact.
Press R when changing subjects, S to save a screenshot, Q to quit.

**2:30-3:30 - Synthetic evidence.**
Start `run_demo.bat`. Say explicitly: "This input is generated, with a known
72 BPM sinusoid and two corrupted patches. Its low-quality labels are supplied
by the simulator. It validates processing behavior, not human accuracy."
Observe the forehead and cheek weights dominate. Show the Welch peak.

**3:30-4:20 - Signals and Systems link.**
At 30 samples/second, a 1.2 Hz pulse is 72 BPM. The filter passband rejects
slow drift and higher-frequency noise. Six-second windows balance latency
and spectral resolution. Show `results/filter_response.png`.

**4:20-5:00 - Scope and results.**
Show the generated test output. Say: "We have software and synthetic
verification. Real webcam-reference accuracy and UBFC accuracy remain
unmeasured. The variability readout is an experimental pulse-interval statistic,
not a validated psychological stress classifier."

If the webcam signal is poor, explain the quality gate and switch to the
explicitly labeled simulation. Do not present the simulator as a live sensor.

## Questions you should be ready to answer

**What is the signal?** Spatially averaged RGB intensity per patch over time;
the pulse contribution is small compared with DC skin reflection and nuisance
variation. The final waveform is a normalized estimate in arbitrary units.

**Why five patches?** They provide spatial redundancy. A shaded or moving patch
can contribute less while other regions remain usable. Correlated artifacts
across the whole face still pose a problem.

**What is POS?** A known color projection method from Wang et al. (2017).
After temporal channel normalization, two chrominance combinations are mixed
using a standard-deviation ratio to reduce intensity-related interference.

**What is your contribution?** A quality-aware five-region integration,
inspection dashboard, timestamp handling, rejection policy, and evaluation
workflow. POS, Butterworth filtering, and Welch PSD are established components.
We do not claim a proven new state-of-the-art algorithm.

**Why not just green?** Green contains useful pulse information, but it also
contains lighting/motion changes. POS exploits color relationships. Our
synthetic clean case slightly favors green, so no method is universally best.

**What is SQI?** A heuristic combination of physical image quality and temporal
spectral concentration. It is not a clinical confidence probability.

**Why normalize pulse amplitudes before fusion?** So region brightness and
raw amplitude do not by themselves determine dominance. Explicit quality
weights decide the contribution. This can amplify noise, so zero-amplitude and
quality gating are important.

**Why 0.75-2.5 Hz?** It corresponds to the chosen 45-150 BPM demonstration
range. Real rates outside that interval cannot be measured reliably by this
configuration. It is a design constraint, not the entire human heart-rate range.

**Is this a fourth-order filter?** It uses an order-4 Butterworth prototype.
Bandpass transformation gives eight poles; forward/backward filtering squares
the magnitude response and yields zero phase on the buffered window.

**Is zero phase zero latency?** No. It uses already buffered samples in both
directions. Initial warm-up is six seconds, output refreshes every half second,
and boundary samples can be revised as the window advances.

**What does zero padding do?** It samples the spectral curve more densely.
It does not create new temporal information or make two close true components
resolvable in a short observation.

**Why not deterministic 30 FPS?** The application can request a rate, while
hardware and drivers decide actual delivery. We measure timestamps and
resample. Large gaps lead to reacquisition.

**Is the whole pipeline LTI?** No. A fixed-rate Butterworth filter is LTI in
its interior, but face tracking, window normalization, POS alpha selection,
quality weights, clipping, and peak selection make the full system nonlinear
and/or time varying. Finite-window padding introduces boundary behavior.

**What is SDNN?** Clinically it is the standard deviation of normal-to-normal
beat intervals. This system screens pulse intervals and reports an SDNN proxy;
it cannot establish ECG-normal beats. Its 60-second duration differs from
conventional five-minute comparisons, and low camera FPS limits timing.

**Does low SDNN mean stress?** Not reliably by itself. Duration, breathing,
posture, artifacts, and rhythm matter. There is no validated stress score here.

**How do you evaluate?** Compare estimates with synchronized reference BPM,
report MAE/RMSE/bias and the fraction of eligible windows accepted, preserve
missing estimates, and report per-subject outcomes. Do not use synthetic errors
as claims about real subjects.

**Why can a photo show a BPM?** Periodic illumination or artificial color
modulation can create a strong signal. A spectral peak is not proof of a living
heartbeat; this illustrates a major limit of appearance-based measurement.

## Suggested presentation outline (10 slides)

1. Title, team, objective: five-region contactless pulse prototype.
2. Optical principle: weak pulsatile color on a large reflectance baseline.
3. System block diagram and software components.
4. Five regions and physical quality formula.
5. POS equations and sliding-window overlap-add.
6. Butterworth response, weighted fusion, and noncausal buffered filtering.
7. Welch PSD, sampling, six-second resolution/latency tradeoff.
8. Dashboard demo and clearly labeled synthetic results table.
9. Experimental variability and UBFC evaluation protocol.
10. Limitations, completed deliverables, and future validation.

Use the supplied figures and dashboard screenshot. Keep the statement
"synthetic validation; real-subject accuracy not measured" on the results slide.
