# Spatially-Weighted Chrominance Fusion (SW-CF)

## Signals and Systems Project Report

**Submission edition:** 27 October 2026

**Team / roll numbers / department / supervisor:** add your own details before submission.

## Abstract

This project implements a software prototype for extracting a contactless pulse
signal from an ordinary RGB webcam. Five facial regions are located with
MediaPipe landmarks. Their spatial mean colors form sampled RGB time series.
Brightness, clipping, and landmark motion provide physical quality gates; a
spectral concentration score supplies temporal quality information. A sliding
Plane-Orthogonal-to-Skin (POS) projection extracts a pulse candidate from each
region, followed by adaptive weighted fusion, Butterworth bandpass filtering,
and Welch power spectral density estimation. The interface displays region
weights, a normalized waveform, an estimated pulse rate, and an experimental
pulse-interval variability statistic. Tests and artificial color signals verify
the software and expected DSP behavior. An evaluator is included for future
UBFC-rPPG validation. No clinical accuracy or stress-detection performance is
claimed.

## 1. Problem statement and objectives

Contact sensors and wearables are unnecessary for a classroom demonstration of
remote photoplethysmography (rPPG). Changes in reflected skin light can contain
a weak blood-volume-related periodic component. Camera motion, illumination,
compression, and automatic exposure often dominate that component. A single
poor facial region can therefore corrupt an otherwise useful measurement.

The engineering objective is to construct and demonstrate a reproducible
pipeline that combines several regions according to their quality, rather than
depending on a single patch. The system must expose its intermediate behavior,
reject obviously inadequate input, export numeric signals, and support
comparison with simpler baselines. A further objective is to explain the
sampling, filtering, estimation, and latency tradeoffs using Signals and
Systems concepts.

The deliverable is an educational research prototype. It does not diagnose
conditions, classify stress, or replace a pulse oximeter or ECG. POS and rPPG
are established methods; SW-CF is this project's integration and weighting
design, not a claim of a newly invented physiological technique [1].

## 2. System architecture

```text
RGB camera / local video
          |
 timestamped frame acquisition
          |
 MediaPipe face landmarks
          |
 five interior facial polygons
          |
 mean RGB + light/clipping/motion quality
          |
 timestamp resampling -> per-region POS -> bandpass
          |
 spectral SQI -> normalized, gated fusion -> bandpass
          |
 Welch PSD -> BPM / rejection -> dashboard and CSV
          |
 separate long history -> screened pulse intervals -> SDNN proxy
```

The regions are forehead, left cheek, right cheek, nose, and chin. Compact
polygons use indices shared with the original 468-point face mesh. The current
MediaPipe Tasks model supplies additional points; unused points are ignored.
Polygons shrink toward their centers to reduce boundary contamination. This is
geometric region selection, not a learned pixel-level skin classifier [2].

## 3. Acquisition, sampling, and buffering

The application requests 640x480 or 1280x720 frames at 30 FPS. On Windows it
prefers DirectShow; on Linux it prefers V4L2. A driver buffer size of 1 is
requested, and a capture thread maintains a single latest-frame slot. This
limits application backlog when computation falls behind, but camera exposure,
driver queues, and operating-system scheduling remain outside the application's
control. OpenCV explicitly documents backend-dependent capture properties [3].

Live timestamps come from a monotonic high-resolution clock at frame delivery.
They are delivery-time proxies, not sensor exposure timestamps. Recorded videos
use frame index divided by reported FPS, assuming constant-frame-rate media.
Consequently processing a video faster than real time does not alter its pulse
frequency.

The short analysis window is six seconds: approximately 180 frames at 30 FPS.
A time-based deque also retains up to 65 seconds for variability analysis.
Frames are linearly resampled onto a uniform grid before digital filtering.
The effective sample rate is (N - 1)/(last timestamp - first timestamp).
Capture gaps over 0.25 seconds trigger reacquisition. Analysis requires an
effective sample rate of at least 8 Hz. Neither interpolation nor frame
discarding can recover information lost to acquisition.

## 4. Spatial signals and quality scores

For patch k and color channel c, the spatial measurement is:

```text
C[k,c,n] = mean of channel c over pixels inside patch k at frame n
```

The per-frame physical quality q[k,n] lies in [0,1]. A soft brightness score
penalizes very dark or very bright regions; the fraction of near-black or
near-saturated pixels reduces quality further. Landmark displacement is
normalized by face width and elapsed time to express motion in face-widths per
second. A completely off-frame patch receives zero quality.

```text
q = light_score * (1 - clipped_fraction)^2 * exp(-3 * normalized_motion)
```

These are transparent engineering heuristics, not a learned confidence
probability or a skin-tone-calibrated model. The mean physical score is combined
with spectral concentration to form patch SQI. If a patch is contaminated by a
periodic lighting artifact, it may still look spectrally strong. This limitation
cannot be resolved by SQI alone.

## 5. POS color-space projection

Wang et al. proposed POS to reduce intensity-related interference by projecting
temporally normalized RGB signals onto a plane orthogonal to a normalized skin
direction [1]. The implementation uses sliding 1.6-second windows:

```text
c(t)  = RGB(t) / temporal_mean(RGB) - 1
s1(t) = g(t) - b(t)
s2(t) = g(t) + b(t) - 2*r(t)
alpha = std(s1) / max(std(s2), epsilon)
h(t)  = s1(t) + alpha*s2(t)
```

Each local h is mean-centered and overlap-added. Counts normalize how many
windows contributed to each sample. POS is applied separately to each patch so
that regions with different DC skin appearance and lighting are normalized
locally. Vectorized sliding-window calculations keep the long-history analysis
practical on a CPU. This processing is time varying and data dependent, rather
than one fixed linear time-invariant operation.

## 6. Digital filtering and weighted fusion

The passband is 0.75-2.5 Hz, corresponding to 45-150 BPM. Linear detrending
precedes filtering. SciPy second-order sections provide better numerical
conditioning than a high-order direct-form polynomial implementation [4].

The code uses `butter(4, [0.75, 2.5], btype='bandpass', fs=fs, output='sos')`.
Here 4 is the low-pass prototype order: the resulting bandpass has eight poles.
Forward/backward `sosfiltfilt` processing has zero phase on the buffered signal
and squares the one-pass magnitude response. Therefore nominal one-pass cutoff
points have about -6 dB combined gain, rather than -3 dB. The project describes
this accurately instead of calling the whole operation simply fourth order.

For each region, spectral concentration rho is the ratio of in-band PSD within
0.18 Hz of its dominant frequency to total passband PSD. Half a second at each
end of the window is excluded from the SQI/PSD calculation to reduce edge
effects. Region signals are standardized by their interior standard deviation.
Constant regions receive zero score.

```text
SQI[k] = mean(q[k,:]) * rho[k]
score[k] = SQI[k]^2
w[k,n] = q[k,n]*score[k] / sum_j(q[j,n]*score[j])
fused[n] = sum_k(w[k,n] * standardized_pulse[k,n])
```

The fused signal is bandpass-filtered again to reduce modulation caused by
changing weights. Normalized time-average weights are shown in the interface.
Equal-POS and mean-green baselines are included for ablation. They still use
the shared acquisition/quality rejection machinery, so they are not completely
ungated end-to-end algorithms.

Zero phase is achieved with samples already stored in the window. This is not
strictly causal streaming or a promise of zero latency. Newly plotted edge
samples can change as later windows arrive. Approximately six seconds of
initial acquisition and a half-second update cadence are intentional design
tradeoffs.

## 7. Heart-rate estimation

Welch's method estimates PSD using Hann-windowed segments of up to six seconds.
The dominant peak inside the allowed passband determines the pulse-rate
estimate [5]. A local parabolic interpolation of log power refines the displayed
peak position:

```text
estimated_BPM = 60 * estimated_peak_frequency_Hz
```

Zero padding provides a finer numerical frequency grid; it does not improve
the information-theoretic resolution of a short observation. A six-second
window has a basic frequency scale near 1/6 Hz, or 10 BPM. Trimming and Hann
windowing broaden spectral features further. A clean single sinusoid can still
yield a small interpolated error, but this does not justify the same accuracy
claim for noisy human video.

BPM is withheld when fused physical quality is below 0.20, dominant-band
concentration is below 0.40, or the signal has negligible amplitude. Face loss
clears the acquisition state. A rejected short window breaks long-history
continuity. These gates reject several obvious failures but cannot prove that
a periodic signal is physiological or correctly distinguish every harmonic.

## 8. Experimental pulse-interval variability

A separate history of at least 61 seconds is used before attempting interval
statistics. Peaks require minimum spacing and prominence; local parabolic
refinement provides a sub-frame timing estimate. The first and last half-second
are excluded. Intervals outside 400-1334 ms or more than 20% from the median
are rejected. At least 90% must pass, and median interval rate must agree with
the spectral estimate. The remaining interval sample standard deviation is
reported in milliseconds using N-1 normalization.

```text
SD_pulse = sqrt(sum((IBI_i - mean_IBI)^2) / (M - 1))
```

The interface calls this an experimental SDNN proxy. It is pulse-rate
variability, not verified ECG normal-to-normal HRV: the application cannot
identify ectopic beats reliably. Screening can bias variability downward and
reject real variation. Thirty-FPS sampling gives about 33.3 ms between frames;
interpolation cannot restore missing beat-timing information. Published HRV
guidance emphasizes duration and artifact handling, with conventional
short-term comparisons commonly using five-minute recordings [6,7].

SDNN depends on duration, posture, breathing, rhythm, and measurement quality.
No validated mapping from this proxy to psychological stress is implemented.
The original 'stress monitor' idea is retained as a future research direction,
requiring an appropriate study and reference labels.

## 9. Implementation and reproducibility

The tested environment is Windows, Python 3.11.9, NumPy 2.4.6, SciPy 1.17.1,
OpenCV contrib 4.14.0.94, and MediaPipe 0.10.35. The implementation uses the
current Tasks face-landmarker interface rather than relying on legacy
`mp.solutions.face_mesh`. Python 3.11 is recommended for reproducing the run.
Pinned installed versions are supplied in `requirements-lock.txt`.

The application supports webcam, constant-frame-rate local video, simulated
RGB input, and UBFC evaluation. It exports raw patch means, quality, accepted
and rejected BPM estimates, final pulse waveform, session metadata, and a
dashboard image. No webcam video file is recorded. Dependency/model downloads
are separate setup actions; image processing occurs on the laptop. Upstream
MediaPipe terms and possible package telemetry are distinct from application
video handling [8].

The test suite covers six target pulse rates, timestamp jitter and ordering,
frame gaps, no-pulse rejection, poor lighting, corrupted regions, filter
attenuation, face-loss reset, interval warm-up, UBFC parsing, coverage accounting,
and real model initialization on blank frames. The simulation script generates
fixed-seed numeric results and figures. An optional public-portrait fixture
checks landmark geometry and the full video path with deliberately artificial
color modulation. It is not a measurement of the pictured person's pulse.

## 10. Results and evidence boundaries

The generated synthetic experiment uses target rates of 48, 60, 72, 90, 120,
and 144 BPM; three random seeds; clean and corrupted conditions; and three
fusion modes. Each condition/mode group contains 18 ten-second artificial
signals with timestamp jitter. The corruption affects two regions. Synthetic
physical quality labels are supplied by the generator as oracle annotations,
so this test isolates fusion and does not validate camera quality estimation.

| Condition | Mode | Accepted / cases | MAE (BPM) |
| --- | --- | --- | --- |
| Clean synthetic | Weighted POS | 18 / 18 | 0.042 |
| Clean synthetic | Equal POS | 18 / 18 | 0.042 |
| Clean synthetic | Mean green | 18 / 18 | 0.037 |
| Corrupted synthetic | Weighted POS | 18 / 18 | 0.042 |
| Corrupted synthetic | Equal POS | 18 / 18 | 0.252 |
| Corrupted synthetic | Mean green | 18 / 18 | 0.230 |

These values come from `results/synthetic_results.csv` and
`results/synthetic_summary.json`. They are intentionally rounded in this report.
The favorable corruption result is conditional on accurate supplied quality
labels. It is a software demonstration, not evidence of superior performance
on real subjects. Green performs slightly better on this uncomplicated clean
simulation; there is no universal superiority claim.

The 70-second dashboard simulation with a 72 BPM target produced a final
estimate of approximately 72.14 BPM and an experimental interval SD near
5.27 ms. Its ideal input has constant rate, so nonzero interval SD represents
noise/processing error rather than real physiological variability. The
color-modulated still-image video yielded approximately 72.08 BPM through the
full face-tracker path. Neither is human pulse validation.

The automated test suite passes. Exact test output is included in
`results/test_run.txt`. No actual webcam-reference accuracy or real UBFC
score has been measured, and no subject-study results are invented.

## 11. UBFC benchmark protocol

The evaluator targets UBFC-rPPG Dataset 2 folders containing `vid.avi` and
`ground_truth.txt`. The authors describe RGB videos recorded at 30 FPS and
640x480, with synchronized CMS50E pulse-oximeter reference data [9]. Users must
obtain the dataset from the authors under the applicable terms.

Three ground-truth rows are parsed as PPG waveform, device BPM, and timestamps.
The first reference timestamp is subtracted. Video time is frame index/FPS,
assuming simultaneous starts and constant frame rate. No lag is optimized.
Mean supplied device BPM within each trailing six-second window is compared
with the estimate every 0.5 seconds. The measured PPG row is loaded and
validated, but is not used to compute the current reference BPM metric.

MAE, RMSE, signed bias, accepted-window coverage, per-subject results, and failed
subjects are saved. Missing predictions stay missing. Coverage uses all
reference-covered evaluation windows, including face-loss/rejection windows,
rather than counting only successful estimates. Reference-uncovered windows
are excluded. Correlated overlapping windows must not be treated as independent
subjects for statistical significance. No train/test leakage is involved in
this fixed-rule prototype, but future threshold tuning should use a separate
development set.

## 12. Limitations and future work

The system assumes one mostly frontal, still face in adequate steady light.
Facial hair, makeup, occlusions, low resolution, motion, compression, changes
in auto-exposure, rolling shutter, and varying skin appearance can reduce
performance. Region quality thresholds are heuristics and have not been
calibrated across people or cameras. No fairness or subgroup accuracy claim is
made. Switching subjects requires a reset.

All patches can contain the same periodic illumination artifact; a strong PSD
peak alone does not prove a heartbeat. The 45-150 BPM band excludes rates
outside that range by design. Short windows trade rate stability for
responsiveness. Linear timestamp interpolation assumes sufficiently dense
samples. The displayed waveform has arbitrary units and cannot be interpreted
as blood pressure, oxygen saturation, or ECG morphology.

Future work should evaluate UBFC and additional datasets, quantify errors and
coverage by motion/lighting conditions, compare longer windows, test robust
skin masks and motion compensation, and measure CPU latency on specified
hardware. Experimental variability needs longer recordings and reliable
beat-level reference timing. Stress modeling would require validated study
labels, subject-separated evaluation, and evidence beyond SDNN alone.

## 13. Conclusion

SW-CF provides an implemented, inspectable classroom pipeline from RGB video
to quality-weighted pulse estimation. It demonstrates spatial averaging,
temporal normalization, sampled-data processing, spectral estimation, adaptive
fusion, and quality rejection in one application. Synthetic tests verify the
software's expected behavior; real-subject accuracy and stress monitoring
remain validation tasks. The project is ready for a transparent software
demonstration and subsequent dataset experiments.

## References

[1] Wang, W., den Brinker, A. C., Stuijk, S., and de Haan, G. (2017).
Algorithmic Principles of Remote-PPG. IEEE Transactions on Biomedical
Engineering, 64(7), 1479-1491. https://doi.org/10.1109/TBME.2016.2609282

[2] Google. Face landmark detection guide for Python.
https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker/python

[3] OpenCV. VideoCapture class reference.
https://docs.opencv.org/4.x/d8/dfe/classcv_1_1VideoCapture.html

[4] SciPy. sosfiltfilt and filtfilt documentation.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html

[5] SciPy. Welch power spectral density estimation.
https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html

[6] Laborde, S., Mosley, E., and Thayer, J. F. (2017). Heart Rate Variability
and Cardiac Vagal Tone in Psychophysiological Research. Frontiers in Psychology.
https://doi.org/10.3389/fpsyg.2017.00213

[7] Shaffer, F., and Ginsberg, J. P. (2017). An Overview of Heart Rate
Variability Metrics and Norms. Frontiers in Public Health.
https://doi.org/10.3389/fpubh.2017.00258

[8] Google. MediaPipe Tasks Vision Package, privacy notice.
https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md

[9] Bobbia, S., Macwan, R., Benezeth, Y., Mansouri, A., and Dubois, J. (2017).
Unsupervised skin tissue segmentation for remote photoplethysmography. Pattern
Recognition Letters. Dataset author page:
https://sites.google.com/view/ybenezeth/ubfcrppg
