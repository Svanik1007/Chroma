# Spatially-Weighted Chrominance Fusion (SW-CF)

A Signals and Systems project: five facial regions, quality-aware POS fusion,
Butterworth filtering, Welch heart-rate estimation, and experimental pulse-rate
variability. All pulse/video processing runs on the laptop.

**Fresh clone or ZIP:** install 64-bit Python 3.11 and run `setup_windows.bat`
once, then double-click `run_demo.bat` for a simulated demonstration or
`run_webcam.bat` for the webcam. The original working laptop already has its
local environment and model prepared; those generated files are not stored in Git.

## Reports and team study material

- [Start here](START_HERE.md)
- [Project report (PDF)](output/pdf/SW_CF_Project_Report.pdf)
- [Complete four-person code and viva guide (PDF)](output/pdf/SW_CF_Four_Person_Code_and_Viva_Guide.pdf)
- [Editable four-person study guide](docs/FOUR_PERSON_STUDY_GUIDE.md)
- [Demo script and viva preparation](docs/DEMO_AND_VIVA.md)
- [Attribution and external assets](THIRD_PARTY.md)

The repository includes code, documentation, and synthetic evidence. Local
camera sessions, personal screenshots, human datasets, downloaded models,
virtual environments, and intermediate PDF-rendering files are ignored.

## Clone this repository

```powershell
git clone https://github.com/Svanik1007/SW-CF.git
cd SW-CF
.\setup_windows.bat
```

This is a private repository, so cloning requires a GitHub account with access.
The Windows GitHub Actions workflow installs dependencies, downloads the face
model, runs the tests, and checks a headless synthetic demonstration. Its
artifacts are simulated data, not measurements from a webcam or a real dataset.

## Included

- Webcam dashboard with five moving region outlines, region weights, pulse, PSD, BPM.
- Deterministic simulation with clearly labeled artificial RGB signals.
- Video-file processing, CSV exports, session metadata, and dashboard screenshots.
- UBFC-rPPG Dataset 2 evaluation with weighted POS, equal POS, and green baselines.
- Tests, reproducible simulation results and plots, report PDF and editable report source.
- Demo script, troubleshooting, viva preparation, and presentation outline.

## Windows installation (another laptop / extracted ZIP)

Install **64-bit Python 3.11** from python.org with the Python launcher, then
double-click `setup_windows.bat`. Internet is needed for dependencies and the
Google face-landmarker model. Subsequent monitoring does not need a cloud API.

Manual PowerShell setup:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --timeout 120 -r requirements.txt
.\.venv\Scripts\python.exe main.py download-model
.\.venv\Scripts\python.exe main.py doctor
.\.venv\Scripts\python.exe main.py demo
.\.venv\Scripts\python.exe main.py webcam
```

Use these exact commands if plain `python` selects another Python version.
Python 3.11 is the tested configuration; the package minimum is 3.10, replacing
the original proposed 3.9 floor to accommodate current dependencies.

Linux/macOS: create a virtual environment with Python 3.11, install
`requirements.txt`, then run the same `main.py` commands with that environment's
Python. Desktop OpenCV and camera permissions are required. These platforms have
not been exercised here.

## Commands

```powershell
# Artificial input, real DSP; no camera or tracker needed
.\.venv\Scripts\python.exe main.py demo --bpm 90

# Fast automated demo, including experimental variability after 61 seconds
.\.venv\Scripts\python.exe main.py demo --headless --duration 70 --output outputs\demo

# Camera options and backend fallback
.\.venv\Scripts\python.exe main.py webcam --camera 0 --width 640 --height 480
.\.venv\Scripts\python.exe main.py webcam --backend msmf

# Local constant-frame-rate recording, processed with its media time
.\.venv\Scripts\python.exe main.py video "C:\path\recording.avi" --headless

# Same implementation, alternative baseline fusion
.\.venv\Scripts\python.exe main.py webcam --mode equal

# Tests and reproducible simulation evidence
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\validate.py
```

In the dashboard: **Q / Escape** quits, **R** clears acquisition, **S** saves a
screenshot. Pulse estimation begins after six usable seconds and refreshes
roughly twice per second. The experimental SDNN proxy requires about 61 seconds
of continuous acceptable data; quality rejection can extend that time.

## Good webcam demonstration conditions

Sit with one face centered, about 40-80 cm from the camera. Use steady diffuse
front lighting; hold still, avoid talking, and keep eyes/lips out of the patches.
Start at 640x480. Camera exposure, compression, facial hair, pose, shadows,
skin appearance, and laptop load affect signal quality. Reset when changing
subjects or conditions. Keep the simulation ready as an explicitly labeled
fallback for the classroom.

## Output files

Each run writes a new `outputs/<source>_<timestamp>/` directory unless `--output`
is supplied. Reusing the same output directory overwrites that session's files.

| File | Contents |
| --- | --- |
| `rgb.csv` | Timestamp, five RGB means, five physical quality scores per detected frame |
| `estimates.csv` | BPM (blank when rejected), quality, concentration, FPS, SDNN proxy, weights |
| `last_pulse.csv` | Final analyzed window, normalized waveform |
| `session.json` | Source, configuration, final state and experimental variability |
| `dashboard.png` | Last dashboard image |

No webcam video is recorded. The screenshot contains the final camera frame.
Treat exported face screenshots and derived pulse information as personal data.
MediaPipe uses its own upstream package/privacy terms; see `THIRD_PARTY.md`.

## UBFC evaluation

Download **UBFC-rPPG Dataset 2** yourself from the
[dataset authors](https://sites.google.com/view/ybenezeth/ubfcrppg), complying with
their terms. Human dataset files are not bundled.

```text
data/UBFC/
  subject1/
    vid.avi
    ground_truth.txt
  subject2/
    vid.avi
    ground_truth.txt
```

```powershell
.\.venv\Scripts\python.exe main.py benchmark data\UBFC --limit 1
.\.venv\Scripts\python.exe main.py benchmark data\UBFC --output outputs\ubfc
```

Ground truth must have three rows: PPG samples, device BPM, timestamps in
seconds. The evaluator subtracts the first reference timestamp and compares
the mean supplied BPM over the same trailing six seconds. It assumes the video
and reference start together; it does not search for a favorable lag. It reports
MAE, RMSE, bias, coverage, per-subject metrics, and failed subjects. Overlapping
windows are correlated; they are not independent experimental subjects.

**No real webcam accuracy or UBFC accuracy is claimed in this submission.**
The included numeric results are synthetic only. The small synthetic errors
do not predict accuracy on real people.

## Scientific scope

The camera requests 30 FPS and a driver buffer size of 1, while a single-slot
capture thread discards old frames. Neither setting guarantees timing or zero
latency. Actual processed timestamps determine the sample rate; gaps trigger
reacquisition. A six-second window is approximately 180 frames at 30 FPS.

Zero-phase filtering runs on buffered windows. It is not a causal zero-delay
filter; its edge samples can change when the window advances. The filter uses
SciPy `butter(4, ..., btype='bandpass')`: an order-4 prototype yields an
eight-pole bandpass; forward/backward processing squares the magnitude response.

The displayed SDNN proxy is the standard deviation of screened **pulse**
intervals, not ECG-confirmed normal RR intervals. 30 FPS limits interval timing;
sub-frame interpolation does not restore missing information. A 60-second
estimate is experimental, depends on duration, and is **not a validated stress
index**. There is no low/medium/high stress classifier.

## Project map

```text
main.py                 CLI entry
swcf/capture.py         camera and timestamps
swcf/vision.py          landmark polygons, RGB means, physical quality
swcf/dsp.py             POS, filtering, fusion, PSD, interval screening
swcf/engine.py          time buffers, warm-up, rejection, HRV history
swcf/dashboard.py       OpenCV dashboard and headless rendering
swcf/benchmark.py       UBFC evaluation and baseline comparison
swcf/synthetic.py       reproducible artificial input
scripts/validate.py    simulation evidence and figures
tests/                 DSP, timing, quality, evaluation, model integration
docs/                  report source, demo guide, viva, slide outline
output/pdf/            printable report
results/               generated evidence and example dashboard
```

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Camera cannot open | Close Teams/Zoom/browser camera tabs, allow desktop camera access, try `--camera 1` or `--backend msmf` |
| No face | Face the light and camera; remove obstruction; check model download |
| Poor signal / no BPM | Hold still in diffuse light for six seconds; avoid auto-exposure changes; try 480p |
| No SDNN proxy | Wait at least 61 seconds of continuous clean signal; rejection is intentional |
| Module missing | Use `.venv\Scripts\python.exe` and rerun setup |
| GUI unavailable | Install desktop OpenCV, not headless OpenCV; use `--headless` for file exports |
| Changing BPM on a still photo | Periodic lighting/noise can resemble pulse; a plausible spectrum does not prove a heartbeat |
| Dataset not found | Use the folder structure above; Dataset 1/UBFC-Phys formats are not supported |

Before submitting, add your real names/course details to the report source and
regenerate the PDF, rehearse the demo, and describe the evidence accurately.

To rebuild the printable report after editing `docs/REPORT.md`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-report.txt
.\.venv\Scripts\python.exe scripts\build_report.py
```

The output is `output/pdf/SW_CF_Project_Report.pdf`. The PDF's figures come
from `results/`; keep those files alongside the source.
