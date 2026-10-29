# Your SW-CF project

## Run now on this laptop

1. Double-click **run_demo.bat** for an artificial 72 BPM demonstration.
2. Double-click **run_webcam.bat** for the actual webcam. Hold still in steady
   front lighting for six seconds. Press Q to quit, R to reset, S for a screenshot.
3. Read **docs/DEMO_AND_VIVA.md** before presenting.

## On another laptop

Extract the ZIP, install 64-bit Python 3.11, and run **setup_windows.bat** once
with Internet access. Then use the two launchers. The ZIP contains source and
results, with dependencies and the face model downloaded during setup.

The same setup is required after cloning the GitHub repository. The `.venv`
folder and downloaded face model are intentionally not committed to Git.

## Submission material

- **output/pdf/SW_CF_Project_Report.pdf** - printable project report.
- **docs/REPORT.md** - editable report source; add your names and course details.
- **docs/DEMO_AND_VIVA.md** - demonstration script, viva questions, slide outline.
- **output/pdf/SW_CF_Four_Person_Code_and_Viva_Guide.pdf** - complete four-person
  code explanations, examples, presentation scripts, and viva answers.
- **results/** - synthetic results, figures, saved dashboard, and test output.
- **README.md** - full installation, commands, evaluation, and troubleshooting.
- **swcf/**, **main.py**, **scripts/**, **tests/** - implementation and checks.

## Be precise when presenting

This is a working software prototype with synthetic verification. The included
demo and numbers are artificial. Real webcam-reference and UBFC accuracy have
not been measured. The SDNN proxy is experimental pulse-interval variability;
it is not a validated stress score. The real UBFC dataset must be obtained from
its authors and evaluated separately with the included command.
