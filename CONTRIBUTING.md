# Working on SW-CF

Use Python 3.11 and install `requirements-dev.txt` in a virtual environment.
Run `python main.py download-model`, then `python -m pytest -q` before pushing
changes to the processing code. A camera-free smoke test is
`python main.py demo --headless --duration 8`.

Team responsibilities and code explanations are in
`docs/FOUR_PERSON_STUDY_GUIDE.md`. The four roles cover acquisition, quality and
fusion, DSP and timing, and application/testing integration respectively.

Keep downloaded datasets, face models, environments, camera session outputs,
and temporary rendering files out of commits. Keep generated example results
clearly labeled as synthetic. Do not present simulated errors as real-subject
accuracy, or the experimental SDNN proxy as a validated stress classifier.

Use a separate branch for substantial changes and describe what changed and
how it was checked. Add tests when a processing change fixes a meaningful bug
or introduces behavior that needs verification.
