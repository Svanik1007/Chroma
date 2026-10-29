# Attribution and external assets

- **POS method:** Wang, den Brinker, Stuijk, and de Haan,
  Algorithmic Principles of Remote-PPG (2017),
  https://doi.org/10.1109/TBME.2016.2609282 . The implementation here was written
  for this project from the described mathematical operations.
- **MediaPipe** and the face-landmarker model are Google/MediaPipe third-party
  components. Model source:
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  . The model downloaded for this workspace is not included in the portable
  submission ZIP; setup downloads it from Google. See upstream terms, model
  cards, and https://github.com/google-ai-edge/mediapipe .
- The MediaPipe Tasks upstream privacy notice distinguishes local input
  processing from utilization/performance metrics:
  https://github.com/google-ai-edge/mediapipe/blob/master/mediapipe/tasks/web/vision/README.md .
  This project does not itself upload camera frames.
- **NumPy, SciPy, OpenCV, Matplotlib, pytest:** retain their respective upstream
  licenses when redistributing their packages. Dependencies and the local
  virtual environment are not bundled in the submission ZIP.
- **UBFC-rPPG** is an external human-subject dataset, not redistributed here.
  Obtain it from https://sites.google.com/view/ybenezeth/ubfcrppg and cite
  Bobbia et al., Unsupervised skin tissue segmentation for remote
  photoplethysmography (2017).
- The optional integration script downloads Google's public test portrait from
  https://storage.googleapis.com/mediapipe-assets/portrait.jpg . Its source image,
  derived frames, and video are excluded from the submission ZIP. Artificial
  color modulation is only an integration fixture, not a physiological signal.
- The demo's illustrated face is generated directly using OpenCV drawing
  primitives. All included pulse charts and synthetic numbers come from the
  supplied simulation scripts.
