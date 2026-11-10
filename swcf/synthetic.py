"""Reproducible RGB simulation, explicitly separate from real measurements."""
import numpy as np
import cv2


def generate(duration=70, fps=30, bpm=72, seed=7, corrupted=True, jitter=False,
             motion_bursts=(), flicker_hz=None, flicker_amp=.006, noise=.045,
             return_background=False):
    """Artificial RGB for 5 patches (+ optional background reference).

    motion_bursts: [(start_s, duration_s), ...] - quality drops to 0.05, the
        patches get large noise during the burst and a lasting level shift
        (the patch lands on slightly different skin), like a real head nod.
    flicker_hz: a coloured illumination flicker common to the whole scene
        (patches and background), like tube lights or auto-exposure hunting.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(round(duration * fps)) / fps
    if jitter:
        t[1:] += rng.uniform(-.005, .005, len(t)-1)
    pulse = np.sin(2*np.pi*bpm/60*t)
    # Skin DC levels and <1% pulsatile changes; common illumination drift.
    base = np.array([[177, 132, 109], [170, 127, 104], [180, 136, 113],
                     [185, 140, 118], [162, 120, 100]], float)
    light = (1 + .025*np.sin(2*np.pi*.18*t))[:, None]
    if flicker_hz:
        # Green-heavy colour change (like fluorescent tubes) that looks like the pulse colour
        # change, so POS alone cannot separate it from the pulse.
        light = light * (1 + flicker_amp*np.sin(2*np.pi*flicker_hz*t)[:, None]*[.2, 1.0, .1])
    rgb = base[None, :, :] * light[:, None, :]
    rgb += pulse[:, None, None] * np.array([.10, .65, .18])[None, None, :]
    rgb += rng.normal(0, noise, rgb.shape)
    quality = np.full((len(t), 5), .95)
    if corrupted:
        # Two deliberately corrupted patches; quality is an oracle annotation
        # here, not an estimate from a camera. This isolates the fusion behavior.
        rgb[:, 3] += (3*np.sin(2*np.pi*1.9*t))[:, None] * [1, .2, 1]
        rgb[:, 4] += rng.normal(0, 1.5, (len(t), 3))
        quality[:, 3] = .08
        quality[:, 4] = .12
    for start, length in motion_bursts:
        during = (t >= start) & (t < start + length)
        rgb[during] += rng.normal(0, 4, (during.sum(), 5, 3))
        rgb[t >= start + length] += rng.normal(0, 1.5, (5, 3))
        quality[during] = .05
    if not return_background:
        return t, rgb, quality
    background = np.array([96, 104, 110], float)[None, :] * light
    background += rng.normal(0, noise, background.shape)
    return t, rgb, quality, background


def demo_frame(rgb, quality):
    frame = np.full((480, 640, 3), (31, 25, 20), np.uint8)
    cv2.ellipse(frame, (320, 242), (124, 180), 0, 0, 360, (104, 127, 170), -1)
    cv2.ellipse(frame, (320, 80), (95, 25), 0, 0, 180, (35, 40, 50), -1)
    cv2.ellipse(frame, (270, 190), (20, 7), 0, 0, 360, (35, 40, 50), -1)
    cv2.ellipse(frame, (370, 190), (20, 7), 0, 0, 360, (35, 40, 50), -1)
    cv2.line(frame, (291, 317), (349, 317), (55, 65, 95), 3)
    bounds = [(275, 105, 365, 145), (221, 222, 278, 277),
              (362, 222, 419, 277), (303, 217, 337, 271), (285, 350, 355, 380)]
    polygons = []
    for p, (x0, y0, x1, y1) in enumerate(bounds):
        poly = np.array([[x0,y0], [x1,y0], [x1,y1], [x0,y1]], np.int32)
        cv2.fillConvexPoly(frame, poly, tuple(int(v) for v in np.clip(rgb[p, ::-1], 0, 255)))
        polygons.append(poly)
    cv2.putText(frame, "SIMULATED INPUT - NOT A PERSON", (25, 455),
                cv2.FONT_HERSHEY_SIMPLEX, .65, (70, 220, 255), 2, cv2.LINE_AA)
    return frame, polygons
