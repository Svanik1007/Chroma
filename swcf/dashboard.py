"""OpenCV dashboard, also renderable to PNG in headless mode."""
import cv2
import numpy as np
from .config import PATCH_NAMES

BG = (24, 19, 15)
PANEL = (39, 32, 25)
WHITE = (240, 236, 225)
MUTED = (170, 155, 138)
CYAN = (199, 223, 67)
ORANGE = (90, 186, 255)


def text(img, value, xy, size=.6, color=WHITE, thick=1):
    cv2.putText(img, str(value), xy, cv2.FONT_HERSHEY_SIMPLEX, size, color, thick, cv2.LINE_AA)


def plot(img, values, rect, color=CYAN, xvalues=None, xrange=None):
    x, y, w, h = rect
    cv2.rectangle(img, (x,y), (x+w,y+h), PANEL, -1)
    for frac in (.25, .5, .75):
        cv2.line(img, (x,y+int(frac*h)), (x+w,y+int(frac*h)), (55,46,36), 1)
    if values is None or len(values) < 2:
        text(img, "Waiting for usable signal...", (x+15,y+h//2), .5, MUTED)
        return
    v = np.asarray(values, float)
    if xvalues is None:
        xx = np.linspace(x, x+w-1, len(v))
    else:
        lo, hi = xrange
        keep = (xvalues >= lo) & (xvalues <= hi)
        xx = x + (xvalues[keep]-lo)/(hi-lo)*(w-1)
        v = v[keep]
    if len(v) < 2:
        return
    v = (v - np.min(v)) / max(float(np.ptp(v)), 1e-12)
    pts = np.column_stack((xx, y+h-10-v*(h-20))).astype(np.int32)
    cv2.polylines(img, [pts], False, color, 2, cv2.LINE_AA)


def render(frame, polygons, quality, state, simulated=False, target_bpm=72, bg_rect=None):
    img = np.full((820, 1280, 3), BG, np.uint8)
    text(img, "SW-CF", (30, 47), 1.15, CYAN, 2)
    text(img, "SPATIALLY-WEIGHTED CHROMINANCE FUSION", (190,42), .68)
    text(img, "SIGNALS & SYSTEMS  /  rPPG RESEARCH PROTOTYPE", (30,76), .47, MUTED)
    tag = f"SIMULATION  |  target {target_bpm:g} BPM" if simulated else "LIVE / VIDEO  |  LOCAL PROCESSING"
    text(img, tag, (825,48), .52, ORANGE)
    view = frame.copy()
    if bg_rect is not None:
        x0, y0, x1, y1 = bg_rect
        cv2.rectangle(view, (x0, y0), (x1, y1), MUTED, 1)
        text(view, "BG", (x0 + 3, y0 - 5), .4, MUTED)
    for p, poly in enumerate(polygons):
        color = CYAN if quality[p] >= .4 else ORANGE
        cv2.polylines(view, [poly], True, color, 2)
        center = poly.mean(axis=0).astype(int)
        text(view, str(p+1), tuple(center), .55, (255,255,255), 2)
    img[105:585, 30:670] = cv2.resize(view, (640,480))
    a = state.analysis
    cv2.rectangle(img, (695,105), (1250,282), PANEL, -1)
    text(img, "HEART RATE", (717,136), .53, MUTED)
    shown = state.bpm_display
    bpm = f"{shown:.0f}" if shown is not None else "--"
    text(img, bpm, (712,213), 2.1, MUTED if state.held else WHITE, 3)
    text(img, "BPM", (880,211), .8, CYAN)
    if state.held:
        text(img, "holding last value", (980,211), .5, ORANGE)
    measured = f"{a.fs:.1f} Hz  |  window {state.window_s:.1f} s" if a is not None else "acquiring"
    text(img, f"Sample rate: {measured}", (717,254), .5, MUTED)
    text(img, "REGION FUSION WEIGHTS", (711,317), .53)
    for p, name in enumerate(PATCH_NAMES):
        y = 349 + p*34
        text(img, f"{p+1}  {name}", (712,y), .45, MUTED)
        cv2.rectangle(img, (872,y-13), (1157,y+2), PANEL, -1)
        weight = a.weights[p] if a is not None else 0
        cv2.rectangle(img, (872,y-13), (872+int(285*weight),y+2), CYAN if quality[p]>=.4 else ORANGE, -1)
        text(img, f"{100*weight:.0f}%", (1173,y), .43)
    sdnn = f"{state.hrv['sdnn_ms']:.1f} ms" if state.hrv else "collecting / quality gated"
    text(img, "EXPERIMENTAL PULSE-INTERVAL SD (SDNN proxy)", (711,545), .44, MUTED)
    text(img, sdnn, (711,576), .65, ORANGE)
    text(img, "PULSE WAVEFORM  /  NORMALIZED AMPLITUDE", (30,618), .49)
    plot(img, a.pulse if a else None, (30,632,775,122))
    text(img, "WELCH PSD  /  45 - 150 BPM", (835,618), .48)
    plot(img, a.power if a else None, (835,632,415,122), ORANGE,
         a.frequencies*60 if a else None, (45,150))
    text(img, f"{state.status}   [t = {state.timestamp:.0f} s]", (30,784), .57,
         CYAN if a and a.bpm else ORANGE)
    text(img, "Q / Esc: quit   R: reset   S: screenshot", (835,784), .43, MUTED)
    text(img, "Educational estimates. SDNN proxy is not a validated stress score.", (30,809), .40, MUTED)
    return img
