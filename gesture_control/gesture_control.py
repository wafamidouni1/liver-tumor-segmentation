import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import json
import time
import math
import os

# ── Paths ─────────────────────────────────────────────────
MODEL_PATH = r"./hand_landmarker.task"
CMD_FILE   = r"./gesture_commands.json"

if not os.path.exists(MODEL_PATH):
    import urllib.request
    print("Downloading hand model...")
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/1/"
           "hand_landmarker.task")
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Done!")

with open(CMD_FILE, 'w') as f:
    json.dump({"command": "none",
               "data": {},
               "timestamp": 0}, f)

# ── State ─────────────────────────────────────────────────
prev_cx      = None
prev_cy      = None
last_sent    = 0
last_gesture = "none"
seg_mode     = "a"
COOLDOWN     = 0.10

# ── Send command ──────────────────────────────────────────
def send_cmd(command, data=None, force=False):
    global last_sent
    now = time.time()
    if not force and now - last_sent < COOLDOWN:
        return False
    try:
        with open(CMD_FILE, 'w') as f:
            json.dump({
                "command":   command,
                "data":      data or {},
                "timestamp": now
            }, f)
        last_sent = now
        return True
    except:
        return False

# ── Gesture detection ─────────────────────────────────────
def detect_gesture(lm):
    """
    Index only  → rotate
    Pinch       → zoom in
    Victory V   → segments
    4 fingers   → zoom out
    Thumb up    → reset all
    """
    # Finger states
    idx = lm[8].y  < lm[5].y   # index up
    mid = lm[12].y < lm[9].y   # middle up
    rng = lm[16].y < lm[13].y  # ring up
    pnk = lm[20].y < lm[17].y  # pinky up

    # Thumb up: tip higher than index MCP
    # and higher than wrist
    thumb_up = (lm[4].y < lm[5].y and
                lm[4].y < lm[0].y)

    # Pinch: thumb tip near index tip
    pdx = lm[4].x - lm[8].x
    pdy = lm[4].y - lm[8].y
    pinch = math.sqrt(pdx*pdx + pdy*pdy) < 0.08

    # ── Thumb up: reset ───────────────────────────────────
    if thumb_up and not idx and \
       not mid and not rng and not pnk:
        return "thumb_up"

    # ── Pinch: zoom in ────────────────────────────────────
    if pinch:
        return "pinch"

    # ── Index only: rotate ────────────────────────────────
    if idx and not mid and \
       not rng and not pnk:
        return "index_only"

    # ── Victory V: segments ───────────────────────────────
    if idx and mid and \
       not rng and not pnk:
        return "victory"

    # ── 4 fingers: zoom out ───────────────────────────────
    if idx and mid and rng and pnk:
        return "four_fingers"

    return "unknown"

# ── Draw hand ─────────────────────────────────────────────
def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]
    pts = [(int(l.x*w), int(l.y*h)) for l in lm]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b],
                 (0,220,100), 2)
    for pt in pts:
        cv2.circle(frame, pt, 5,
                   (255,255,255), -1)
        cv2.circle(frame, pt, 5,
                   (0,150,80), 2)
    # Highlight index fingertip
    cv2.circle(frame, pts[8], 10,
               (0,255,255), -1)
    return frame

# ── Draw UI ───────────────────────────────────────────────
def draw_ui(frame, gesture):
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0,0), (w,110),
                  (10,10,30), -1)
    cv2.addWeighted(ov, 0.8, frame,
                    0.2, 0, frame)

    colors = {
        "index_only":   (0,255,255),
        "pinch":        (0,200,255),
        "victory":      (200,80,255),
        "four_fingers": (255,200,0),
        "thumb_up":     (0,255,120),
        "unknown":      (150,150,150),
        "none":         (80,80,80),
    }
    c = colors.get(gesture, (180,180,180))

    cv2.putText(frame,
                f"Gesture: {gesture.upper()}",
                (15, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2, c, 2)

    guide = [
        ("1 finger (index)", "Rotate 3D",   (0,255,255)),
        ("Pinch",            "Zoom in",     (0,200,255)),
        ("Victory V",        "Segments",    (200,80,255)),
        ("4 fingers",        "Zoom out",    (255,200,0)),
        ("Thumb up",         "Reset all",   (0,255,120)),
        ("Q",                "Quit",        (150,150,150)),
    ]
    for i, (g, a, gc) in enumerate(guide):
        cv2.putText(frame, f"{g}",
                    (w-310, 140+i*30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, gc, 1)
        cv2.putText(frame, f"-> {a}",
                    (w-130, 140+i*30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (200,200,200), 1)
    return frame

# ── Main ──────────────────────────────────────────────────
def main():
    global prev_cx, prev_cy, last_gesture, seg_mode

    print("="*45)
    print("  LIVER 3D GESTURE CONTROL")
    print("="*45)
    print("  1 finger (index) -> Rotate 3D")
    print("  Pinch            -> Zoom in")
    print("  Victory V        -> Toggle Segments")
    print("  4 fingers        -> Zoom out")
    print("  Thumb up         -> Reset all")
    print("  Q                -> Quit")
    print("="*45)

    base_opt = python.BaseOptions(
        model_asset_path=MODEL_PATH)
    opts = vision.HandLandmarkerOptions(
        base_options=base_opt,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.45,
        min_tracking_confidence=0.45)
    detector = vision.HandLandmarker\
                     .create_from_options(opts)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Webcam ready!\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        h, w    = frame.shape[:2]
        gesture = "none"

        rgb    = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb)
        result = detector.detect(mp_img)

        if result.hand_landmarks:
            lm      = result.hand_landmarks[0]
            gesture = detect_gesture(lm)

            # Track index fingertip for rotation
            cx = int(lm[8].x * w)
            cy = int(lm[8].y * h)

            frame = draw_hand(frame, lm)

            # ── INDEX: pan up/down/left/right ──
            if gesture == "index_only":
                if prev_cx is not None:
                    dx = cx - prev_cx
                    dy = cy - prev_cy
                    if abs(dx) > 4 or abs(dy) > 4:
                        sent = send_cmd(
                            "pan",
                            {"dx": int(dx),
                             "dy": int(dy)})
                        if sent:
                            print(
                                f"Pan "
                                f"dx={dx:+d} "
                                f"dy={dy:+d}")

            # ── PINCH: zoom in ─────────────────
            elif gesture == "pinch":
                sent = send_cmd(
                    "zoom", {"factor": 1.08})
                if sent:
                    print("Zoom IN")

            # ── VICTORY: toggle segments ───────
            elif gesture == "victory":
                if last_gesture != "victory":
                    if seg_mode == "a":
                        send_cmd(
                            "show_option_b",
                            force=True)
                        seg_mode = "b"
                        print("All Segments")
                    else:
                        send_cmd(
                            "show_option_a",
                            force=True)
                        seg_mode = "a"
                        print("Liver+Tumor")

            # ── 4 FINGERS: zoom out ────────────
            elif gesture == "four_fingers":
                sent = send_cmd(
                    "zoom", {"factor": 0.93})
                if sent:
                    print("Zoom OUT")

            # ── THUMB UP: reset all ────────────
            elif gesture == "thumb_up":
                if last_gesture != "thumb_up":
                    send_cmd(
                        "reset_all",
                        force=True)
                    seg_mode = "a"
                    print("RESET ALL")

            prev_cx      = cx
            prev_cy      = cy
            last_gesture = gesture

        else:
            prev_cx      = None
            prev_cy      = None
            last_gesture = "none"

        frame = draw_ui(frame, gesture)
        # Resize window to small size
        small = cv2.resize(frame, (520, 390))
        cv2.imshow('Gesture Control', small)
        cv2.moveWindow('Gesture Control', 10, 10)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Stopped")

if __name__ == "__main__":
    main()