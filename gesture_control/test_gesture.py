import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import math

MODEL_PATH = r"./hand_landmarker.task"

base_opt = python.BaseOptions(model_asset_path=MODEL_PATH)
opts = vision.HandLandmarkerOptions(
    base_options=base_opt,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4)
detector = vision.HandLandmarker.create_from_options(opts)

cap = cv2.VideoCapture(0)
print("Show your hand — press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(
        image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    if result.hand_landmarks:
        lm    = result.hand_landmarks[0]
        idx   = lm[8].y  < lm[5].y
        mid   = lm[12].y < lm[9].y
        rng   = lm[16].y < lm[13].y
        pnk   = lm[20].y < lm[17].y
        pdx   = lm[4].x  - lm[8].x
        pdy   = lm[4].y  - lm[8].y
        pinch = math.sqrt(pdx*pdx + pdy*pdy) < 0.08
        thumb = (lm[4].y < lm[5].y and
                 lm[4].y < lm[0].y)
        print(f"idx={int(idx)} mid={int(mid)} "
              f"rng={int(rng)} pnk={int(pnk)} "
              f"pinch={int(pinch)} thumb={int(thumb)}")
        cv2.imshow("Test", frame)
    else:
        print("No hand detected")
        cv2.imshow("Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()