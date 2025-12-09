import cv2
import mediapipe as mp
import math
import numpy as np
import time
from pynput.keyboard import Key, Controller

# --- 1. Configuration ---
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

keyboard = Controller()

# --- 2. Virtual Slider Settings ---
# We track the volume state internally
current_virtual_vol = 100 
VOL_STEP = 2   # Windows usually changes volume by 2% per press

# Calibration: Hand Range
MIN_DIST = 30
MAX_DIST = 180

print("Initializing... Setting volume to 100% for calibration.")

# --- 3. Calibration (Start at 100%) ---
# We force the volume to 100% so our internal counter matches reality
for _ in range(50):
    keyboard.press(Key.media_volume_up)
    keyboard.release(Key.media_volume_up)
    time.sleep(0.005)

print("Calibration Complete. System Active.")

# --- 4. Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        break
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # Visual Bar Default (Show current state if no hand)
    bar_height = np.interp(current_virtual_vol, [0, 100], [400, 150])
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            if lmList:
                x1, y1 = lmList[4][1], lmList[4][2] 
                x2, y2 = lmList[8][1], lmList[8][2] 
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Visuals
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                # 1. Calculate Target based on Hand Distance
                length = math.hypot(x2 - x1, y2 - y1)
                
                # Map range 30-180 to 0-100
                raw_target = np.interp(length, [MIN_DIST, MAX_DIST], [0, 100])
                
                # Snap to nearest 2 (to match Windows steps)
                target_vol = round(raw_target / VOL_STEP) * VOL_STEP
                
                # 2. Update Visual Bar IMMEDIATELY (Responsiveness)
                # We update the bar based on where your hand IS, not just where the volume is.
                bar_height = np.interp(target_vol, [0, 100], [400, 150])
                
                # 3. Control Logic (Chase the Target)
                diff = target_vol - current_virtual_vol
                
                # Only press keys if we are out of sync
                if abs(diff) >= VOL_STEP:
                    if diff > 0:
                        keyboard.press(Key.media_volume_up)
                        keyboard.release(Key.media_volume_up)
                        current_virtual_vol += VOL_STEP
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED) # Green
                    elif diff < 0:
                        keyboard.press(Key.media_volume_down)
                        keyboard.release(Key.media_volume_down)
                        current_virtual_vol -= VOL_STEP
                        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED) # Red
                    
                    # Very small sleep to prevent system lag, but kept fast
                    time.sleep(0.01) 

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # --- Draw UI ---
    # Background Bar
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    
    # Active Bar (Uses the responsive bar_height calculated above)
    cv2.rectangle(img, (50, int(bar_height)), (85, 400), (0, 255, 0), cv2.FILLED)
    
    # Text
    cv2.putText(img, f'{int(current_virtual_vol)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Gesture Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()