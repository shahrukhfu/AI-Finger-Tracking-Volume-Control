import cv2
import mediapipe as mp
import math
import numpy as np
import time
from pynput.keyboard import Key, Controller

# --- 1. Setup ---
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

keyboard = Controller()

# --- 2. Virtual Slider Config ---
# We track volume internally starting at 100 now
current_virtual_vol = 100 
VOL_STEP = 2  # Windows volume usually jumps by 2 per key press

# Hand Distance Range
MIN_DIST = 30
MAX_DIST = 180

print("Initializing... Setting volume to 100% for calibration.")

# --- 3. Initial Calibration (Set System Volume to 100) ---
# Spam 'Volume Up' 50 times to guarantee we start at 100%
for _ in range(50):
    keyboard.press(Key.media_volume_up)
    keyboard.release(Key.media_volume_up)
    time.sleep(0.005)

print("Calibration Done. System Active at 100%.")

# --- 4. Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        break
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    target_vol = current_virtual_vol 
    
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
                
                # Calculate Distance
                length = math.hypot(x2 - x1, y2 - y1)
                
                # Map Distance to 0-100 scale
                raw_target = np.interp(length, [MIN_DIST, MAX_DIST], [0, 100])
                target_vol = round(raw_target / VOL_STEP) * VOL_STEP
                
                # --- The Slider Logic ---
                diff = target_vol - current_virtual_vol
                
                if abs(diff) >= VOL_STEP:
                    if diff > 0:
                        keyboard.press(Key.media_volume_up)
                        keyboard.release(Key.media_volume_up)
                        current_virtual_vol += VOL_STEP
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED) 
                        
                    elif diff < 0:
                        keyboard.press(Key.media_volume_down)
                        keyboard.release(Key.media_volume_down)
                        current_virtual_vol -= VOL_STEP
                        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED) 
                        
                    time.sleep(0.02) 

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # --- Draw UI ---
    volBar = np.interp(current_virtual_vol, [0, 100], [400, 150])
    
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(current_virtual_vol)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Gesture Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()