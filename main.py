import cv2
import mediapipe as mp
import math
import numpy as np
import time
from pynput.keyboard import Key, Controller

# --- 1. Configuration ---
# Camera Setup
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# MediaPipe Setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Keyboard Controller
keyboard = Controller()

# --- Calibration Constants ---
# Windows volume usually moves in steps of 2 (e.g., 0, 2, 4... 100)
# We assume 50 steps total to get from 0 to 100.
VOL_STEP = 2 
current_virtual_vol = 0 # We track the volume level in this variable

# Hand Range (The "Container" Size)
MIN_DIST = 30   # Fingers touching
MAX_DIST = 180  # Fingers stretched

print("Initializing... Calibrating volume to 0 (Please wait 2 seconds)")

# --- 2. Initial Calibration (Empty the Container) ---
# We spam Volume Down to ensure we start at 0
for _ in range(50):
    keyboard.press(Key.media_volume_down)
    keyboard.release(Key.media_volume_down)
    time.sleep(0.01)

print("Calibration Complete. System Active.")
print("Press 'q' to exit.")

# --- 3. Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    target_vol = current_virtual_vol # Default to staying put
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            if lmList:
                # 1. Get Finger Coordinates (Thumb & Index)
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # 2. visual Aids
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                # 3. Calculate Distance
                length = math.hypot(x2 - x1, y2 - y1)
                
                # 4. Map Distance to Target Volume (0 to 100)
                # This creates the "Container" effect.
                # If length is high (apart), target is 100. If low (close), target is 0.
                target_vol = np.interp(length, [MIN_DIST, MAX_DIST], [0, 100])
                
                # Snap to nearest step of 2 to match Windows steps
                target_vol = round(target_vol / 2) * 2

                # 5. Smoothly Move Volume towards Target
                # "Speed" Control: We only press the key once per frame if needed.
                
                # If our tracked volume is LOWER than your finger position -> Increase
                if current_virtual_vol < target_vol:
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    current_virtual_vol += VOL_STEP
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED) # Green feedback
                    
                # If our tracked volume is HIGHER than your finger position -> Decrease
                elif current_virtual_vol > target_vol:
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
                    current_virtual_vol -= VOL_STEP
                    cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED) # Red feedback
                    
                else:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED) # Stable

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # Display Text for current volume
    cv2.putText(img, f'Vol: {int(current_virtual_vol)}%', (40, 50), 
                cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Gesture Volume Control", img)
    
    # Speed Control Sleep
    # Increase this number (e.g. 0.05 or 0.1) to make the volume change SLOWER
    # Decrease this number (e.g. 0.001) to make it FASTER
    time.sleep(0.02) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()