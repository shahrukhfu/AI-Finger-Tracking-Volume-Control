import cv2
import mediapipe as mp
import math
import numpy as np
import time
from pynput.keyboard import Key, Controller

# --- 1. Configuration & Setup ---
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# MediaPipe Setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Pynput Keyboard Controller
keyboard = Controller()

# --- Calibration Constants (Virtual Slider) ---
VOL_STEP = 2             # Volume changes by 2% per key press
current_virtual_vol = 0  # Program's internal tracking of volume (0-100)

MIN_DIST = 30            # Fingers touching (0% Volume)
MAX_DIST = 180           # Fingers stretched (100% Volume)

# Speed Control: Increase this to slow down the volume change rate
SLIDER_DELAY = 0.035 

print("Initializing... Calibrating volume to 0 (Please wait).")

# --- 2. Initial Calibration (Set System Volume to 0) ---
# Sends 50 VolDown key presses to ensure we sync with 0%
for _ in range(50):
    keyboard.press(Key.media_volume_down)
    keyboard.release(Key.media_volume_down)
    time.sleep(0.01)

print("Calibration Complete. Switched to Virtual Slider Mode.")

# --- 3. Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
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
                
                # Map Distance to Target Volume (0-100)
                target_vol = np.interp(length, [MIN_DIST, MAX_DIST], [0, 100])
                target_vol = round(target_vol / VOL_STEP) * VOL_STEP # Snap to nearest 2% step

                # --- Virtual Slider Logic ---
                
                # Check how much we need to adjust
                difference = int(target_vol - current_virtual_vol) 

                if abs(difference) >= VOL_STEP:
                    
                    if difference > 0:
                        # Volume Up needed
                        keyboard.press(Key.media_volume_up)
                        keyboard.release(Key.media_volume_up)
                        current_virtual_vol += VOL_STEP
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED) # Green
                        
                    elif difference < 0:
                        # Volume Down needed
                        keyboard.press(Key.media_volume_down)
                        keyboard.release(Key.media_volume_down)
                        current_virtual_vol -= VOL_STEP
                        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED) # Red

                    # Apply delay for smooth speed control
                    time.sleep(SLIDER_DELAY) 
                    
                else:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # --- Draw UI (Based on Virtual Volume) ---
    volBar = np.interp(current_virtual_vol, [0, 100], [400, 150])
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(current_virtual_vol)}%', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow("Gesture Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()