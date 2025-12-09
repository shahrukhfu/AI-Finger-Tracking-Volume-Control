import cv2
import mediapipe as mp
import math
import numpy as np
import time
from pynput.keyboard import Key, Controller

# --- 1. Configuration & Setup ---

# Camera Setup
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# MediaPipe Hands Setup
mpHands = mp.solutions.hands
# max_num_hands=1 ensures we only detect one hand for cleaner control
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Pynput Volume Control Setup
keyboard = Controller()

# Define thresholds for triggering volume change (Calibrate these to your hand size!)
VOLUME_DOWN_THRESHOLD = 40  # If length < 40, Volume Down is triggered
VOLUME_UP_THRESHOLD = 150   # If length > 150, Volume Up is triggered

# Cooldown to prevent rapid-fire key presses
last_vol_change_time = time.time()
COOLDOWN_TIME = 0.15 # Minimum time between volume presses (in seconds)

# Variables for visualization
volBar = 400
volPer = 0

print("System initialized. Use finger distance to control volume. Press 'q' to exit.")

# --- 2. Main Loop ---
while True:
    success, img = cap.read()
    if not success:
        print("Failed to grab frame")
        break
    
    # Convert image to RGB (MediaPipe requires RGB input)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    current_time = time.time()
    
    # Check if any hands are detected
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            
            # List to store landmark coordinates
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            
            # If we have landmarks, proceed
            if lmList:
                # Get coordinates for Thumb Tip (4) and Index Tip (8)
                # [index][x-coord][y-coord]
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                
                # Calculate the center point between fingers (for visual aesthetics)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Draw circles on tips and a line between them
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED) # Thumb
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED) # Index
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                # Calculate Length (Euclidean Distance)
                length = math.hypot(x2 - x1, y2 - y1)
                
                # Update visual metrics (based on length mapping from 20 to 180 pixels)
                volBar = np.interp(length, [20, 180], [400, 150])
                volPer = np.interp(length, [20, 180], [0, 100])
                
                # --- Volume Control Logic using Pynput ---
                if (current_time - last_vol_change_time) > COOLDOWN_TIME:
                    
                    # Volume Down (Fingers pinching together)
                    if length < VOLUME_DOWN_THRESHOLD:
                        keyboard.press(Key.media_volume_down)
                        keyboard.release(Key.media_volume_down)
                        last_vol_change_time = current_time
                        cv2.circle(img, (cx, cy), 15, (0, 0, 255), cv2.FILLED) # Red for down

                    # Volume Up (Fingers spreading wide)
                    elif length > VOLUME_UP_THRESHOLD:
                        keyboard.press(Key.media_volume_up)
                        keyboard.release(Key.media_volume_up)
                        last_vol_change_time = current_time
                        cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED) # Green for up
                    
                    # Neutral center point
                    else:
                        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
                
                # If currently in cooldown, just draw the neutral point
                else:
                    cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)


            # Draw standard hand landmarks
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    # --- 3. Draw UI on Screen ---
    # Draw Volume Bar background
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    # Draw Volume Level
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    # Draw Percentage Text
    cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 3)
    
    cv2.imshow("Gesture Volume Control", img)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()