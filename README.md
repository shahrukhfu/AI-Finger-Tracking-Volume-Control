# AI Finger Tracking Volume Control

A computer vision application that enables touch-free system volume control using hand gestures. This project uses a webcam to track the distance between the user's thumb and index finger, mapping it to the system volume level in real-time.

## Description

This project leverages computer vision to create a virtual volume slider. By detecting the distance between specific hand landmarks, the application translates physical gestures into system volume commands. It utilizes a "container" analogy where bringing fingers together empties the volume (0%) and spreading them apart fills it (100%).

## Features

- **Real-Time Hand Tracking:** Uses MediaPipe to detect hand landmarks with high precision and low latency.
- **Gesture Control:** Intuitive pinch-to-zoom style gesture to control audio levels.
- **Virtual Slider Logic:** Implements an absolute control scheme (finger position = specific volume percentage) rather than relative steps.
- **Visual Feedback:** Displays a dynamic volume bar and percentage on the camera feed that responds instantly to hand movements.
- **Auto-Calibration:** Automatically synchronizes with the system volume upon startup to ensure accuracy.
- **Cross-Platform Compatibility:** Uses keyboard simulation (pynput) to control volume, avoiding specific audio driver conflicts common on Windows.

## Prerequisites

Before running the project, ensure you have Python installed. You will need a working webcam.

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/shahrukhfu/AI-Finger-Tracking-Volume-Control.git](https://github.com/shahrukhfu/AI-Finger-Tracking-Volume-Control.git)

2. Navigate to the project directory

    ```bash
    cd AI-Finger-Tracking-Volume-Control

3. Install the required dependencies

    ```bash
    pip install opencv-python mediapipe pynput numpy

## Usage

### Run the main script
    python main.py

## Calibration Phase

When the application starts, it will automatically force your system volume to **100%**. This is a necessary calibration step to synchronize the program's internal tracker with your computer's actual volume.

**Warning:**  
Please ensure you are not playing loud audio or wearing headphones during startup.

## Control

- **Decrease Volume:** Bring your thumb and index finger close together.  
- **Increase Volume:** Spread your thumb and index finger apart.  
- The on-screen bar will update to show your target setting.  
- **Exit:** Press `q` on your keyboard while the camera window is active to stop the application.

## How It Works

- **Detection:** The application captures video input using OpenCV and processes each frame with MediaPipe to find hand landmarks.  
- **Calculation:** It identifies the coordinates of the thumb tip (Landmark **4**) and index finger tip (Landmark **8**) and calculates the Euclidean distance between them.  
- **Mapping:** This pixel distance (calibrated between **30px** and **180px**) is mapped linearly to a volume percentage range (**0%** to **100%**).  
- **Actuation:** The program calculates the difference between the current volume and the target volume. It uses the `pynput` library to simulate the exact number of "Volume Up" or "Volume Down" key presses required to reach the target level.

## Troubleshooting

### Volume is out of sync

Since the program simulates key presses, manually changing the volume using your physical keyboard or mouse while the script is running may cause the program's internal counter to drift from the real system volume.

**Fix:**  
Simply bring your fingers fully apart (to **100%**) or fully together (to **0%**) to force the system to re-align.

### Camera not opening

Check if another application (like Zoom or Teams) is currently using the webcam.

Try changing the camera index in `main.py` (line 11):

    cap = cv2.VideoCapture(0)

Change `0` to `1` or `2`.

## License

This project is open source and available for modification and distribution.

