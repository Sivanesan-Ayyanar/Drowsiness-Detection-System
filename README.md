# Drowsiness Detection System

This project is designed to detect drowsiness in real-time using a webcam. It uses the Eye Aspect Ratio (EAR) to determine if a person is drowsy and triggers an alarm if the EAR falls below a certain threshold. The project utilizes OpenCV for video capture and processing, dlib for facial landmark detection, scipy for calculating distances, and pygame for playing an alarm sound.

## Features
- Real-time drowsiness detection using a webcam.
- Calculates Eye Aspect Ratio (EAR) to detect drowsiness.
- Plays an alarm sound when drowsiness is detected.

## Requirements
- Python 3.x
- OpenCV
- dlib
- scipy
- pygame

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/drowsiness-detection.git
    cd drowsiness-detection
    ```

2. Install the required packages:
    ```bash
    pip install opencv-python dlib scipy pygame
    ```

3. Download the pre-trained facial landmark predictor model from [dlib's model zoo](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and extract it to the project directory.

## Usage
1. Run the `drowsiness_detection.py` script:
    ```bash
    python drowsiness_detection.py
    ```

2. The webcam feed will start, and the system will begin detecting faces and monitoring the Eye Aspect Ratio (EAR).

3. If the EAR falls below the threshold of 0.23, an alarm sound will be played, and a warning message will be displayed on the screen.

## Code Explanation
### calculate_EAR
This function calculates the Eye Aspect Ratio (EAR) for a given eye.
```python
def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A + B) / (2.0 * C)
    return ear_aspect_ratio
