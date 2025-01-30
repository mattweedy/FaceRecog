# FaceRecog - Facial Recognition and Attendance System
![Python](https://img.shields.io/badge/Python-3.9-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![face_recognition](https://img.shields.io/badge/face__recognition-1.x-orange)
![Multithreading](https://img.shields.io/badge/Multithreading-Enabled-brightgreen)

A real-time facial recognition and attendance system built using Python, OpenCV, and the `face_recognition` library. This project leverages multithreading to improve performance and responsiveness.
___

## Features

- **Real-Time Face Detection**: Detects faces in real-time using a webcam.
- **Face Recognition**: Recognizes known faces and marks attendance.
- **Multithreading**: Uses separate threads for video capture, processing, and display to improve performance.
- **Attendance Logging**: Logs attendance with timestamps in a CSV file.
- **Customizable**: Easily add new faces to the system by placing images in the `ImagesWebcam` folder.
___

## Requirements

- Python 3.9 or higher
- OpenCV (`opencv-python`)
- `face_recognition` library
- NumPy
- A webcam
___

## Installation

1. **Clone the repository**:
  ```bash
  git clone https://github.com/your-username/your-repo-name.git
  cd your-repo-name
   ```
2. **Set up and activate virtual environment**:
  ```bash
  python -m venv venv
  ```
  Ensure the virtual environment includes `cmake` in its PATH:
  - add `$env:PATH = "C:\Program Files\CMake\bin;" + $env:PATH` at the end of `\venv\Scripts\Activate.ps1` (Windows)
  Then run the virtual environment:
  ```bash
  .\venv\Scripts\Activate.ps1
  ```
3. **Install dependencies**:
  ```bash
  pip install -r requirements.txt
  ```
4. **Add known faces**:
  - Place images of known faces in the `ImagesWebcam` folder.
  - The file name should be the person's name (e.g., `John_Doe.jpg`).
___
## Usage
1. **Run the program**:
  ```bash
  python WebcamAttendance.py
  ```
2. **Interact with the system**:
  - The webcam feed will open, and the system will detect and recognize faces in real-time.
  - Recognized faces will be logged in Logs/Attendance.csv.
3. **Exit the program**:
  - Press `q` to stop the program.
___
## Folder Structure
```
.
├── ImagesWebcam/          # Folder containing images of known faces
├── Logs/                  # Folder for attendance logs
├── FrameProcessor.py      # Class for processing video frames
├── VideoGrab.py           # Class for capturing video frames
├── VideoShow.py           # Class for displaying video frames
├── WebcamAttendance.py    # Main script for the attendance system
├── requirements.txt       # List of dependencies
└── README.md              # This file
```
___
## Performance Tips
  - Downscale frames: Reduce the resolution of frames for faster processing.
  - Skip frames: Process every Nth frame to reduce CPU load.
  - Use HOG model: The HOG model is faster than CNN for face detection.
  - Enable hardware acceleration: Rebuild OpenCV with OpenCL or CUDA support for GPU acceleration.
___
## Acknowledgments
  - [face_recognition](https://github.com/ageitgey/face_recognition) by Adam Geitgey.
  - [OpenCV](https://opencv.org/) for real-time video processing.
  - [Multithreading with OpenCV](https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/) by Nisarg Shah.
