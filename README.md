# 🖐️ AirSketch : AI-Based Touchless Drawing System Using Hand Gestures 🎨


## 📖 Overview

AirSketch is an advanced **AI-powered touchless drawing system** that allows users to draw in the air using **hand gestures**.It leverages **MediaPipe Hands** for real-time hand tracking and **OpenCV** for rendering a virtual drawing canvas.This project demonstrates the integration of **gesture recognition**, **real-time video processing**, and **interactive UI design**, making it an excellent example of **Human–Computer Interaction (HCI)** applications.


## 🛠 Technical Stack

* **Python 3.7+** – Core programming language 🐍
* **OpenCV 4.5+** – Computer vision library for image processing and rendering 📷
* **MediaPipe 0.8.9+** – Machine learning framework for real-time hand tracking ✋
* **NumPy 1.19+** – Efficient numerical computing library 📊


## 🌟 Key Features

1. **Real-Time Hand Tracking** – Tracks 21 3D hand landmarks at 30+ FPS.
2. **Gesture Recognition** – Detects raised index finger to start/stop drawing.
3. **Dynamic Color Selection** – On-screen color palette for instant color switching.
4. **Eraser Mode** – Activate by raising index + middle finger.
5. **Adaptive Line Drawing** – Minimizes jitter using distance-based point sampling.
6. **Clear Canvas** – Touch the **CLEAR** button to reset the canvas.
7. **Save Drawing** – Press `S` to save your artwork as a PNG image.
8. **Brush Size Adjustment** – Increase or decrease line thickness with `+` / `-`.
9. **Optimized Performance** – Reduced resolution and efficient drawing algorithms for smooth experience.


## 🏗 System Architecture

The project follows a **modular architecture** for clarity and extensibility:

1. **Input Module** – Captures webcam frames and prepares them for processing.
2. **Hand Detection Module** – Uses MediaPipe to detect and track hand landmarks.
3. **Gesture Recognition Module** – Determines finger positions to decide drawing mode.
4. **Drawing Module** – Updates the canvas with lines or eraser strokes.
5. **UI Module** – Renders color palette, clear button, and other interface elements.
6. **Output Module** – Combines webcam feed, canvas, and UI into a final display frame.


## 🔍 Key Algorithms

### Hand Landmark Detection

Uses MediaPipe’s hand detection and landmark models to identify **21 3D points** of a hand in real time.

### Index Finger Raise Detection

```python
def is_index_finger_raised(hand_landmarks):
    return hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
```

This function compares the **index fingertip** (landmark 8) to the **middle knuckle** (landmark 6) to determine if the finger is raised.

### Adaptive Line Drawing

```python
if prev_point and np.linalg.norm(np.array(index_tip) - np.array(prev_point)) > min_distance:
    cv2.line(canvas, prev_point, index_tip, colors[colorIndex], line_thickness)
    prev_point = index_tip
```

Only draws when the finger moves a significant distance, ensuring smooth lines and reducing jitter.


## ⚡ Performance Considerations

1. **Frame Resolution**: Reduced to 640x480 for optimal balance between quality and speed.
2. **Detection Confidence**: Hand tracking confidence set to 0.5 for faster processing.
3. **Canvas Optimization**: Direct drawing and pre-rendered UI elements reduce per-frame computation.
4. **Lightweight Rendering**: Uses `cv2.addWeighted` for seamless blending of webcam feed and canvas.


## 🚀 Installation

1. Ensure **Python 3.7+** is installed.
2. Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

3. Clone the repository

## 📋 Usage

Run the application:

python AirSketchPro.py

**Controls & Gestures:**

| Action              | Gesture / Key               |
| ------------------- | --------------------------- |
| Draw                | ☝️ Index finger up          |
| Erase               | ✌️ Index + Middle finger up |
| Change Color        | Touch top color circles     |
| Clear Canvas        | Touch **CLEAR** button      |
| Save Drawing        | Press `S`                   |
| Increase Brush Size | Press `+`                   |
| Decrease Brush Size | Press `-`                   |
| Quit                | Press `Q`                   |


## 🔮 Future Enhancements

* Multi-hand support for collaborative drawing.
* AI-assisted gesture recognition and customizable gestures.
* 3D drawing using depth estimation techniques.
* Voice command integration for fully touchless control.
* Mobile optimization using TensorFlow Lite.

##  🤝 Contributing

Contributions are welcome!

1.Fork the repository.
2.Create a new branch (git checkout -b feature/YourFeature).
3.Commit your changes (git commit -m 'Add new feature').
4.Push to the branch (git push origin feature/YourFeature).
5.Submit a Pull Request.

## 🙏 Acknowledgments

* **MediaPipe Team** – For providing the hand-tracking solution.
* **OpenCV Contributors** – For their powerful computer vision tools.
* **NumPy Contributors** – For efficient numerical operations.

