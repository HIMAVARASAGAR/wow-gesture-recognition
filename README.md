# WOW Hand Gesture Recognition

This project implements recognition of a **WOW / OK hand gesture** using a hybrid  
**Deep Learning + Classical Computer Vision** approach.

The system detects a hand gesture where the **thumb and index finger form a circular loop**
and the remaining fingers are extended. The gesture is visualized using
**circles and lines derived from hand landmarks**.

---

## Project Overview

The objective of this project is to demonstrate how **semantic landmark detection**
can be combined with **geometric shape analysis** for robust hand gesture recognition.

Instead of relying purely on end-to-end gesture classification, the project separates
the problem into two interpretable stages:
- Detecting **where** hand joints are located
- Detecting **what geometric shapes** those joints form

This design makes the system lightweight, explainable, and suitable for real-time systems.

---

## Methodology

### 1. Hand Landmark Detection (Deep Learning)

MediaPipe **Hand Landmarker** is used to detect **21 key hand landmarks** from the input image.
Each landmark corresponds to a specific joint or fingertip, providing precise structural
information about the hand.

### 2. Geometric Structure Formation

Using the detected landmarks:
- The **thumb–index loop** is represented using landmark points along the circular arc
- **Extended fingers** are represented as straight line segments between relevant joints

### 3. Shape Detection (Classical Computer Vision)

- **Hough Circle Transform** is applied to detect the circular loop formed by the thumb and index finger
- **Linear structures** are derived from landmark-based finger connections
- The final visualization overlays:
  - A circle around the thumb–index loop
  - Lines representing extended fingers

This satisfies the requirement of detecting **circles and lines formed by landmarks**.

---

## Technologies Used

- Python  
- MediaPipe Hand Landmarker  
- OpenCV  
- NumPy  

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt

### 2. Run the Script
```bash
python wow_gesture.py

### 3.Output 
The script processes the input image and generates an output image highlighting:
- The circular loop formed by the thumb and index finger
- Linear structures corresponding to extended fingers

### Future Work

This project can be extended in several directions, including:
Real-time gesture recognition using a live webcam feed
- Recognition of multiple hand gestures such as OK, Thumbs Up, Peace, Fist, etc.
- Temporal gesture recognition using video streams
- Integration with human–computer interaction systems, touchless controls, or AR/VR interfaces

The current landmark-based geometric pipeline is already suitable for real-time execution.


## Key Idea

Deep learning detects where things are.
Classical computer vision explains what shape they form.

This project combines both approaches to achieve accurate and interpretable hand gesture recognition.
