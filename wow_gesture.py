import cv2
import numpy as np
import mediapipe as mp
import math

# ================= CONFIG =================
MODEL_PATH = "hand_landmarker.task"
IMAGE_PATH = "input.jpg"
OUTPUT_PATH = "output_wow_fixed.jpg"
# ==========================================

# 1. Load image
try:
    mp_image = mp.Image.create_from_file(IMAGE_PATH)
    image = cv2.imread(IMAGE_PATH)
except:
    print(f"Error: Could not find {IMAGE_PATH}")
    exit()

h, w, _ = image.shape

# 2. MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1
)

with HandLandmarker.create_from_options(options) as landmarker:
    result = landmarker.detect(mp_image)

if not result.hand_landmarks:
    print("No hand detected")
    exit()

pts = [(int(lm.x * w), int(lm.y * h)) for lm in result.hand_landmarks[0]]

# ==================================================
# PART 1: GEOMETRY CALCULATION (IMPROVED)
# ==================================================
# The "O" shape is formed by the thumb and index finger creating a loop
# The CENTER should be in the middle of the HOLE, not at the fingertips
# Key landmarks:
# 2 = Thumb second joint (inside the loop)
# 3 = Thumb third joint (inside the loop)
# 4 = Thumb tip
# 5 = Index MCP (knuckle, inside the loop)
# 6 = Index PIP (middle joint, inside the loop)
# 8 = Index tip

p2 = np.array(pts[2])  # Thumb second joint
p3 = np.array(pts[3])  # Thumb third joint
p5 = np.array(pts[5])  # Index MCP (knuckle)
p6 = np.array(pts[6])  # Index PIP

# The center of the loop is between the INNER points (not the tips)
# Use the thumb inner joints (2,3) and index inner joint (5)
# Add slight influence from point 6 (index middle joint) for better alignment
cx = int((p2[0] + p3[0] + p5[0] + p6[0]) / 4)
cy = int((p2[1] + p3[1] + p5[1] + p6[1]) / 4)

# Now calculate the radius based on how far the fingertips are from this center
p4 = np.array(pts[4])  # Thumb tip
p8 = np.array(pts[8])  # Index tip

distances = [
    np.linalg.norm(p4 - np.array([cx, cy])),
    np.linalg.norm(p8 - np.array([cx, cy])),
    np.linalg.norm(p3 - np.array([cx, cy])),
    np.linalg.norm(p6 - np.array([cx, cy]))
]
radius = int(np.mean(distances))  # Use average for a balanced circle

# ==================================================
# PART 2: HOUGH TRANSFORM (Assignment Requirement)
# ==================================================
# Create a mask with the geometric circle
mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask, (cx, cy), radius, 255, 3)

# Edge detection
edges = cv2.Canny(mask, 50, 150)

# Detect circles using Hough Transform
circles = cv2.HoughCircles(
    edges,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=100,
    param2=10, 
    minRadius=int(radius * 0.7),
    maxRadius=int(radius * 1.3)
)

# ==================================================
# PART 3: DRAWING
# ==================================================

# Draw Circle (Blue)
if circles is not None:
    # Use the Hough-detected circle
    circles = np.uint16(np.around(circles))
    det_x, det_y, det_r = circles[0, 0]
    cv2.circle(image, (det_x, det_y), det_r, (255, 0, 0), 4)
    print(f"✓ Hough Circle detected at ({det_x}, {det_y}) with radius {det_r}")
else:
    # Fallback to geometric calculation
    cv2.circle(image, (cx, cy), radius, (255, 0, 0), 4)
    print(f"✓ Geometric Circle drawn at ({cx}, {cy}) with radius {radius}")

# Draw Lines on Extended Fingers (Green)
# These connect the knuckle (MCP) to the fingertip
finger_lines = [
    (9, 12),   # Middle finger
    (13, 16),  # Ring finger
    (17, 20)   # Pinky finger
]

for start, end in finger_lines:
    cv2.line(image, pts[start], pts[end], (0, 255, 0), 4)

# Save result
cv2.imwrite(OUTPUT_PATH, image)
print(f"✅ Output saved to: {OUTPUT_PATH}")
print(f"   Circle center: ({cx}, {cy})")
print(f"   Circle radius: {radius}")