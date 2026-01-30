import cv2
import numpy as np
import mediapipe as mp
import math


MODEL_PATH = "hand_landmarker.task"
IMAGE_PATH = "input.jpg"
OUTPUT_PATH = "output_wow_fixed.jpg"

try:
    mp_image = mp.Image.create_from_file(IMAGE_PATH)
    image = cv2.imread(IMAGE_PATH)
except:
    print(f"Could not find {IMAGE_PATH}")
    exit()

h, w, _ = image.shape


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


p2 = np.array(pts[2])  
p3 = np.array(pts[3])  
p5 = np.array(pts[5])  
p6 = np.array(pts[6])  

cx = int((p2[0] + p3[0] + p5[0] + p6[0]) / 4)
cy = int((p2[1] + p3[1] + p5[1] + p6[1]) / 4)

p4 = np.array(pts[4])  
p8 = np.array(pts[8]) 

distances = [
    np.linalg.norm(p4 - np.array([cx, cy])),
    np.linalg.norm(p8 - np.array([cx, cy])),
    np.linalg.norm(p3 - np.array([cx, cy])),
    np.linalg.norm(p6 - np.array([cx, cy]))
]
radius = int(np.mean(distances))  

mask = np.zeros((h, w), dtype=np.uint8)
cv2.circle(mask, (cx, cy), radius, 255, 3)

edges = cv2.Canny(mask, 50, 150)

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

if circles is not None:
    circles = np.uint16(np.around(circles))
    det_x, det_y, det_r = circles[0, 0]
    cv2.circle(image, (det_x, det_y), det_r, (255, 0, 0), 4)
    
else:
    cv2.circle(image, (cx, cy), radius, (255, 0, 0), 4)
    
finger_lines = [
    (9, 12),   
    (13, 16),  
    (17, 20)   
]

for start, end in finger_lines:
    cv2.line(image, pts[start], pts[end], (0, 255, 0), 4)

cv2.imwrite(OUTPUT_PATH, image)
print(f" Output saved to: {OUTPUT_PATH}")