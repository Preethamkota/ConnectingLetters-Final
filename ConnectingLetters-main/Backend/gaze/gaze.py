import mediapipe as np
from collections import deque
import cv2

class GazeSmoother:
    def __init__(self,window_size=5):
        self.buffer = deque(maxlen=window_size)

    def update(self,value):
        self.buffer.append(value)
        return sum(self.buffer)/len(self.buffer)
    
smoother=GazeSmoother(window_size=5)

def compute_eye_ratio(landmarks,left_idx,right_idx,pupil_idx):
    left=landmarks[left_idx]
    right=landmarks[right_idx]
    pupil=landmarks[pupil_idx]

    width=right.x-left.x
    if abs(width)<1e-6:
        return 0.5
    
    ratio = (pupil.x - left.x)/width
    return max(0.0,min(1.0,ratio))

def compute_gaze_ratio(landmarks):
    l_ratio = compute_eye_ratio(landmarks,33,133,468)
    r_ratio = compute_eye_ratio(landmarks,362,263,473)
    return (l_ratio+r_ratio)/2.0

def get_head_pose(landmarks, frame_shape):
    h, w, _ = frame_shape

    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),     # Nose tip
        (landmarks[152].x * w, landmarks[152].y * h), # Chin
        (landmarks[33].x * w, landmarks[33].y * h),   # Left eye
        (landmarks[263].x * w, landmarks[263].y * h), # Right eye
        (landmarks[61].x * w, landmarks[61].y * h),   # Left mouth
        (landmarks[291].x * w, landmarks[291].y * h)  # Right mouth
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),        
        (0.0, -330.0, -65.0),   
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), 
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0) 
    ])

    focal_length = w
    center = (w / 2, h / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0, 0

    rmat, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch, yaw, roll = angles

    return yaw, pitch


def is_focused(gaze_ratio,yaw,pitch,gaze_thresh=0.15,yaw_thresh=20,pitch_thresh=15):
    eye_centered = abs(gaze_ratio-0.5)<gaze_thresh
    head_forward = abs(yaw) < yaw_thresh and abs(pitch) < pitch_thresh

    return 1 if (eye_centered and head_forward) else 0

def get_gaze(landmarks,frame_shape):
    raw_ratio = compute_gaze_ratio(landmarks)
    smooth_ratio=smoother.update(raw_ratio)

    yaw,pitch = get_head_pose(landmarks,frame_shape)
    focus=is_focused(smooth_ratio,yaw,pitch)

    return {
        "gaze_ratio":float(smooth_ratio),
        "yaw":float(yaw),
        "pitch":float(pitch),
        "focused":int(focus)
    }