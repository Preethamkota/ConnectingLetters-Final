import os
import cv2 as cv
import mediapipe as mp
import csv
import random
import numpy as np

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)

DATASET_PATH=r"C:\Users\preet\Downloads\ConnectingLettersFull\ConnectingLetters-main\Backend\data\train3"
OUTPUT_FILE = r"C:\Users\preet\Downloads\ConnectingLettersFull\ConnectingLetters-main\Backend\preprocessed\landmarks.csv"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

label_map={
    "angry":0,
    "fear":1,
    "happy":2,
    "neutral":3,
    "sad":4,
    "surprise":5
}

data=[]
total=0
missed=0

def dist(a,b):
    return np.linalg.norm(np.array(a)-np.array(b))

for label in os.listdir(DATASET_PATH):
    label_path=os.path.join(DATASET_PATH,label)

    if not os.path.isdir(label_path) or label not in label_map:
        continue

    print(f"processing {label}")

    for img_name in os.listdir(label_path):
        if not img_name.lower().endswith(('.png','.jpg','.jpeg')):
            continue

        img_path=os.path.join(label_path,img_name)
        total+=1

        img=cv.imread(img_path)
        if img is None:
            continue

        img=cv.resize(img,(224,224))

        img_rgb=cv.cvtColor(img,cv.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            landmarks=results.multi_face_landmarks[0].landmark
            cx = sum(lm.x for lm in landmarks)/len(landmarks)
            cy= sum(lm.y for lm in landmarks)/len(landmarks)

            scale=max(
                ((lm.x - cx)**2 + (lm.y - cy)**2)**0.5 for lm in landmarks
            )

            if scale< 1e-6:
                continue

            row=[]
            norm_points=[]
            for lm in landmarks:
                row.extend([
                    (lm.x - cx)/scale,
                    (lm.y-cy)/scale
                ])
                norm_points.append([
                    (lm.x-cx)/scale,
                    (lm.y-cy)/scale
                ])
            try:
                left_eye = norm_points[33]
                right_eye = norm_points[263]
                mouth_left = norm_points[61]
                mouth_right = norm_points[291]
                upper_lip = norm_points[13]
                lower_lip = norm_points[14]
                upper_eye = norm_points[159]
                lower_eye=norm_points[145]
                left_brow = norm_points[70]
                right_brow = norm_points[300]
                eye_top = norm_points[159]
                left_mouth = norm_points[61]
                right_mouth = norm_points[291]
                top_mouth = norm_points[13]

                mid_x = (left_mouth[0] + right_mouth[0]) / 2
                mid_y = (left_mouth[1] + right_mouth[1]) / 2
                mouth_curve = top_mouth[1] - mid_y 


                eye_dist = dist(left_eye,right_eye)
                mouth_width=dist(mouth_left,mouth_right)
                mouth_open=dist(upper_lip,lower_lip)
                eye_open = dist(upper_eye, lower_eye)

                ratio=mouth_width/(eye_dist+ 1e-6)

                brow_left_dist = dist(left_brow, eye_top)
                brow_right_dist = dist(right_brow, eye_top)

                mouth_height = mouth_open
                mouth_ratio2 = mouth_width / (mouth_height + 1e-6)

                left_eye_open = eye_open
                right_eye_open = dist(norm_points[386], norm_points[374])
                eye_symmetry = abs(left_eye_open - right_eye_open)

                row.extend([eye_dist,mouth_width,mouth_open,ratio,eye_open])
                row.extend([brow_left_dist, brow_right_dist])
                row.append(mouth_ratio2)
                row.append(eye_symmetry)
                row.append(mouth_curve)
            except Exception:
                missed+=1
                continue
            row.append(label_map[label])
            data.append(row)
        else:
            missed+=1

random.shuffle(data)

header=[]
for i in range(468):
    header+=[f"x_{i}",f"y_{i}"]
header += [
    "eye_dist",
    "mouth_width",
    "mouth_open",
    "mouth_ratio",
    "eye_open",
    "brow_left_dist",
    "brow_right_dist",
    "mouth_ratio2",
    "eye_symmetry",
    "mouth_curve"
]
header.append("label")

with open(OUTPUT_FILE,"w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

face_mesh.close()

print("\n Preprocessingg done")
print(f"Saved: {len(data)}")
print(f"Missed: {missed}/{total}")