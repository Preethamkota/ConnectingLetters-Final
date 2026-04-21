import os
import cv2 as cv
import mediapipe as mp
import csv
import random
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
            cz = sum(lm.z for lm in landmarks)/len(landmarks)

            scale=max(
                ((lm.x - cx)**2 + (lm.y - cy)**2)**0.5 for lm in landmarks
            )

            if scale==0:
                continue

            row=[]
            for lm in landmarks:
                row.extend([
                    (lm.x - cx)/scale,
                    (lm.y-cy)/scale,
                    (lm.z-cz)/scale
                ])
            row.append(label_map[label])
            data.append(row)
        else:
            missed+=1

random.shuffle(data)

header=[]
for i in range(468):
    header+=[f"x_{i}",f"y_{i}",f"z_{i}"]
header.append("label")

with open(OUTPUT_FILE,"w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

face_mesh.close()

print("\n✅ Done!")
print(f"Saved: {len(data)}")
print(f"Missed: {missed}/{total}")