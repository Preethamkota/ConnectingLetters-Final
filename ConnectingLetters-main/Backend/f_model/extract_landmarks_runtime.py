import mediapipe as mp

mp_face_mesh=mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

def get_landmarks(frame):
    landmarks=face_mesh.process(frame)

    if not landmarks.multi_face_landmarks:
        return None
    
    return landmarks.multi_face_landmarks[0].landmark