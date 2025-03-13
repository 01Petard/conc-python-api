import face_recognition
import pickle
import os


class FaceManager:
    def __init__(self, face_library_path='face_library.pkl'):
        self.face_library_path = face_library_path
        if os.path.exists(face_library_path):
            with open(face_library_path, 'rb') as f:
                self.face_library = pickle.load(f)
        else:
            self.face_library = {}

    def detect_faces(self, frame):
        face_locations = face_recognition.face_locations(frame)
        return face_locations

    def recognize_face(self, frame, face_location, pitch, yaw, frame_num):
        face_encoding = face_recognition.face_encodings(frame, [face_location])[0]
        name = "Unknown Person"
        for known_name, known_encoding in self.face_library.items():
            if face_recognition.compare_faces([known_encoding], face_encoding)[0]:
                name = known_name
                break
        return name
