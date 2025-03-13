import face_recognition
import cv2
import dlib
import numpy as np
import pickle
import os
import time
from datetime import datetime
from tqdm import tqdm
from face_verifier import FaceVerifier as FaceVerifierBase  # 导入face_verifier.py中的FaceVerifier类


class FaceVerifier(FaceVerifierBase):
    def __init__(self, face_library_path='face_library.pkl'):
        super().__init__(face_library_path)
        # 加载人脸检测器和特征点检测器
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        # 3D 模型点
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 鼻尖
            (0.0, -330.0, -65.0),        # 下巴
            (-225.0, 170.0, -135.0),     # 左眼左角
            (225.0, 170.0, -135.0),      # 右眼右角
            (-150.0, -150.0, -125.0),    # 左嘴角
            (150.0, -150.0, -125.0)      # 右嘴角
        ])
        self.debug = True
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"face_data_{self.timestamp}"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def verify_faces(self, path):
        if os.path.isfile(path):
            if path.lower().endswith(('.mp4', '.avi')):
                self._verify_single_video(path)
            elif path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self._verify_single_image(path)
            else:
                print(f"不支持的文件类型: {path}")
        elif os.path.isdir(path):
            self._verify_directory(path)
        else:
            print(f"输入的路径 {path} 既不是文件也不是文件夹，请检查。")

    def _verify_single_image(self, image_path):
        try:
            unknown_image = cv2.imread(image_path)
            rgb_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

            for frame_num, ((top, right, bottom, left), face_encoding) in enumerate(zip(face_locations, face_encodings), start=1):
                name = super()._get_face_name(face_encoding)  # 调用父类的_get_face_name方法
                pitch, yaw = self._estimate_head_pose(unknown_image, (top, right, bottom, left))
                self._save_face_data(frame_num, name, (left, top, right, bottom), pitch, yaw)

                if self.debug:
                    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.putText(unknown_image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            if self.debug:
                cv2.imshow('Face Recognition', unknown_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        except Exception as e:
            print(f"验证图像 {image_path} 时出错: {e}")

    def _verify_single_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 1

        with tqdm(total=frame_count, desc="Processing video", unit="frame") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    name = super()._get_face_name(face_encoding)  # 调用父类的_get_face_name方法
                    pitch, yaw = self._estimate_head_pose(frame, (top, right, bottom, left))
                    self._save_face_data(frame_num, name, (left, top, right, bottom), pitch, yaw)

                    if self.debug:
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

                if self.debug:
                    cv2.imshow('Face Recognition', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_num += 1
                pbar.update(1)

        cap.release()
        if self.debug:
            cv2.destroyAllWindows()

    def _verify_directory(self, directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            if os.path.isfile(file_path):
                if file_path.lower().endswith(('.mp4', '.avi')):
                    self._verify_single_video(file_path)
                elif file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self._verify_single_image(file_path)
                else:
                    print(f"不支持的文件类型: {file_path}")

    def _estimate_head_pose(self, frame, face_location):
        top, right, bottom, left = face_location
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_rect = dlib.rectangle(left, top, right, bottom)
        shape = self.predictor(gray, face_rect)

        image_points = np.array([
            (shape.part(30).x, shape.part(30).y),  # 鼻尖
            (shape.part(8).x, shape.part(8).y),    # 下巴
            (shape.part(36).x, shape.part(36).y),  # 左眼左角
            (shape.part(45).x, shape.part(45).y),  # 右眼右角
            (shape.part(48).x, shape.part(48).y),  # 左嘴角
            (shape.part(54).x, shape.part(54).y)   # 右嘴角
        ], dtype="double")

        size = frame.shape
        focal_length = size[1]
        center = (size[1] / 2, size[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))
        (success, rotation_vector, translation_vector) = cv2.solvePnP(self.model_points, image_points, camera_matrix,
                                                                      dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

        pitch = euler_angles[0, 0]
        yaw = euler_angles[1, 0]
        return pitch, yaw

    def _save_face_data(self, frame_num, name, face_box, pitch, yaw):
        file_path = os.path.join(self.save_dir, f"{name}.txt")
        with open(file_path, "a") as f:
            f.write(f"帧序号: {frame_num}, 人脸名称: {name}, 人脸坐标框: {face_box}, 俯仰角: {pitch}, 偏航角: {yaw}\n")

    def batch_read_face_data(self, data_dir):
        if not os.path.exists(data_dir):
            print(f"目录 {data_dir} 不存在。")
            return
        for filename in os.listdir(data_dir):
            if filename.endswith(".txt"):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        print(line.strip())


if __name__ == "__main__":
    verifier = FaceVerifier()

    while True:
        print("\n请选择操作:")
        print("1. 验证单张人脸图像或视频")
        print("2. 验证文件夹中的所有人脸图像或视频")
        print("3. 批量读取人脸欧拉角数据")
        print("4. 退出")
        choice = input("请输入选项编号: ")

        if choice == '1':
            path = input("请输入图像或视频的路径: ")
            verifier.verify_faces(path)
        elif choice == '2':
            path = input("请输入文件夹的路径: ")
            verifier.verify_faces(path)
        elif choice == '3':
            data_dir = input("请输入数据目录的路径: ")
            verifier.batch_read_face_data(data_dir)
        elif choice == '4':
            break
        else:
            print("无效的选项，请重新输入。")