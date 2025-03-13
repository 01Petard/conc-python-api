import cv2
import face_recognition
import numpy as np

class HeadPoseEstimator:
    def __init__(self):
        # 3D 模型点
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # 鼻尖
            (0.0, -330.0, -65.0),        # 下巴
            (-225.0, 170.0, -135.0),     # 左眼左角
            (225.0, 170.0, -135.0),      # 右眼右角
            (-150.0, -150.0, -125.0),    # 左嘴角
            (150.0, -150.0, -125.0)      # 右嘴角
        ])

    def estimate_head_pose_in_image(self, image_path):
        """
        估计单张图像中的人脸欧拉角
        :param image_path: 图像文件路径
        :return: 每张人脸的俯仰角和偏航角列表
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"无法读取图像: {image_path}")
                return []
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_image)
            poses = []
            for face_location in face_locations:
                face_landmarks = face_recognition.face_landmarks(rgb_image, [face_location])[0]
                # 检查关键点是否都存在
                required_landmarks = ['nose_tip', 'chin', 'left_eye', 'right_eye', 'mouth_left', 'mouth_right']
                if not all(key in face_landmarks for key in required_landmarks):
                    print("关键点信息不完整，跳过此人脸")
                    continue
                image_points = np.array([
                    (face_landmarks['nose_tip'][0][0], face_landmarks['nose_tip'][0][1]),  # 鼻尖
                    (face_landmarks['chin'][8][0], face_landmarks['chin'][8][1]),  # 下巴
                    (face_landmarks['left_eye'][0][0], face_landmarks['left_eye'][0][1]),  # 左眼左角
                    (face_landmarks['right_eye'][3][0], face_landmarks['right_eye'][3][1]),  # 右眼右角
                    (face_landmarks['mouth_left'][0][0], face_landmarks['mouth_left'][0][1]),  # 左嘴角
                    (face_landmarks['mouth_right'][0][0], face_landmarks['mouth_right'][0][1])  # 右嘴角
                ], dtype="double")

                size = image.shape
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

                if success:
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
                    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
                    pitch = euler_angles[0, 0]
                    yaw = euler_angles[1, 0]
                    poses.append((pitch, yaw))
                else:
                    print("求解PnP失败，跳过此人脸")
            return poses
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")
            return []


if __name__ == "__main__":
    estimator = HeadPoseEstimator()
    image_path = "test.jpg"  # 替换为实际的图像路径
    poses = estimator.estimate_head_pose_in_image(image_path)
    print("检测到的人脸欧拉角:", poses)