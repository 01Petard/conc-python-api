import face_recognition
import cv2
import pickle
import os

class FaceVerifier:
    def __init__(self, face_library_path='face_library.pkl'):
        self.face_library_path = face_library_path
        if os.path.exists(face_library_path):
            with open(face_library_path, 'rb') as f:
                self.face_library = pickle.load(f)
        else:
            self.face_library = {}

    def verify_faces(self, path):
        if os.path.isfile(path):
            self._verify_single_image(path)
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

            # 生产模式
            for face_encoding in face_encodings:
                name = self._get_face_name(face_encoding)
                print(f"文件：{image_path}, 识别的人脸：{name}")

            # 调试模式
            # for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            #     name = self._get_face_name(face_encoding)
            #     cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
            #     cv2.putText(unknown_image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
            #
            # cv2.imshow('Face Recognition', unknown_image)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

        except Exception as e:
            print(f"验证图像 {image_path} 时出错: {e}")

    def _verify_directory(self, directory_path):
        for filename in os.listdir(directory_path):
            image_path = os.path.join(directory_path, filename)
            if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self._verify_single_image(image_path)

    def _get_face_name(self, face_encoding):
        known_encodings = list(self.face_library.values())
        names = list(self.face_library.keys())
        matches = face_recognition.compare_faces(known_encodings, face_encoding,tolerance=0.40)
        name = "Unknown Person"
        if True in matches:
            first_match_index = matches.index(True)
            name = names[first_match_index]
        return name


if __name__ == "__main__":
    verifier = FaceVerifier()

    while True:
        print("\n请选择操作:")
        print("1. 验证单张人脸图像")
        print("2. 验证文件夹中的所有人脸图像")
        print("3. 退出")
        choice = input("请输入选项编号: ")

        if choice == '1':
            path = input("请输入图像的路径: ")
            verifier.verify_faces(path)
        elif choice == '2':
            path = input("请输入文件夹的路径: ")
            verifier.verify_faces(path)
        elif choice == '3':
            break
        else:
            print("无效的选项，请重新输入。")