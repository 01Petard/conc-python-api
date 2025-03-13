import face_recognition
import pickle
import os


class FaceEncodingManager:
    def __init__(self, face_library_path='face_library.pkl'):
        self.face_library_path = face_library_path
        if os.path.exists(face_library_path):
            with open(face_library_path, 'rb') as f:
                self.face_library = pickle.load(f)
        else:
            self.face_library = {}

    def save_face_encodings(self, path):
        if os.path.isfile(path):
            self._process_single_image(path)
        elif os.path.isdir(path):
            self._process_directory(path)
        else:
            print(f"输入的路径 {path} 既不是文件也不是文件夹，请检查。")
        self._save_face_library()

    def _process_single_image(self, image_path):
        try:
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                encoding = encodings[0]
                name = os.path.splitext(os.path.basename(image_path))[0]
                self.face_library[name] = encoding
                print(f"文件: {image_path}, 已录入人脸: {name}")
            else:
                print(f"未在图像 {image_path} 中检测到人脸。")
        except Exception as e:
            print(f"处理图像 {image_path} 时出错: {e}")

    def _process_directory(self, directory_path):
        for filename in os.listdir(directory_path):
            image_path = os.path.join(directory_path, filename)
            if os.path.isfile(image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                self._process_single_image(image_path)

    def _save_face_library(self):
        with open(self.face_library_path, 'wb') as f:
            pickle.dump(self.face_library, f)

    def delete_face(self, name):
        if name in self.face_library:
            del self.face_library[name]
            self._save_face_library()
            print(f"已删除人脸: {name}")
        else:
            print(f"人脸库中不存在名为 {name} 的人脸。")

    def get_face_names(self):
        return list(self.face_library.keys())


if __name__ == "__main__":
    manager = FaceEncodingManager()

    while True:
        print("\n请选择操作:")
        print("1. 录入人脸（单个图像或文件夹）")
        print("2. 删除人脸")
        print("3. 查看人脸库中的人脸名称")
        print("4. 退出")
        choice = input("请输入选项编号: ")

        if choice == '1':
            path = input("请输入图像或文件夹的路径: ")
            manager.save_face_encodings(path)
        elif choice == '2':
            name = input("请输入要删除的人脸名称: ")
            manager.delete_face(name)
        elif choice == '3':
            names = manager.get_face_names()
            if names:
                print("人脸库中的人脸名称:")
                for name in names:
                    print(name)
            else:
                print("人脸库为空。")
        elif choice == '4':
            break
        else:
            print("无效的选项，请重新输入。")