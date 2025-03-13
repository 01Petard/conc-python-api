import face_recognition
import cv2

# 读取图像
image = cv2.imread('face1.png')

# 转换为RGB格式
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 查找人脸编码
face_encodings = face_recognition.face_encodings(rgb_image)

if len(face_encodings) > 0:
    # 人脸的编码
    for face_encoding in face_encodings:
        print(f"Face encoding: {face_encoding}")
else:
    print("No face found in the image")
