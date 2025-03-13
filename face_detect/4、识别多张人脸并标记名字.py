import face_recognition
import cv2
import time

# 加载已知人物图像和编码
known_image = face_recognition.load_image_file('person_face/person_1_1.png')
known_encoding = face_recognition.face_encodings(known_image)[0]

# 加载待识别图像
unknown_image = cv2.imread('person_face/person_1_2.png')

# 转换为RGB格式
rgb_image = cv2.cvtColor(unknown_image, cv2.COLOR_BGR2RGB)

# 查找图像中的人脸
face_locations = face_recognition.face_locations(rgb_image)
face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

# 将已知人脸编码添加到一个列表中
known_encodings = [known_encoding]

# 对每个识别出的人脸进行比对
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(known_encodings, face_encoding)

    name = "Unknown Person"

    if True in matches:
        first_match_index = matches.index(True)
        # 查询添加已知人物的姓名
        name = "Known Person's Name"

    # 绘制矩形框并标记姓名
    cv2.rectangle(unknown_image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(unknown_image, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

# 显示图像
cv2.imshow('Face Recognition', unknown_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
