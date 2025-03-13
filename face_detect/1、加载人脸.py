import cv2
import face_recognition

# 读取图像
image = cv2.imread('face2.png')

# 转换为RGB格式（OpenCV默认读取BGR格式）
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 查找所有的人脸位置
face_locations = face_recognition.face_locations(rgb_image)

print("Found {} faces in the image.".format(len(face_locations)))
print("face locations in the image: {}".format(face_locations))

# 在图像中标记人脸位置
for (top, right, bottom, left) in face_locations:
    # 绘制矩形框标记人脸
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

# 显示图像
cv2.imshow('Face Recognition', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
