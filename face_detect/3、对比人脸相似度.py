import face_recognition

# 加载两张图片
image1 = face_recognition.load_image_file('person_face/person_1_1.png')
image2 = face_recognition.load_image_file('person_face/person_1_1.png')

# 获取两张图片的人脸编码
encoding1 = face_recognition.face_encodings(image1)[0]
encoding2 = face_recognition.face_encodings(image2)[0]

print("Encoding 1:", encoding1)
print("Encoding 2:", encoding2)

# 比较两张人脸编码
results = face_recognition.compare_faces([encoding1], encoding2, tolerance=0.6)

print("Results:", results)

if results[0]:
    print("The two faces are the same person!")
else:
    print("The two faces are different people.")
