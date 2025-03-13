import cv2
from HPE.FaceManager.face_manager import FaceManager
from HPE.HPE_FSANet.head_post_estimate import HeadPoseEstimator


def handle_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_num = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1
        yield frame_num, frame
    cap.release()


def main(video_path):
    estimator = HeadPoseEstimator()
    face_util = FaceManager()

    for frame_num, frame in handle_video(video_path):
        face_locations = face_util.detect_faces(frame)
        for face_location in face_locations:
            pitch, yaw = estimator.estimate_head_pose(frame, face_location)
            face_name = face_util.recognize_face(frame, face_location, pitch, yaw, frame_num)
            print(("frame:{},name:{},box:{},pitch:{},yaw:{}\n", frame_num, face_name, face_location, pitch, yaw))

if __name__ == "__main__":
    video_path = input("请输入视频文件的路径: ")
    main(video_path)
