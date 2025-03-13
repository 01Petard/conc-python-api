import cv2
import numpy as np
import onnxruntime

class HeadPoseEstimator:
    def __init__(self):
        self.sess1 = onnxruntime.InferenceSession("D:/Projects_Another/demo_ai/HPE/HPE_FSANet/fsanet-1x1-iter-688590.onnx")
        self.sess2 = onnxruntime.InferenceSession("D:/Projects_Another/demo_ai/HPE/HPE_FSANet/fsanet-var-iter-688590.onnx")

    def estimate_head_pose(self, frame, face_location):
        top, right, bottom, left = face_location
        face_roi = frame[top:bottom, left:right]
        face_roi = cv2.resize(face_roi, (64, 64))
        face_roi = face_roi.transpose((2, 0, 1))
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = (face_roi - 127.5) / 128
        face_roi = face_roi.astype(np.float32)

        res1 = self.sess1.run(["output"], {"input": face_roi})[0]
        res2 = self.sess2.run(["output"], {"input": face_roi})[0]
        yaw, pitch, roll = np.mean(np.vstack((res1, res2)), axis=0)
        yaw, pitch, roll = -yaw, pitch, roll
        return pitch, yaw