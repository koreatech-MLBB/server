import cv2
import mediapipe as mp
import numpy as np
from PoseVal import pose_val
from multiprocessing import Semaphore, shared_memory
import time
# from threading import Thread, Lock

class PoseEstimation:
    def __init__(self, min_detection_confidence: float, min_tracking_confidence: float, shared_frame: np.ndarray, shared_frame_pop_idx: np.ndarray, shared_frame_push_idx: np.ndarray, shared_frame_rotation_idx: np.ndarray, shared_position: np.ndarray, shared_box: np.ndarray):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_drawing = mp.solutions.drawing_utils
        self.shared_memory = shared_frame
        self.shared_frame_pop_idx = shared_frame_pop_idx
        self.shared_frame_push_idx = shared_frame_push_idx
        self.shared_frame_rotation_idx = shared_frame_rotation_idx
        self.shared_position = shared_position
        self.shared_box = shared_box


    def run(self):
        while True:
            start_time = time.time()
            while not self.shared_frame_rotation_idx[0] and self.shared_frame_pop_idx[0] == self.shared_frame_push_idx[0]:
                pass

            image = np.copy(self.shared_memory[self.shared_frame_pop_idx[0]])
            if self.shared_frame_pop_idx[0] + 1 >= 30:
                self.shared_frame_rotation_idx[0] -= 1
            self.shared_frame_pop_idx[0] = (self.shared_frame_pop_idx[0] + 1) % 30

            try:

                results = self.pose.process(image)
                # print(results)

            except Exception as e:
                print(e)

            # Extract landmarks

            try:
                landmarks = results.pose_landmarks.landmark
                # Get coordinates
                coordinates = {}
                for name, val in pose_val.items():
                    coordinates[name] = [round(landmarks[pose_val[name]].x, 3), round(landmarks[pose_val[name]].y, 3)]

                print(coordinates)

            except Exception as e:
                print(e)

            end_time = time.time()
            print(end_time - start_time)

    def calculate_angle(self, a, b, c):
        a = np.array(a)  # First
        b = np.array(b)  # Mid
        c = np.array(c)  # End

        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        if angle > 180.0:
            angle = 360 - angle

        return angle

    def calculate_distance(self, a, b):
        a = np.array(a)
        b = np.array(b)
        distance = np.linalg.norm(a - b)
        return distance
