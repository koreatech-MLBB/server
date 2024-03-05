import cv2
import mediapipe as mp
import numpy as np
from PoseVal import pose_val
from multiprocessing import Semaphore, Manager, shared_memory, Lock, Process
import time
# from threading import Thread, Lock

class PoseEstimation:
    def __init__(self, min_detection_confidence: float, min_tracking_confidence: float, shared_memory: np.ndarray, semaphore: Semaphore):


        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence, min_tracking_confidence=min_tracking_confidence)
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mp_drawing = mp.solutions.drawing_utils
        self.semaphore = semaphore
        self.shared_memory = shared_memory
        # self.shared_memory = shared_memory.SharedMemory(name=shared_memory_name)

    def run(self):
        # print("ttset")
        # print(self.cap.isOpened())
        # while self.cap.isOpened():
        while True:
            start_time = time.time()
            # print(time.ctime())
            # print("test")
            # ret, frame = self.cap.read()

            # print(type(frame))

            # Recolor image to RGB
            # image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            # image.flags.writeable = False

            # Make detection
            # self.semaphore.acquire()

            with self.semaphore:
                image = np.copy(self.shared_memory)
            # self.semaphore.release()


            # print(image)

            # with self.pose as pose:
            # with self.mp_pose.Pose(min_detection_confidence=self.min_detection_confidence, min_tracking_confidence=self.min_tracking_confidence) as pose:
            try:

                results = self.pose.process(image)
                # print(results)

            except Exception as e:
                print(e)

            # Recolor back to BGR
            # image.flags.writeable = True
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Extract landmarks

            try:
                landmarks = results.pose_landmarks.landmark
                # Get coordinates
                coordinates = {}
                for name, val in pose_val.items():
                    coordinates[name] = [round(landmarks[pose_val[name]].x, 3), round(landmarks[pose_val[name]].y, 3)]

                print(coordinates)

                # calculate angle
                # angle = self.calculate_angle(coordinates["LEFT_SHOULDER"], coordinates["LEFT_ELBOW"], coordinates["LEFT_WRIST"])
                # for x, y in coordinates.items():
                #     print(x, y)

                # Visualize angle
                # cv2.putText(image, str(angle),
                #             tuple(np.multiply(coordinates["LEFT_ELBOW"], [640, 480]).astype(int)),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                print(e)

                # Render detections
            # self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
            #                           self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            #                           self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            #
            # cv2.imshow('Mediapipe Feed', image)

            # if cv2.waitKey(10) & 0xFF == ord('q'):
            #     self.cap.release()
            #     cv2.destroyAllWindows()
            #     break
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
