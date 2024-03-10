import pickle

# import cv2
import mediapipe as mp
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from PoseVal import pose_val, hand_val


# from threading import Thread, Lock

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_distance(a, b):
    a = np.array(a)
    b = np.array(b)
    distance = np.linalg.norm(a - b)
    return distance


def is_inside_region(point, top_left, width, height):
    x, y = point
    top_x, top_y = top_left
    bottom_x = top_x + width
    bottom_y = top_y + height

    if x < top_x or x > bottom_x or y < top_y or y > bottom_y:
        return False
    return True


# 손바닥 펴고 있는지 여부 판단하는 로직
def hand_signal(landmarks, hand_val):
    open_hand_count = 0

    w = landmarks.landmark[hand_val['WRIST']]
    thumb = landmarks.landmark[hand_val['THUMB_TIP']]

    stand_dis = calculate_distance([w.x, w.y], [thumb.x, thumb.y])

    for key in ['INDEX_FINGER_TIP', 'PINKY_TIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_TIP']:
        f = landmarks.landmark[hand_val[key]]
        distance = calculate_distance([w.x, w.y], [f.x, f.y])
        print("distance : ", distance)
        # 임계값 설정 (이 값은 조정이 필요할 수 있음)
        if distance > stand_dis:  # 임계값, 상황에 따라 조정 필요
            open_hand_count += 1

    is_open = open_hand_count >= 3
    return is_open


def hand_signal_seri(landmarks, hand_val):
    open_hand_count = 0

    w = landmarks[hand_val['WRIST']]
    thumb = landmarks[hand_val['THUMB_TIP']]

    stand_dis = calculate_distance([w.x, w.y], [thumb.x, thumb.y])

    for key in ['INDEX_FINGER_TIP', 'PINKY_TIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_TIP']:
        f = landmarks[hand_val[key]]
        distance = calculate_distance([w.x, w.y], [f.x, f.y])

        if distance > stand_dis:  # 임계값, 상황에 따라 조정 필요
            open_hand_count += 1

    is_open = open_hand_count >= 3
    return is_open


CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

REGULAR = 1
BOLD = 2

model = YOLO('./yolov8_pretrained/yolov8n.pt')


class PoseEstimation:
    def __init__(self, shared_frame: np.ndarray, shared_frame_pop_idx: np.ndarray, shared_frame_push_idx: np.ndarray, shared_frame_rotation_idx: np.ndarray, shared_position: np.ndarray, shared_box: np.ndarray):
        self.mp_pose = mp.solutions.pose
        self.mp_hand = mp.solutions.hands
        # self.pose = self.mp_pose.Pose()
        # self.hand = self.mp_hand.Hands()

        self.mp_drawing = mp.solutions.drawing_utils

        self.tracker = DeepSort()
        self.track_id = '0'

        self.shared_memory = shared_frame
        self.shared_frame_pop_idx = shared_frame_pop_idx
        self.shared_frame_push_idx = shared_frame_push_idx
        self.shared_frame_rotation_idx = shared_frame_rotation_idx
        self.shared_position = shared_position
        self.shared_box = shared_box

    def run(self):
        while True:
            # check shared memorry is available to get
            while (not self.shared_frame_rotation_idx[0]
                   and self.shared_frame_pop_idx[0] == self.shared_frame_push_idx[0]):
                pass
            # 공유 메모리에서 이미지 꺼내기
            frame = np.copy(self.shared_memory[self.shared_frame_pop_idx[0]])

            if self.shared_frame_pop_idx[0] + 1 >= 30:
                self.shared_frame_rotation_idx[0] -= 1
            self.shared_frame_pop_idx[0] = (self.shared_frame_pop_idx[0] + 1) % 30

            detection = model.predict(source=[frame])[0]
            results = []

            # 감지된 물체의 라벨(종류)과 확신도
            for data in detection.boxes.data.tolist():
                # data : [xmin, ymin, xmax, ymax, confidence_score, type]
                confidence = float(data[4])
                label = int(data[5])
                if confidence < CONFIDENCE_THRESHOLD or label != 0:
                    continue
                # 감지된 물체가 사람이고 confidence가 0.6보다 크면
                xmin, ymin, xmax, ymax = map(int, [data[0], data[1], data[2], data[3]])
                results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, label])

            tracks = self.tracker.update_tracks(results, frame=frame)

            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id: str = track.track_id
                box = track.to_ltrb()  # (min x, min y, max x, max y)
                human_box = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

                if human_box.size == 0:
                    continue

                # hand = self.mp_hand.Hands().process(cv2.cvtColor(human_box, cv2.COLOR_RGB2BGR))
                hand = self.mp_hand.Hands().process(human_box)

                if hand:
                    for hand_landmarks in hand:  # 반복문을 활용해 인식된 손의 주요 부분을 그림으로 그려 표현
                        # self.mp_drawing.draw_landmarks(
                        #     human_box,
                        #     hand_landmarks,
                        #     self.mp_hand.HAND_CONNECTIONS,
                        # )
                        # check = hand_signal(hand_landmarks, hand_val)
                        check = hand_signal_seri(hand_landmarks, hand_val)

                        print(f'check : {check}')
                        if check:
                            self.track_id = track_id
                            self.process(frame, track)
                            return

                xmin, ymin, xmax, ymax = map(int, [box[0], box[1], box[2], box[3]])
                print("box : ", xmin, ymin, xmax, ymax)

            #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), GREEN, BOLD)
            #     cv2.rectangle(frame, (xmin, ymin - 20), (xmin + 20, ymin), YELLOW, -1)
            #     cv2.putText(frame, track_id, (xmin + 5, ymin - 8),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 2)
            #
            # cv2.imshow('hand detect', frame)

    def process(self, frame, track):
        while True:
            box = track.to_ltrb()  # (min x, min y, max x, max y)
            human_box = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            xmin, ymin, xmax, ymax = map(int, [human_box[0], human_box[1], human_box[2], human_box[3]])
            # self.shared_box = np.array([xmin + xmax/2, ymin + ymin/2, xmax, ymax])
            np.copyto(self.shared_box, np.array([xmin + xmax/2, ymin + ymin/2, xmax, ymax]))

            # body = self.mp_pose.Pose().process(cv2.cvtColor(human_box, cv2.COLOR_RGB2BGR))
            body = self.mp_pose.Pose().process(human_box)

            if body.pose_landmarks:
                body_landmarks = body.pose_landmarks.landmark
                array_buf = np.empty((34, 4))
                for i, landmark in enumerate(body_landmarks):
                    for j, tup in enumerate(str(landmark).strip().split('\n')):
                        value = float(tup.split(': ')[1])
                        array_buf[i, j] = value
                self.shared_position = array_buf.copy()

                self.mp_drawing.draw_landmarks(
                    human_box,
                    body.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )

                coordinates = {}
                for name, val in pose_val.items():
                    coordinates[name] = [round(body_landmarks[pose_val[name]].x, 3),
                                         round(body_landmarks[pose_val[name]].y, 3)]
            # cv2.imshow('Pose Estimation', frame)

            # check shared memorry is available to get
            while (not self.shared_frame_rotation_idx[0]
                   and self.shared_frame_pop_idx[0] == self.shared_frame_push_idx[0]):
                pass
            # 공유 메모리에서 이미지 꺼내기
            frame = np.copy(self.shared_memory[self.shared_frame_pop_idx[0]])
            if self.shared_frame_pop_idx[0] + 1 >= 30:
                self.shared_frame_rotation_idx[0] -= 1
            self.shared_frame_pop_idx[0] = (self.shared_frame_pop_idx[0] + 1) % 30


