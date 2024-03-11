# import cv2
import pickle

import mediapipe as mp
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from PoseVal import hand_val


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


# 손바닥 펴고 있는지 여부 판단하는 로직
def hand_signal_seri(landmarks, hand_val):
    open_hand_count = 0

    w = landmarks[hand_val['WRIST']]
    thumb = landmarks[hand_val['THUMB_TIP']]

    stand_dis = calculate_distance([w.get('x'), w.get('y')], [thumb.get('x'), thumb.get('y')])

    for key in ['INDEX_FINGER_TIP', 'PINKY_TIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_TIP']:
        f = landmarks[hand_val[key]]
        distance = calculate_distance([w.get('x'), w.get('y')], [f.get('x'), f.get('y')])

        if distance > stand_dis:  # 임계값, 상황에 따라 조정 필요
            open_hand_count += 1

    is_open = open_hand_count >= 3
    return is_open


def serialize_hand_landmarks(multi_hand_landmarks):
    serialized_data = []
    if multi_hand_landmarks:
        for hand_landmarks in multi_hand_landmarks:
            hand_data = []
            for landmark in hand_landmarks.landmark:
                # 각 랜드마크의 x, y, z 좌표와 visibility를 딕셔너리 형태로 저장
                hand_data.append({
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                })
            serialized_data.append(hand_data)
    else:
        return multi_hand_landmarks
    return serialized_data


def serialize_pose_landmarks(pose_landmarks):
    serialized_data = []
    if pose_landmarks:
        pose_data = []
        for landmark in pose_landmarks.landmark:
            # 각 랜드마크의 x, y, z 좌표와 visibility를 딕셔너리 형태로 저장
            pose_data.append({
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z,
                'visibility': landmark.visibility
            })
        serialized_data.append(pose_data)
    else:
        return pose_landmarks
    return serialized_data


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
            pass
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

                hand = mp.solutions.hands.Hands().process(human_box)
                serialized_hand_landmarks = serialize_hand_landmarks(results.multi_hand_landmarks)
                serialized_bytes = pickle.dumps(serialized_hand_landmarks)
                hand = pickle.loads(serialized_bytes)

                if hand:
                    for hand_landmarks in hand:
                        if hand_signal_seri(hand_landmarks, hand_val):
                            self.track_id = track_id
                            return

                xmin, ymin, xmax, ymax = map(int, [box[0], box[1], box[2], box[3]])
                print("box : ", xmin, ymin, xmax, ymax)

    def process(self, frame, track):
        while True:
            box = track.to_ltrb()  # (min x, min y, max x, max y)
            human_box = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            xmin, ymin, xmax, ymax = map(int, [human_box[0], human_box[1], human_box[2], human_box[3]])
            np.copyto(self.shared_box, np.array([xmin + xmax/2, ymin + ymin/2, xmax, ymax]))

            body = mp.solutions.pose.Pose().process(human_box)
            serialized_pose = serialize_pose_landmarks(body.pose_landmarks)
            serialized_bytes = pickle.dumps(serialized_pose)
            body = pickle.loads(serialized_bytes)

            if body:
                self.shared_position = body
                # coordinates = {}
                # for name, val in pose_val.items():
                #     coordinates[name] = [round(body_landmarks[pose_val[name]].x, 3),
                #                          round(body_landmarks[pose_val[name]].y, 3)]

            # check shared memorry is available to get
            while (not self.shared_frame_rotation_idx[0]
                   and self.shared_frame_pop_idx[0] == self.shared_frame_push_idx[0]):
                pass
            # 공유 메모리에서 이미지 꺼내기
            frame = np.copy(self.shared_memory[self.shared_frame_pop_idx[0]])
            if self.shared_frame_pop_idx[0] + 1 >= 30:
                self.shared_frame_rotation_idx[0] -= 1
            self.shared_frame_pop_idx[0] = (self.shared_frame_pop_idx[0] + 1) % 30


