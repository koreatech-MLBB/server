import pickle
from multiprocessing import shared_memory as sm

import cv2
import mediapipe as mp
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

from PoseVal import hand_val

# import time

CONFIDENCE_THRESHOLD = 0.6
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

REGULAR = 1
BOLD = 2

model = YOLO('./yolov8_pretrained/yolov8n.pt')

def PoseEstimation(shared_memories: dict):
        # shared_frame_name: str, shared_frame_pop_idx_name: str,
        #            shared_frame_push_idx_name: str, shared_frame_rotation_idx_name: str,
        #            shared_position_name: str, shared_box_name: str):

    tracker = DeepSort()
    track_id = '0'

    # print("PoseEstimation")

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

    # def make_shared_memory(memories: dict):
    #     result = []
    #     for name, val in memories.items():
    #         mem = sm.SharedMemory(name=name)
    #         result.append(np.ndarray(shape=val[0], dtype=val[1], buffer=mem.buf))
    #     return result

    def process(previous_frame, track):
        # let = True
        frame = previous_frame

        box = track.to_ltrb()

        # why sleep?
        # time.sleep(3)
        print("proces!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        while True:

            shared_mem_list = []
            for name, val in shared_memories.items():
                shm = sm.SharedMemory(name=name)
                shared_mem_list.append(shm)

            shared_mem = {}
            for idx, val in enumerate(shared_memories.items()):
                buf = np.ndarray(shape=val[1][0], dtype=val[1][1], buffer=shared_mem_list[idx].buf)
                shared_mem[val[0]] = buf

            landmark_buf = []

            # shared_frame, shared_frame_pop_idx, shared_frame_push_idx, shared_frame_rotation_idx, shared_position, shared_box = make_shared_memory(memories=shared_memories)

            # box = track.to_ltrb()  # (min x, min y, max x, max y)
            detection = model.predict(source=[frame])[0]
            xmin, ymin, xmax, ymax = map(int, [box[0], box[1], box[2], box[3]])
            human_box = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            # xmin, ymin, xmax, ymax = map(int, [human_box[0], human_box[1], human_box[2], human_box[3]])
            np.copyto(shared_mem["shared_box"], np.array([xmin + xmax/2, ymin + ymin/2, xmax, ymax]))

            if human_box.size != 0:
                body = mp.solutions.pose.Pose().process(human_box)
                # print(type(body))
                for landmark in body.pose_landmarks.landmark:
                    landmark_buf.append(landmark.x)
                    landmark_buf.append(landmark.y)
                    landmark_buf.append(landmark.z)
                    landmark_buf.append(landmark.visibility)
                landmark_buf = np.reshape(landmark_buf, (33, -1))

                np.copyto(shared_mem["shared_position"], landmark_buf)

            # print(shared_frame_pop_idx, shared_frame_push_idx, shared_frame_rotation_idx)

            results = []

            for data in detection.boxes.data.tolist():
                confidence = float(data[4])
                label = int(data[5])
                if confidence < CONFIDENCE_THRESHOLD or label != 0:
                    continue

                xmin, ymin, xmax, ymax = map(int, [x for x in data[:4]])
                results.append([[xmin, ymin, xmax - xmin, ymax - ymin], confidence, label])

            tracks = tracker.update_tracks(results, frame=frame)
            for t in tracks:
                if t.track_id == track_id:
                    track = t
                    box = track.to_ltrb()
                    # shared_box = np.array([(box[0] + box[2]) // 2, (box[1] + box[3]) // 2, box[2], box[3]])
                    np.copyto(shared_mem["shared_box"], np.array([(box[0] + box[2]) // 2, (box[1] + box[3]) // 2, box[2], box[3]]))
                    continue

            if shared_mem["shared_frame_rotation_idx"][0] == 0 and (
                    shared_mem["shared_frame_pop_idx"][0] == shared_mem["shared_frame_push_idx"][0]):
                # print("Pose in if")
                continue
            # 공유 메모리에서 이미지 꺼내기
            frame = np.copy(shared_mem["img_shared"][shared_mem["shared_frame_pop_idx"][0]])
            # cv2.imshow("pose", frame)
            if shared_mem["shared_frame_pop_idx"][0] + 1 >= 30:
                shared_mem["shared_frame_rotation_idx"][0] = 0
            shared_mem["shared_frame_pop_idx"][0] = (shared_mem["shared_frame_pop_idx"][0] + 1) % 30

    while True:
        # shared_frame, shared_frame_pop_idx, shared_frame_push_idx, shared_frame_rotation_idx, shared_position, shared_box = make_shared_memory(memories=shared_memories)

        shared_mem_list = []
        for name, val in shared_memories.items():
            shm = sm.SharedMemory(name=name)
            shared_mem_list.append(shm)

        shared_mem = {}
        for idx, val in enumerate(shared_memories.items()):
            buf = np.ndarray(shape=val[1][0], dtype=val[1][1], buffer=shared_mem_list[idx].buf)
            shared_mem[val[0]] = buf

        if (shared_mem["shared_frame_pop_idx"][0] == shared_mem["shared_frame_push_idx"][0]) and shared_mem["shared_frame_rotation_idx"][0] == 0:
            # print("Pose in if")
            continue

        frame = np.copy(shared_mem["img_shared"][shared_mem["shared_frame_pop_idx"][0]])

        cv2.imshow("pose", frame)
        cv2.waitKey(1)

        if shared_mem["shared_frame_pop_idx"][0] + 1 >= 30:
            shared_mem["shared_frame_rotation_idx"][0] = 0
        shared_mem["shared_frame_pop_idx"][0] = (shared_mem["shared_frame_pop_idx"][0] + 1) % 30

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

        tracks = tracker.update_tracks(results, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id: str = track.track_id
            box = track.to_ltrb()  # (min x, min y, max x, max y)
            human_box = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            xmin, ymin, xmax, ymax = map(int, [human_box[0], human_box[1], human_box[2], human_box[3]])

            # NOTE: shared_box
            shared_box = [xmin + xmax / 2, ymin + ymax / 2, xmax, ymax]

            if human_box.size == 0:
                continue

            hand = mp.solutions.hands.Hands().process(human_box)
            serialized_hand_landmarks = serialize_hand_landmarks(hand.multi_hand_landmarks)
            serialized_bytes = pickle.dumps(serialized_hand_landmarks)
            hand = pickle.loads(serialized_bytes)

            if hand:
                for hand_landmarks in hand:
                    if hand_signal_seri(hand_landmarks, hand_val):
                        track_id = track_id
                        process(frame, track)
                        return

