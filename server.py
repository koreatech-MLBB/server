from DBConnection import *
from PoseEstimation import *
from DroneController import *
from ESPConnection import *
# from threading import Thread, Lock
from multiprocessing import Process, shared_memory, Semaphore
import numpy as np
import time


if __name__ == "__main__":

    # db semaphore 생성
    # db_semaphore = Semaphore(1)

    # db 공유 메모리 생성
    # db_shared_dict = {"func": "", "table": "", "condition": [], "values": [], "order": [], "result": []}
    # db_shared_bytes = pickle.dumps(db_shared_dict)
    # db_shared_size = len(db_shared_bytes)
    # db_shm = shared_memory.SharedMemory(size=1024, name="db_shared")
    # db_shm = shared_memory.SharedMemory(create=True, size=1024, name="db_shared")
    # db_shm.buf[:len(db_shared_bytes)] = db_shared_bytes

    # db 객체 생성
    # db = DBConnection(user='rad', password='1234', database='rad', shared_memory_name="db_shared", semaphore=db_semaphore)

    # iamge semaphore 생성
    img_semaphore = Semaphore(1)
    

    # image 공유 메모리 생성
    img_size = (480, 640)

    try:
        img_shared_frame = shared_memory.SharedMemory(name="img_shared")
    except FileNotFoundError:
        img_shared_frame = shared_memory.SharedMemory(create=True, name="img_shared", size=img_size[0]*img_size[1]*3*30)

    # 이미지 push index 공유 메모리 생성
    try:
        shared_frame_push_idx = shared_memory.SharedMemory(name="shared_frame_push_idx")
    except FileNotFoundError:
        shared_frame_push_idx = shared_memory.SharedMemory(create=True, name="shared_frame_push_idx", size=1)

    # 이미지 pop index 공유 메모리 생성
    try:
        shared_frame_pop_idx = shared_memory.SharedMemory(name="shared_frame_pop_idx")
    except FileNotFoundError:
        shared_frame_pop_idx = shared_memory.SharedMemory(create=True, name="shared_frame_pop_idx", size=1)

    # 이미지 push/pop index 상태 공유 메모리 생성
    try:
        shared_frame_rotation_idx = shared_memory.SharedMemory(name="shared_frame_idx_rotation")
    except FileNotFoundError:
        shared_frame_rotation_idx = shared_memory.SharedMemory(create=True, name="shared_frame_idx_rotation", size=1)

    # 공유메모리 numpy.ndarray로 변환
    shared_frame = np.ndarray(shape=(480, 640, 3, 30), dtype=np.uint8, buffer=img_shared_frame.buf)
    frame_push_idx = np.ndarray(shape=(1, ), dtype=np.uint8, buffer=shared_frame_push_idx.buf)
    frame_pop_idx = np.ndarray(shape=(1, ), dtype=np.uint8, buffer=shared_frame_pop_idx.buf)
    frame_rotation_idx = np.ndarray(shape=(1, ), dtype=np.uint8, buffer=shared_frame_rotation_idx.buf)

    # position 공유 메모리 생성
    try:
        shared_position_memory = shared_memory.SharedMemory(name="shared_position")
    except FileNotFoundError:
        shared_position_memory = shared_memory.SharedMemory(create=True, name="shared_position", size=528)

    # YOLO-box 공유 메모리 생성
    try:
        shared_box_memory = shared_memory.SharedMemory(name="shared_box")
    except FileNotFoundError:
        shared_box_memory = shared_memory.SharedMemory(create=True, name="shared_box", size=32)

    # 공유 메모리 numpy.ndarray로 변환
    shared_position = np.ndarray(shape=(33, 2), dtype=np.float64, buffer=shared_position_memory)
    shared_box = np.ndarray(shape=(4), dtype=np.float64, buffer=shared_box_memory)

    # 프로세스 리스트 생성
    procs = []

    # PoseEstimation 객체 생성
    pe = PoseEstimation(min_detection_confidence=0.5, min_tracking_confidence=0.5, shared_frame=shared_frame, shared_frame_pop_idx=frame_pop_idx, shared_frame_push_idx=frame_push_idx, shared_frame_rotation_idx=frame_rotation_idx, shared_position=shared_position, shared_box=shared_box)

    # ESPConnection 객체 생성
    ec = ESPConnection(shared_frame=shared_frame, img_size=(480, 640), serverPort=4703, shared_frame_pop_idx=frame_pop_idx, shared_frame_push_idx=frame_push_idx, shared_frame_rotation_idx=frame_rotation_idx)

    # 드론 ssid, password
    ssid = ""
    password = ""

    # DroneController 객체 생성
    dc = DroneController(ssid=ssid, password=password, shared_position=shared_position, shared_box=shared_box, standard_box=300, critical_value=10, img_size=img_size, sensor_size=(3.6, 3.6), subject_height=1.8, k_yaw=0.002)

    # pe 프로세스 생성
    pe_process = Process(target=pe.run, name="pose_estimation")
    procs.append(pe_process)
    pe_process.start()

    # ec 프로세스 생성
    ec_process = Process(target=ec.run, name="esp_connection")
    procs.append(ec_process)
    ec_process.start()

    for p in procs:
        p.join()

    img_shared_frame.close()
    shared_frame_pop_idx.close()
    shared_frame_push_idx.close()
    shared_frame_rotation_idx.close()