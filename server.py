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


    try:
        shared_frame_push_idx = shared_memory.SharedMemory(name="shared_frame_push_idx")
    except FileNotFoundError:
        shared_frame_push_idx = shared_memory.SharedMemory(create=True, name="shared_frame_push_idx", size=4)

    try:
        shared_frame_pop_idx = shared_memory.SharedMemory(name="shared_frame_pop_idx")
    except FileNotFoundError:
        shared_frame_pop_idx = shared_memory.SharedMemory(create=True, name="shared_frame_pop_idx", size=4)

    try:
        shared_frame_idx_rotation = shared_memory.SharedMemory(name="shared_frame_idx_rotation")
    except FileNotFoundError:
        shared_frame_idx_rotation = shared_memory.SharedMemory(create=True, name="shared_frame_idx_rotation", size=4)

    shared_frame = np.ndarray(shape=(480, 640, 3, 30), dtype=np.uint8, buffer=img_shared_frame.buf)

    frame_push_idx = shared_frame_push_idx.buf.cast('i')
    frame_push_idx[0] = 0

    frame_pop_idx = shared_frame_pop_idx.buf.cast('i')
    frame_pop_idx[0] = 0

    frame_idx_rotation = shared_frame_idx_rotation.buf.cast('i')
    frame_idx_rotation[0] = 0

    procs = []

    # PoseEstimation 객체 생성
    pe = PoseEstimation(min_detection_confidence=0.5, min_tracking_confidence=0.5, shared_frame=shared_frame, semaphore=img_semaphore, shared_frame_push_idx_name='shared_frame_push_idx', shared_frame_pop_idx_name='shared_frame_pop_idx', shared_frame_idx_rotation_name='shared_frame_idx_rotation')

    ec = ESPConnection(shared_frame=shared_frame, shared_frame_push_idx_name='shared_frame_push_idx', shared_frame_pop_idx_name='shared_frame_pop_idx', shared_frame_idx_rotation_name='shared_frame_idx_rotation', img_size=(480, 640), serverPort=4703)


    pe_process = Process(target=pe.run)
    procs.append(pe_process)
    pe_process.start()

    ec_process = Process(target=ec.run)
    procs.append(ec_process)
    ec_process.start()

    for p in procs:
        p.join()

    img_shared_frame.close()
    shared_frame_pop_idx.close()
    shared_frame_push_idx.close()
    shared_frame_idx_rotation.close()