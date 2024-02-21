from DBConnection import *
from PoseEstimation import *
from DroneController import *
from multiprocessing import Process, shared_memory, Semaphore
import pickle
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

    # PoseEstimation 객체 생성
    pe = PoseEstimation(min_tracking_confidence=0.5, min_detection_confidence=0.5, camNum=0)

    # db_process = Process(target=db.run, args=())
    pe_process = Process(target=pe.run, args=())
    # db_process.start()
    pe_process.start()
    # db_process.join()
    pe_process.join()


    # pe = pe = PoseEstimation(min_detection_confidence=0.5, min_tracking_confidence=0.5, camNum=0)