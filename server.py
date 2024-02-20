from DBConnection import *
from PoseEstimation import PoseEstimation

from multiprocessing import Process, shared_memory, Semaphore
import pickle
import random
import numpy as np
import time

def test(shared_memory_name, semaphore):

    db_shared_memory = shared_memory.SharedMemory(name=shared_memory_name)

    semaphore.acquire()
    db_shared_bytes = db_shared_memory.buf.tobytes()
    db_shared_dict = pickle.loads(db_shared_bytes)
    db_shared_dict["func"] = random.choice(["select", "insert", "update", "delete"])
    semaphore.release()
    time.sleep(0.5)

if __name__ == "__main__":

    # db semaphore 생성
    db_semaphore = Semaphore(1)

    # db 공유 메모리 생성
    db_shared_dict = {"func": "", "table": "", "condition": [], "values": [], "order": []}
    db_shared_bytes = pickle.dumps(db_shared_dict)
    db_shared_size = len(db_shared_bytes)
    db_shm = shared_memory.SharedMemory(create=True, size=1024, name="db_shared")
    db_shm.buf[:len(db_shared_bytes)] = db_shared_bytes

    # db 객체 생성
    db = DBConnection(user='rad', password='1234', database='rad', shared_memory_name="db_shared", semaphore=db_semaphore)
    db2 = DBConnection(user='rad', password='1234', database='rad', shared_memory_name="db_shared", semaphore=db_semaphore)

    db_process = Process(target=db.run, args=())
    db_process2 = Process(target=db2.run, args=())
    db_process.start()
    db_process2.start()
    db_process.join()
    db_process2.join()


    # pe = pe = PoseEstimation(min_detection_confidence=0.5, min_tracking_confidence=0.5, camNum=0)