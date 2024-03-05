from DBConnection import *
from PoseEstimation import *
from DroneController import *
# from threading import Thread, Lock
from multiprocessing import Process, shared_memory, Semaphore
import numpy as np
import time

def img_save(shared_frame, semaphore):
    cam = cv2.VideoCapture(0)
    # print(cam.isOpened())
    while cam.isOpened():
        # print("test3")
        ret, frame = cam.read()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        # semaphore.acquire()
        with semaphore:
            np.copyto(shared_frame, image)
        # semaphore.release()

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
    img_shared_frame = shared_memory.SharedMemory(create=True, name="img_shared", size=img_size[0]*img_size[1]*3*30)
    # img_shared_frame = shared_memory.SharedMemory(name="img_shared", size=img_size[0]*img_size[1]*3*30)
    #
    shared_frame_push_idx = shared_memory.SharedMemory(create=True, name="shared_frame_push_idx")
    shared_frame_pop_idx = shared_memory.SharedMemory(create=True, name="shared_frame_pop_idx")
    shared_frame = np.ndarray(shape=(480, 640, 3, 30), dtype=np.uint8, buffer=img_shared_frame.buf)

    frame_push_idx = shared_frame_push_idx.buf.cast('i')
    frame_push_idx[0] = 0

    frame_pop_idx = shared_frame_pop_idx.buf.cast('i')
    frame_pop_idx[0] = 0

    # cam 객체 생성

    # while cam.isOpened():
    #     ret, frame = cam.read()
    #     print(frame.dtype, frame.shape, frame.nbytes)

    procs = []

    # PoseEstimation 객체 생성
    pe = PoseEstimation(min_detection_confidence=0.5, min_tracking_confidence=0.5, shared_memory=shared_frame, semaphore=img_semaphore)

    # p = Process(target=img_save, args=(shared_frame, img_lock, cam))
    
    # p = Thread(target=img_save, args=(shared_frame, img_lock, cam, ))

    

    p = Process(target=img_save, args=(shared_frame, img_semaphore))
    p.start()
    procs.append(p)
    # print("프로세스 시작됨")
    

    # print("test1")
    # p.start()
    # print("test2")
    # db_process = Process(target=db.run, args=())
    # pe_process = Thread(target=pe.run)
    # try:
    pe_process = Process(target=pe.run)
    procs.append(pe_process)
    #     # db_process.start()
    pe_process.start()
    #     # db_process.join()
    # p.join()
    # pe_process.join()
    # except Exception as e:
    #     print(f"예외 발생 {e}")

    # pe.run()
    for p in procs:
        p.join()

    # pe = pe = PoseEstimation(min_detection_confidence=0.5, min_tracking_confidence=0.5, camNum=0)
