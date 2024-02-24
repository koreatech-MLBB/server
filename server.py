from DBConnection import *
from PoseEstimation import *
from DroneController import *
from threading import Thread, Lock
import pickle
import numpy as np
import time

def img_save(shared_frame, lock, cam):
    while cam.isOpened():
        ret, frame = cam.read()
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image.flags.writeable = False
        # 921600
        with lock:
            np.copyto(shared_frame, image)

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
    img_lock = Lock()

    # image 공유 메모리 생성
    # img_shared_frame = shared_memory.SharedMemory(create=True, name="img_shared", size=921600)
    img_shared_frame = shared_memory.SharedMemory(name="img_shared", size=921600)
    shared_frame = np.ndarray(shape=(480, 640, 3), dtype=np.uint8, buffer=img_shared_frame.buf)
    # cam 객체 생성
    cam = cv2.VideoCapture(0)

    time.sleep(2)

    # while cam.isOpened():
    #     ret, frame = cam.read()
    #     print(frame.dtype, frame.shape, frame.nbytes)

    # PoseEstimation 객체 생성
    pe = PoseEstimation(min_detection_confidence=0.5, min_tracking_confidence=0.5, cam=cam, shared_memory=shared_frame, lock=img_lock)
    # pe.run()
    p = Thread(target=img_save, args=(shared_frame, img_lock, cam, ))
    p.start()

    # db_process = Process(target=db.run, args=())
    pe_process = Thread(target=pe.run)
    # db_process.start()
    pe_process.start()
    # db_process.join()
    p.join()
    pe_process.join()

    # pe = pe = PoseEstimation(min_detection_confidence=0.5, min_tracking_confidence=0.5, camNum=0)
