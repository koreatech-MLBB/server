from DBConnection import *
from PoseEstimation import *
from DroneController import *
from ESPConnection import *
# from threading import Thread, Lock
from multiprocessing import Process, shared_memory as sm, Lock
# from multiprocessing import Semaphore
from multiprocessing import Pool
import numpy as np
import time


def img_save(shared_frame_name, shared_frame_rotation_idx_name,
             shared_frame_pop_idx_name, shared_frame_push_idx_name, img_shape, lock):
    cam = cv2.VideoCapture(0)
    shared_frame_buf = sm.SharedMemory(name=shared_frame_name)
    shared_frame_pop_idx_buf = sm.SharedMemory(name=shared_frame_pop_idx_name)
    shared_frame_push_idx_buf = sm.SharedMemory(name=shared_frame_push_idx_name)
    shared_frame_rotation_idx_buf = sm.SharedMemory(name=shared_frame_rotation_idx_name)

    shared_frame = np.ndarray(shape=(30, img_shape[0], img_shape[1], 3), dtype=np.uint8, buffer=shared_frame_buf.buf)
    shared_frame_pop_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_pop_idx_buf.buf)
    shared_frame_push_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_push_idx_buf.buf)
    shared_frame_rotation_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_rotation_idx_buf.buf)

    start = time.time()
    while cam.isOpened():
        ret, frame = cam.read()
        # cv2.imshow('image', frame)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow(winname="test", mat=image)
        # print("kakao")

        # semaphore.acquire()
        # print(f'shared memory - img : ', shared_frame)
        # print(f'img_save: pop - img : ', shared_frame_pop_idx[0], end=", ")
        # print(f'push - img : ', shared_frame_push_idx[0], end=", ")
        # print(f'rotation - img : ', shared_frame_rotation_idx[0])

        # with lock:
            # push가 한 바퀴 돌아서, pop 인덱스랑 push 인덱스가 같으면 세마포어 반납
        if shared_frame_rotation_idx[0] == 1 and (shared_frame_pop_idx[0] == shared_frame_push_idx[0]):
            # semaphore.release()
            # print("==================")
            continue

        # 프레임 넣기
        np.copyto(shared_frame[shared_frame_push_idx[0]], image)
        # push인덱스 갱신
        # shared_frame_push_idx[0] = (shared_frame_push_idx[0] + 1) % 30
        # shared_frame_push_idx[0] = shared_frame_push_idx[0] + 1
        if shared_frame_push_idx[0] + 1 >= 30:
            shared_frame_rotation_idx[0] = 1
        shared_frame_push_idx[0] = (shared_frame_push_idx[0] + 1) % 30
            # print(
            #     f"is_shared_values: push_idx_{shared_frame_push_idx[0]}, pop_idx_{shared_frame_pop_idx[0]}, rot_idx_{shared_frame_rotation_idx[0]}")
            # push가 마지막 인덱스에 위치해 있고, 아직 한바퀴 돌지 않았으면, 회전 정보 표시
            # if shared_frame_push_idx[0] >= 30 and shared_frame_rotation_idx[0] == 0:
            #     shared_frame_rotation_idx[0] = 1
            #     shared_frame_push_idx[0] = 0

        # semaphore.release()
        # time.sleep(0.25)

    end = time.time()
    # print(f'time : {end-start}')
    cam.release()
    cv2.destroyAllWindows()

def pe_run(lock):
    pe = PoseEstimation(shared_frame_name="img_shared",
                        shared_frame_pop_idx_name="shared_frame_pop_idx",
                        shared_frame_push_idx_name="shared_frame_push_idx",
                        shared_frame_rotation_idx_name="shared_frame_rotation_idx",
                        shared_position_name="shared_position",
                        shared_box_name="shared_box",
                        # semaphore=semaphore_pe,
                        img_shape=(480, 640),
                        lock=lock)
    pe.run()


if __name__ == "__main__":
    try:
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
        # img_semaphore = Semaphore(1)

        # image 공유 메모리 생성
        img_size = (480, 640)

        try:
            img_shared_frame = shared_memory.SharedMemory(name="img_shared")
        except FileNotFoundError:
            img_shared_frame = shared_memory.SharedMemory(create=True, name="img_shared",
                                                          size=img_size[0] * img_size[1] * 3 * 30 * 8)

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
            shared_frame_rotation_idx = shared_memory.SharedMemory(name="shared_frame_rotation_idx")
        except FileNotFoundError:
            shared_frame_rotation_idx = shared_memory.SharedMemory(create=True, name="shared_frame_rotation_idx",
                                                                   size=1)

        # 공유메모리 numpy.ndarray로 변환
        # shared_frame = np.ndarray(shape=(30, img_size[0], img_size[1], 3), dtype=np.uint8, buffer=img_shared_frame.buf)
        # frame_push_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_push_idx.buf)
        # frame_pop_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_pop_idx.buf)
        # frame_rotation_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_rotation_idx.buf)

        # position 공유 메모리 생성
        try:
            shared_position_memory = shared_memory.SharedMemory(name="shared_position")
        except FileNotFoundError:
            shared_position_memory = shared_memory.SharedMemory(create=True, name="shared_position", size=1056)

        # YOLO-box 공유 메모리 생성
        try:
            shared_box_memory = shared_memory.SharedMemory(name="shared_box")
        except FileNotFoundError:
            shared_box_memory = shared_memory.SharedMemory(create=True, name="shared_box", size=32)

        # 공유 메모리 numpy.ndarray로 변환
        # shared_position = np.ndarray(shape=(33, 4), dtype=np.float64, buffer=shared_position_memory.buf)
        # shared_box = np.ndarray(shape=(4), dtype=np.float64, buffer=shared_box_memory.buf)
        #
        # semaphore = Semaphore()
        # semaphore_pe = Semaphore()
        print("make semaphore")

        # 프로세스 리스트 생성
        procs = []

        # print("test_before pe")

        # PoseEstimation 객체 생성
        # pe = PoseEstimation(shared_frame=shared_frame, shared_frame_pop_idx=frame_pop_idx, shared_frame_push_idx=frame_push_idx, shared_frame_rotation_idx=frame_rotation_idx, shared_position=shared_position, shared_box=shared_box, semaphore=semaphore)
        # pe = PoseEstimation(shared_frame_name="img_shared",
        #                     shared_frame_pop_idx_name="shared_frame_pop_idx",
        #                     shared_frame_push_idx_name="shared_frame_push_idx",
        #                     shared_frame_rotation_idx_name="shared_frame_rotation_idx",
        #                     shared_position_name="shared_position",
        #                     shared_box_name="shared_box",
        #                     # semaphore=semaphore_pe,
        #                     img_shape=img_size)

        print("make semaphore - pe")

        time.sleep(10)

        # print("test_after pe")

        # ESPConnection 객체 생성
        # ec = ESPConnection(shared_frame=shared_frame, img_size=(480, 640), serverPort=4703, shared_frame_pop_idx=frame_pop_idx, shared_frame_push_idx=frame_push_idx, shared_frame_rotation_idx=frame_rotation_idx)

        # 드론 ssid, password
        ssid = ""
        password = ""

        # DroneController 객체 생성
        # dc = DroneController(ssid=ssid, password=password, shared_position=shared_position, shared_box=shared_box, standard_box=300, critical_value=10, img_size=img_size, sensor_size=(3.6, 3.6), subject_height=1.8)

        lock = Lock()

        # # 이미지 저장 프로세스
        img_process = Process(target=img_save,
                              args=("img_shared", "shared_frame_rotation_idx",
                                    "shared_frame_pop_idx",
                                    "shared_frame_push_idx",
                                    img_size,
                                    lock),
                              name="test_img_save")
        procs.append(img_process)
        img_process.start()

        # print("make semaphore - img")

        # time.sleep(4)

        # pe 프로세스 생성
        # pe_process = Process(target=pe.run, name="pose_estimation")
        pe_process = Process(target=pe_run, args=(lock, ), name="pose_estimation")
        procs.append(pe_process)
        # print("make semaphore - before start pe.run")
        pe_process.start()
        # print("make semaphore - pe.run")

        # ec 프로세스 생성
        # ec_process = Process(target=ec.run, name="esp_connection")
        # procs.append(ec_process)
        # ec_process.start()

        # dc 프로세스 생성
        # dc_process = Process(target=dc.run, name="drone_controller")
        # procs.append(dc_process)
        # dc_process.start()

        # cam = cv2.VideoCapture(0)

        # for p in procs:
        #     p.start()

        for p in procs:
            p.join()
        # while True:
        #     pass

    except KeyboardInterrupt as e:
        print(e)
    finally:
        img_shared_frame.close()
        shared_frame_pop_idx.close()
        shared_frame_push_idx.close()
        shared_frame_rotation_idx.close()
