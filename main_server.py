# from DBConnection import *
from PoseEstimation import *
from DroneController import *
from ESPConnection import *
from multiprocessing import Process, shared_memory
import keyboard
import numpy as np


class server:
    def __init__(self):

        self.img_size = (480, 640)

        # drone
        self.ssid = ""
        self.password = ""
        self.sensor_size = (3.6, 3.6)
        self.subject_height = 1.8

        self.shared_memories = []

    def init_shared_memory(self):
        print("Initialize shared memories")
        try:
            img_shared_frame = shared_memory.SharedMemory(name="img_shared")
        except FileNotFoundError:
            img_shared_frame = shared_memory.SharedMemory(create=True,
                                                          name="img_shared",
                                                          size=self.img_size[0] * self.img_size[1] * 3 * 5 * 8)
        # 이미지 push index 공유 메모리 생성
        try:
            shared_frame_push_idx = shared_memory.SharedMemory(name="shared_frame_push_idx")
        except FileNotFoundError:
            shared_frame_push_idx = shared_memory.SharedMemory(create=True,
                                                               name="shared_frame_push_idx",
                                                               size=1)

        # 이미지 pop index 공유 메모리 생성
        try:
            shared_frame_pop_idx = shared_memory.SharedMemory(name="shared_frame_pop_idx")
        except FileNotFoundError:
            shared_frame_pop_idx = shared_memory.SharedMemory(create=True,
                                                              name="shared_frame_pop_idx",
                                                              size=1)

        # 이미지 push/pop index 상태 공유 메모리 생성
        try:
            shared_frame_rotation_idx = shared_memory.SharedMemory(name="shared_frame_rotation_idx")
        except FileNotFoundError:
            shared_frame_rotation_idx = shared_memory.SharedMemory(create=True,
                                                                   name="shared_frame_rotation_idx",
                                                                   size=1)

        # position 공유 메모리 생성
        try:
            shared_position_memory = shared_memory.SharedMemory(name="shared_position")
        except FileNotFoundError:
            shared_position_memory = shared_memory.SharedMemory(create=True,
                                                                name="shared_position",
                                                                size=1056)

        # YOLO-box 공유 메모리 생성
        try:
            shared_box_memory = shared_memory.SharedMemory(name="shared_box")
        except FileNotFoundError:
            shared_box_memory = shared_memory.SharedMemory(create=True,
                                                           name="shared_box",
                                                           size=32)

        # shared_frame_buf = np.ndarray(shape=(5, 480, 640, 3), dtype=np.uint8, buffer=img_shared_frame.buf)
        shared_frame_rotation_idx_buf = np.ndarray(shape=(1, ), dtype=np.uint8, buffer=shared_frame_rotation_idx.buf)
        shared_frame_push_idx_buf = np.ndarray(shape=(1, ), dtype=np.uint8, buffer=shared_frame_push_idx.buf)
        shared_frame_pop_idx_buf = np.ndarray(shape=(1, ), dtype=np.uint8, buffer=shared_frame_pop_idx.buf)
        # shared_box_memory_buf = np.ndarray(shape=(5, 480, 640, 3), dtype=np.uint8, buffer=img_shared_frame.buf)
        # shared_position_memory_buf = np.ndarray(shape=(5, 480, 640, 3), dtype=np.uint8, buffer=img_shared_frame.buf)

        shared_frame_pop_idx_buf[0] = 0
        shared_frame_push_idx_buf[0] = 0
        shared_frame_rotation_idx_buf[0] = 0

        # print(shared_frame_rotation_idx_buf[0])

        self.shared_memories.append(img_shared_frame)
        self.shared_memories.append(shared_frame_rotation_idx)
        self.shared_memories.append(shared_frame_push_idx)
        self.shared_memories.append(shared_frame_pop_idx)
        self.shared_memories.append(shared_box_memory)
        self.shared_memories.append(shared_position_memory)

    def close_shared_memory(self):
        for sm in self.shared_memories:
            sm.close()

    def pose_estimation_run(self):
        PoseEstimation(shared_memories={"img_shared": [(30, 480, 640, 3), np.uint8],
                                        "shared_frame_pop_idx": [(1,), np.uint8],
                                        "shared_frame_push_idx": [(1,), np.uint8],
                                        "shared_frame_rotation_idx": [(1,), np.uint8],
                                        "shared_position": [(33, 4), np.float64],
                                        "shared_box": [(4,), np.float64]})

    def esp_connection_run(self):
        ESPConnection(shared_memories={"img_shared": [(30, 480, 640, 3), np.uint8],
                                       "shared_frame_pop_idx": [(1,), np.uint8],
                                       "shared_frame_push_idx": [(1,), np.uint8],
                                       "shared_frame_rotation_idx": [(1,), np.uint8]},
                      img_size=(480, 640),
                      serverPort=3333,
                      ip='')

    def drone_controller_run(self):
        DroneController(shared_memories={"shared_position": [(33, 4), np.float64],
                                         "shared_box": [(4,), np.float64]})

    def run(self):

        self.init_shared_memory()

        processes = []

        ec_process = Process(target=self.esp_connection_run,
                             name="esp_connection")
        pe_process = Process(target=self.pose_estimation_run,
                             name="pose_estimation")
        # dc_process = Process(target=self.drone_controller_run,
        #                      name="drone_controller")

        processes.append(ec_process)
        processes.append(pe_process)
        # processes.append(dc_process)

        for p in processes:
            p.start()

        # for p in processes:
        #     p.join()

        try:
            while True:

                # mem = sm.SharedMemory(name="img_shared")
                # rot = np.ndarray(shape=(1, ), dtype=np.uint8, buffer=self.shared_memories[2].buf)
                # print(f"in main: {rot[0]}")
                #
                # for i in range(5):
                #     cv2.imshow("main", img[i])

                if keyboard.is_pressed("q"):
                    raise Exception("q가 눌림")
        except BaseException as e:
            print(f"main_procs: {e.__str__()}")
            # self.close_shared_memory()
            for p in processes:
                p.terminate()


if __name__ == "__main__":
    main = server()
    main.run()
