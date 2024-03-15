from DBConnection import *
from PoseEstimationOldVersion import *
from DroneController import *
from ESPConnectionOldVersion import *
from multiprocessing import Process, shared_memory
import keyboard


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
                                                          size=self.img_size[0] * self.img_size[1] * 3 * 30 * 8)
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
        PoseEstimation(shared_frame_name="img_shared",
                        shared_frame_pop_idx_name="shared_frame_pop_idx",
                        shared_frame_push_idx_name="shared_frame_push_idx",
                        shared_frame_rotation_idx_name="shared_frame_rotation_idx",
                        shared_position_name="shared_position",
                        shared_box_name="shared_box")

    def esp_connection_run(self):
        ESPConnection(shared_frame="img_shared",
                       img_size=(480, 640),
                       serverPort=4703,
                       shared_frame_pop_idx="shared_frame_pop_idx",
                       shared_frame_push_idx="shared_frame_push_idx",
                       shared_frame_rotation_idx="shared_frame_pop_idx")

    def drone_controller_run(self):
        dc = DroneController(ssid=self.ssid,
                             password=self.password,
                             shared_position="shared_position",
                             shared_box="shared_box",
                             standard_box=300,
                             critical_value=10,
                             img_size=self.img_size,
                             sensor_size=self.sensor_size,
                             subject_height=self.subject_height)

        dc.run()

    def run(self):

        self.init_shared_memory()

        processes = []

        ec_process = Process(target=self.esp_connection_run,
                             name="esp_connection")
        pe_process = Process(target=self.pose_estimation_run,
                             name="pose_estimation")
        dc_process = Process(target=self.drone_controller_run,
                             name="drone_controller")

        processes.append(ec_process)
        processes.append(pe_process)
        processes.append(dc_process)

        for p in processes:
            p.start()

        # for p in processes:
        #     p.join()

        while True:
            if keyboard.is_pressed("q"):
                for p in processes:
                    p.close()
        self.close_shared_memory()

if __name__=="__main__":
    main = server()
    main.run()
