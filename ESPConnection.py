from multiprocessing import shared_memory as sm
from socket import *

import cv2
import numpy as np


def ESPConnection(shared_memories: dict, ip: str, serverPort: str, img_size: tuple):
    serverSocket = socket(AF_INET, SOCK_DGRAM)
    serverSocket.bind((ip, serverPort))
    buf_size = img_size[0] * img_size[1] * 3

    try:
        while True:
            # print("ESPConnection")
            shared_mem_list = []
            for name, val in shared_memories.items():
                shm = sm.SharedMemory(name=name)
                shared_mem_list.append(shm)

            shared_mem = {}
            for idx, val in enumerate(shared_memories.items()):
                buf = np.ndarray(shape=val[1][0], dtype=val[1][1], buffer=shared_mem_list[idx].buf)
                shared_mem[val[0]] = buf

            message, _ = serverSocket.recvfrom(buf_size)
            message = np.frombuffer(message, dtype=np.uint8)
            image = cv2.imdecode(message, 1)

            cv2.imshow('img', image)
            cv2.waitKey(1)

            print(f"ESP-PUSH: {shared_mem['shared_frame_push_idx'][0]}")

            check = shared_mem["shared_frame_push_idx"][0] - shared_mem["shared_frame_pop_idx"][0]
            print(f"check -esp : {check}")
            if check == 10:
                for smem in shared_mem_list[1:]:
                    smem.close()
                continue

            np.copyto(shared_mem["img_shared"][shared_mem["shared_frame_push_idx"][0] % 30], image)
            shared_mem["shared_frame_push_idx"][0] = (shared_mem["shared_frame_push_idx"][0] + 1) % 60
            for smem in shared_mem_list:
                smem.close()

    except Exception as e:
        print(f"esp_error: {e.__str__()}")
