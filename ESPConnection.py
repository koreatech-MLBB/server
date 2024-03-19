from multiprocessing import shared_memory as sm, Semaphore
from socket import *

import cv2
import numpy as np


def ESPConnection(shared_memories: dict, sem: Semaphore, ip: str, serverPort: int, img_size: tuple):
    serverSocket = socket(AF_INET, SOCK_DGRAM)
    serverSocket.bind((ip, serverPort))
    buf_size = img_size[0] * img_size[1] * 3

    try:
        while True:
            print("ESPConnection")
            message, _ = serverSocket.recvfrom(buf_size)
            message = np.frombuffer(message, dtype=np.uint8)
            image = cv2.imdecode(message, 1)

            with sem:
                shared_mem_list = []
                for name, val in shared_memories.items():
                    shm = sm.SharedMemory(name=name)
                    shared_mem_list.append(shm)

                shared_mem = {}
                for idx, val in enumerate(shared_memories.items()):
                    buf = np.ndarray(shape=val[1][0], dtype=val[1][1], buffer=shared_mem_list[idx].buf)
                    shared_mem[val[0]] = buf

                flag = False
                while not flag:
                    # 뽑을 수 있는 상태라면, flag -> Ture
                    print(f"push : {shared_mem['shared_frame_push_idx'][0]}")
                    if shared_mem["shared_frame_push_idx"][0] >= 30:  # (30 ~ 59)
                        flag = shared_mem["shared_frame_push_idx"][0] - 30 <= shared_mem["shared_frame_pop_idx"][0]
                    else:
                        flag = True

                print(f"ESP - PUSH: {shared_mem['shared_frame_push_idx'][0]}")

                np.copyto(shared_mem["img_shared"][shared_mem["shared_frame_push_idx"][0] % 30], image)
                shared_mem["shared_frame_push_idx"][0] = (shared_mem["shared_frame_push_idx"][0] + 1) % 60

                cv2.imshow('img', image)
                cv2.waitKey(1)

    except Exception as e:
        print(f"esp_error: {e.__str__()}")
