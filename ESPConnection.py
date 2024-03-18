from socket import *
from multiprocessing import shared_memory as sm
import numpy as np
import cv2
import copy


def ESPConnection(shared_memories: dict, ip: str, serverPort: str, img_size: tuple):
    # shared_frame_name: str, shared_frame_push_idx_name: str, shared_frame_pop_idx_name: str, shared_frame_rotation_idx_name: str, img_size: tuple = (480, 640), serverPort: int = 4703, ip: str = ''):
    serverSocket = socket(AF_INET, SOCK_DGRAM)
    serverSocket.bind((ip, serverPort))
    buf_size = img_size[0] * img_size[1] * 3

    # print(f"buf?: {shared_memories}")

    try:
        while True:
            print("ESPConnection")
            # shared_frame, shared_frame_pop_idx, shared_frame_push_idx, shared_frame_rotation_idx = make_shared_memory(memories_info=shared_memories)
            # print(shared_frame_pop_idx.shape, shared_frame_pop_idx[0])
            # shared_mem_list = [sm.SharedMemory(name=name) for name, val in shared_memories.items()]
            # shared_mem = {val[0]: np.ndarray(shape=val[1][0], dtype=val[1][1], buffer=shared_mem_list[idx].buf) for idx, val in enumerate(shared_memories.items())}

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

            # cv2.imshow('img', image)
            # cv2.waitKey(1)
                # print(image.shape)
                # print(f"shared_frame_rotation_idx: {shared_frame_pop_idx[0]}")
                # print(shared_frame_pop_idx, shared_frame_push_idx, shared_frame_rotation_idx)
                # while shared_frame_rotation_idx[0] and shared_frame_pop_idx[0] == shared_frame_push_idx[0]:
                #     # print("ESP in while")
                #     pass
            # print(shared_mem)

            print(f"shared_mem_push_idx: {shared_mem['shared_frame_push_idx'][0]}")

            if shared_mem["shared_frame_rotation_idx"][0] and shared_mem["shared_frame_pop_idx"][0] == shared_mem["shared_frame_push_idx"][0]:
                for idx in range(1, 4):
                    shared_mem_list[idx].close()
                continue

            np.copyto(shared_mem["img_shared"][shared_mem["shared_frame_push_idx"][0]], image)
            # np.copyto(shared_frame[shared_frame_push_idx[0]], image)
            if shared_mem["shared_frame_push_idx"][0] + 1 >= 5:
                shared_mem["shared_frame_rotation_idx"][0] = 1
            shared_mem["shared_frame_push_idx"][0] = (shared_mem["shared_frame_push_idx"][0] + 1) % 5

            # for idx in range(1, 4):
            #     shared_mem_list[idx]

                # print(f"message: {message}")
                # print(f"image: {image}")
    except Exception as e:
        print(f"esp_error: {e.__str__()}")


