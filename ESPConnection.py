from socket import *
from multiprocessing import shared_memory as sm
import numpy as np
import cv2


def ESPConnection(shared_memories: dict, ip: str, serverPort: str, img_size: tuple):
    # shared_frame_name: str, shared_frame_push_idx_name: str, shared_frame_pop_idx_name: str, shared_frame_rotation_idx_name: str, img_size: tuple = (480, 640), serverPort: int = 4703, ip: str = ''):
    serverSocket = socket(AF_INET, SOCK_DGRAM)
    serverSocket.bind((ip, serverPort))
    buf_size = img_size[0] * img_size[1] * 3

    print(f"buf?: {shared_memories}")

    def make_shared_memory(memories: dict):
        # result = ['', np.zeros(1, ), np.zeros(1, ), np.zeros(1, )]
        # result = []
        # print(result)
        try:
            for name, val in memories.items():
                # print(name, val)
                mem = sm.SharedMemory(name=name)
                print(f"val : {val[0]}, {val[1]}")
                re = np.ndarray(shape=val[0], dtype=val[1], buffer=mem.buf)
                print(f"test: {type(re)}, {re.shape}")
                # result.append(re)
                yield re
                # print(result)
                # print(f"in def: {result[-1][0]}")
            # for x in result:
            #     print(f"res: {x}")
            # return result[0][:], result[1][:], result[2][:], result[3][:]
        except BaseException as e:
            print(f"error: {e.__str__()}")

    while True:
        print("ESPConnection")
        shared_frame, shared_frame_pop_idx, shared_frame_push_idx, shared_frame_rotation_idx = make_shared_memory(
            memories=shared_memories)
        print(shared_frame.tolist())
        # shared_frame, shared_frame_pop_idx, shared_frame_push_idx, shared_frame_rotation_idx = None, None, None, None
        # shared_mem = {""}
        # for name, val in shared_memories.items():
        # print(name, val)
        # mem = sm.SharedMemory(name=name)
        # np.ndarray(shape=val[0], dtype=val[1], buffer=mem.buf)
        # print(type(shared_frame), shared_frame_rotation_idx[0])
        # print(make_shared_memory(memories=shared_memories))
        try:
            message, _ = serverSocket.recvfrom(buf_size)
            message = np.frombuffer(message, dtype=np.uint8)
            image = cv2.imdecode(message, 1)

            cv2.imshow('img', image)
            print(image.shape)
            print(f"shared_frame_rotation_idx: {shared_frame_pop_idx[0]}")
            while shared_frame_rotation_idx[0] and shared_frame_pop_idx[0] == shared_frame_push_idx[0]:
                print("ESP in while")
                pass

            np.copyto(shared_frame[shared_frame_push_idx[0]], image)
            if shared_frame_pop_idx[0] + 1 >= 30:
                shared_frame_rotation_idx[0] += 1
            shared_frame_pop_idx[0] = (shared_frame_rotation_idx[0] + 1) % 30

            print(f"message: {message}")
            print(f"image: {image}")

        except Exception as e:
            print(e)
