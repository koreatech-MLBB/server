from socket import *
from multiprocessing import shared_memory as sm
import numpy as np
import cv2

def ESPConnection(shared_memories: dict, ip: str, serverPort: str, img_size: tuple):
        # shared_frame_name: str, shared_frame_push_idx_name: str, shared_frame_pop_idx_name: str, shared_frame_rotation_idx_name: str, img_size: tuple = (480, 640), serverPort: int = 4703, ip: str = ''):
    serverSocket = socket(AF_INET, SOCK_DGRAM)
    serverSocket.bind((ip, serverPort))
    buf_size = img_size[0] * img_size[1] * 3

    def make_shared_memory(memories: dict):
        for name, val in memories.items():
            yield np.ndarray(shape=val[0], dtype=val[1], buffer=sm.SharedMemory(name=name).buf)

    while True:

        shared_frame, shared_frame_pop_idx, shared_frame_push_idx, shared_frame_rotation_idx = make_shared_memory(memories=shared_memories)

        try:
            message, _ = serverSocket.recvfrom(buf_size)
            message = np.frombuffer(message, dtype=np.uint8)
            image = cv2.imdecode(message, cv2.IMREAD_COLOR)

            while shared_frame_rotation_idx[0] and shared_frame_pop_idx[0] == shared_frame_push_idx[0]:
                pass

            np.copyto(shared_frame[shared_frame_push_idx[0]], image)
            if shared_frame_pop_idx[0] + 1 >= 30:
                shared_frame_rotation_idx[0] += 1
            shared_frame_pop_idx[0] = (shared_frame_rotation_idx[0] + 1) % 30

            print(f"message: {message}")
            print(f"image: {image}")

        except Exception as e:
            print(e)
