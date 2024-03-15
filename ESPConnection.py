from socket import *
from multiprocessing import shared_memory as sm
import numpy as np
import cv2

def ESPConnection(shared_frame_name: str, shared_frame_push_idx_name: str, shared_frame_pop_idx_name: str, shared_frame_rotation_idx_name: str, img_size: tuple = (480, 640), serverPort: int = 4703, ip: str = ''):
    serverSocket = socket(AF_INET, SOCK_DGRAM)
    serverSocket.bind((ip, serverPort))
    buf_size = img_size[0] * img_size[1] * 3
    while True:
        shared_frame_buf = sm.SharedMemory(name=shared_frame_name)
        shared_frame_pop_idx_buf = sm.SharedMemory(name=shared_frame_pop_idx_name)
        shared_frame_push_idx_buf = sm.SharedMemory(name=shared_frame_push_idx_name)
        shared_frame_rotation_idx_buf = sm.SharedMemory(name=shared_frame_rotation_idx_name)

        shared_frame = np.ndarray(shape=(30, 480, 640, 3), dtype=np.uint8, buffer=shared_frame_buf.buf)
        shared_frame_pop_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_pop_idx_buf.buf)
        shared_frame_push_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_push_idx_buf.buf)
        shared_frame_rotation_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_rotation_idx_buf.buf)
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
