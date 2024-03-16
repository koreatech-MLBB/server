from socket import *
from multiprocessing import shared_memory
import numpy as np
import cv2
import matplotlib.pyplot as plt




import time
class ESPConnection:
    def __init__(self, shared_frame: np.ndarray, shared_frame_push_idx: np.ndarray, shared_frame_pop_idx: np.ndarray, shared_frame_rotation_idx: np.ndarray, img_size: tuple = (480, 640), serverPort: int = 4703, ip: str = ''):
    # def __init__(self, img_size: tuple = (480, 640), serverPort: int = 27032, ip: str = ''):
    #     self.serverPort = serverPort
        self.serverSocket = socket(AF_INET, SOCK_DGRAM)
        # self.serverSocket.bind(('', self.serverPort))
        self.serverSocket.bind((ip, serverPort))
        self.shared_frame = shared_frame
        self.shared_frame_pop_idx = shared_frame_pop_idx
        self.shared_frame_push_idx = shared_frame_push_idx
        self.shared_frame_rotation_idx = shared_frame_rotation_idx
        self.buf_size = img_size[0] * img_size[1] * 3
    def run(self):
        while True:
            try:
                message, _ = self.serverSocket.recvfrom(self.buf_size)
                message = np.frombuffer(message, dtype=np.uint8)
                image = cv2.imdecode(message, cv2.IMREAD_COLOR)

                while self.shared_frame_rotation_idx[0] and self.shared_frame_pop_idx[0] == self.shared_frame_push_idx[0]:
                    pass

                np.copyto(self.shared_frame[self.shared_frame_push_idx[0]], image)
                if self.shared_frame_pop_idx[0] + 1 >= 30:
                    self.shared_frame_rotation_idx[0] += 1
                self.shared_frame_pop_idx[0] = (self.shared_frame_rotation_idx[0] + 1) % 30

                print(f"message: {message}")
                print(f"image: {image}")

            except Exception as e:
                print(e)

# test = ESPConnection(img_size=(320, 240), serverPort=3333)
# test.run()
