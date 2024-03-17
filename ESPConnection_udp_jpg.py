
from socket import *
from multiprocessing import shared_memory
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

class ESPConnection:
    # def __init__(self, shared_frame: np.ndarray, shared_frame_push_idx: np.ndarray, shared_frame_pop_idx: np.ndarray, shared_frame_rotation_idx: np.ndarray, img_size: tuple = (480, 640), serverPort: int = 4703):
    def __init__(self, img_size: tuple = (480, 640), serverPort: int = 27032, ip: str = ''):
        self.serverSocket = socket(AF_INET, SOCK_DGRAM)
        self.serverSocket.bind((ip, serverPort))
        self.buf_size = img_size[0] * img_size[1] * 3
    def run(self):
        while True:
            try:
                message, _ = self.serverSocket.recvfrom(self.buf_size)
                message = np.frombuffer(message, dtype=np.uint8)
                decoded_image = cv2.imdecode(message, 1)
                cv2.imshow('img', decoded_image)

                if cv2.waitKey(1) != -1:
                    break

            except Exception as e:
                print(e)
                # raise e
        cv2.destroyAllWindows()

test = ESPConnection(img_size=(320, 240), serverPort=3333)
test.run()