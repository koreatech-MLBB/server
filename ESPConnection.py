from socket import *
from multiprocessing import shared_memory
import numpy as np
import cv2
import matplotlib.pyplot as plt




import time
class ESPConnection:
    def __init__(self, shared_frame: np.ndarray, shared_frame_push_idx_name: str, shared_frame_pop_idx_name: str, shared_frame_idx_rotation_name: str, img_size: tuple = (480, 640), serverPort: int = 4703):
    # def __init__(self, img_size: tuple = (480, 640), serverPort: int = 27032):
        # self.serverPort = serverPort
        self.serverSocket = socket(AF_INET, SOCK_DGRAM)
        # self.serverSocket.bind(('', self.serverPort))
        self.serverSocket.bind(('', serverPort))
        self.shared_frame = shared_frame
        self.shared_frame_push_idx = shared_memory.SharedMemory(name=shared_frame_push_idx_name).buf.cast('i')
        self.shared_frame_pop_idx = shared_memory.SharedMemory(name=shared_frame_pop_idx_name).buf.cast('i')
        self.shared_frame_idx_rotation = shared_memory.SharedMemory(name=shared_frame_idx_rotation_name).buf.cast('i')
        self.buf_size = img_size[0] * img_size[1] * 3
    def run(self):
        while True:
            try:
                message, _ = self.serverSocket.recvfrom(self.buf_size)
                message = np.frombuffer(message, dtype=np.uint8)
                image = cv2.imdecode(message, cv2.IMREAD_COLOR)

                while self.shared_frame_idx_rotation[0] and self.shared_frame_pop_idx[0] == self.shared_frame_push_idx[0]:
                    pass

                np.copyto(self.shared_frame[self.shared_frame_push_idx[0]], image)
                if self.shared_frame_pop_idx[0] + 1 >= 30:
                    self.shared_frame_idx_rotation[0] += 1
                self.shared_frame_pop_idx[0] = (self.shared_frame_idx_rotation[0] + 1) % 30
            except Exception as e:
                print(e)
        # a = np.frombuffer(a, dtype=np.uint8)
        # img = cv2.imdecode(a, cv2.IMREAD_COLOR)
        # print(img.shape)
        # print(len(a))
        # print(a)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # cv2.imshow('img', img)
        # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # plt.show()

        # time.sleep(10)





        # '''
        # self.shared_frame활용 데이터 저장 코드 작성하기
        # '''

# test = ESPConnection(img_size=(338, 344), serverPort=27033)
# test.run()
