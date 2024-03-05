from multiprocessing.shared_memory import SharedMemory
from socket import *
from multiprocessing import shared_memory
import numpy as np
import cv2
import matplotlib.pyplot as plt




import time
class ESPConnection:
    # def __init__(self, shared_frame: np.ndarray, shared_frame_push_idx: shared_memory, shared_frame_pop_idx: shared_memory, img_size: tuple = (480, 640), serverPort: int = 4703):
    def __init__(self, img_size: tuple = (480, 640), serverPort: int = 27032):
        # self.serverPort = serverPort
        self.serverSocket = socket(AF_INET, SOCK_DGRAM)
        # self.serverSocket.bind(('', self.serverPort))
        self.serverSocket.bind(('', serverPort))
        # self.shared_frame = shared_frame
        # self.shared_frame_push_idx = shared_frame_push_idx
        # self.shared_frame_pop_idx = shared_frame_pop_idx
        self.buf_size = img_size[0] * img_size[1] * 3
    def run(self):
        # cnt = 0
        # a = b''
        while True:
            try:
                message, _ = self.serverSocket.recvfrom(self.buf_size)
                message = np.frombuffer(message, dtype=np.uint8)
                img = cv2.imdecode(message, cv2.IMREAD_COLOR)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show()
                # a = b''.join([a, message])
                # a += message
                # cnt += 1
                # if cnt == 2: break
                # np.reshape(message, (338, 344, 3))
                # cv2.imshow('image', message)S[
                # print(message)
                # print(type(message))
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





        '''
        self.shared_frame활용 데이터 저장 코드 작성하기
        '''

test = ESPConnection(img_size=(338, 344), serverPort=27033)
test.run()
