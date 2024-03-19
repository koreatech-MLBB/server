from multiprocessing import shared_memory as sm, Process, shared_memory, Semaphore
from socket import *

import cv2
import keyboard
import numpy as np
import time

# TODO: push 딴에서 기다리도록 코드 작성하기 !!!!!!!!!!

# pose 함수고
def func1(names, sem):
    while True:

        with sem:
            shared_image = sm.SharedMemory(name=names[0])
            frame_mem = np.ndarray(shape=(30, 480, 640, 3), dtype=np.uint8, buffer=shared_image.buf)
            push_buf = sm.SharedMemory(name=names[1])
            push_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=push_buf.buf)
            pop_buf = sm.SharedMemory(name=names[2])
            pop_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=pop_buf.buf)

            flag = False
            while not flag:
                # 뽑을 수 있는 상태라면, flag -> Ture
                if push_idx[0] < 30:  # (0~ 29)
                    flag = pop_idx[0] < push_idx[0]
                else:  # (30 ~ 59)
                    flag = push_idx[0] % 30 <= pop_idx[0]
                print(f"pop, push : {pop_idx[0]}, {push_idx[0]}")
            print(f'image idx - pose : {pop_idx}')
            image = frame_mem[pop_idx[0]][:]
            pop_idx[0] = (pop_idx[0] + 1) % 30

        cv2.imshow('pose', image)
        cv2.waitKey(1)


# esp 함수
def func2(names, sem):
    serverSocket = socket(AF_INET, SOCK_DGRAM)
    serverSocket.bind(('', 3333))
    buf_size = 480 * 640 * 3
    while True:
        시작 = time.time()
        message, _ = serverSocket.recvfrom(buf_size)
        message = np.frombuffer(message, dtype=np.uint8)
        image = cv2.imdecode(message, 1)

        with sem:
            shared_image = sm.SharedMemory(name=names[0])
            frame_mem = np.ndarray(shape=(30, 480, 640, 3), dtype=np.uint8, buffer=shared_image.buf)
            push_buf = sm.SharedMemory(name=names[1])
            push_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=push_buf.buf)
            pop_buf = sm.SharedMemory(name=names[2])
            pop_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=pop_buf.buf)

            frame_mem[push_idx[0] % 30][:] = image
            push_idx[0] = (push_idx[0] + 1) % 60
        종료 = time.time()
        print(f'image idx - esp : {push_idx}')
        print(f'time to say good bye : {종료 - 시작}')
        cv2.imshow('esp', image)
        cv2.waitKey(1)


if __name__ == "__main__":
    print("start")
    try:
        img_shared_frame = shared_memory.SharedMemory(name="img_shared")
    except FileNotFoundError:
        img_shared_frame = shared_memory.SharedMemory(create=True,
                                                      name="img_shared",
                                                      size=480 * 640 * 3 * 5 * 8)
    try:
        shared_frame_push_idx = shared_memory.SharedMemory(name="shared_frame_push_idx")
    except FileNotFoundError:
        shared_frame_push_idx = shared_memory.SharedMemory(create=True,
                                                           name="shared_frame_push_idx",
                                                           size=1)
    try:
        shared_frame_pop_idx = shared_memory.SharedMemory(name="shared_frame_pop_idx")
    except FileNotFoundError:
        shared_frame_pop_idx = shared_memory.SharedMemory(create=True,
                                                          name="shared_frame_pop_idx",
                                                          size=1)
    names = ["img_shared", "shared_frame_push_idx", 'shared_frame_pop_idx']
    sem = Semaphore(2)

    p1 = Process(target=func1, args=(names, sem),
                 name="esp")
    p2 = Process(target=func2, args=(names, sem),
                 name='pose')

    p2.start()
    p1.start()

    # p1.join()
    # p2.join()

    try:
        while True:
            if keyboard.is_pressed("q"):
                raise Exception("q가 눌림")
    except BaseException as e:
        print(f"main_procs: {e.__str__()}")
        p1.terminate()
        p2.terminate()
