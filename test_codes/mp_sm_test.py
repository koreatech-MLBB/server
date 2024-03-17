# from multiprocessing import shared_memory as sm, Process
# import numpy as np
# import time
#
#
# def func1(sm_push_name, sm_pop_name, sm_rot_name):
#     shared_frame_rotation_idx = sm.SharedMemory(name=sm_rot_name)
#     shared_frame_pop_idx = sm.SharedMemory(name=sm_pop_name)
#     shared_frame_push_idx = sm.SharedMemory(name=sm_push_name)
#
#     frame_rotation_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_rotation_idx.buf)
#     frame_pop_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_pop_idx.buf)
#     frame_push_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_push_idx.buf)
#     while True:
#         if frame_pop_idx[0] + 1 >= 30:
#             frame_rotation_idx[0] -= 1
#         frame_pop_idx[0] = (frame_pop_idx[0] + 1) % 30
#
#         print(f"func1: rot_{frame_rotation_idx[0]}, pop_{frame_pop_idx[0]}, push_{frame_push_idx[0]}")
#         time.sleep(0.1)
#
#
# def func2(sm_push_name, sm_pop_name, sm_rot_name):
#     # shared_frame_buf = sm.SharedMemory(name=shared_frame)
#     shared_frame_rotation_idx = sm.SharedMemory(name=sm_rot_name)
#     shared_frame_pop_idx = sm.SharedMemory(name=sm_pop_name)
#     shared_frame_push_idx = sm.SharedMemory(name=sm_push_name)
#
#     # shared_frame = np.ndarray(shape=())
#     frame_rotation_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_rotation_idx.buf)
#     frame_pop_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_pop_idx.buf)
#     frame_push_idx = np.ndarray(shape=(1,), dtype=np.uint8, buffer=shared_frame_push_idx.buf)
#
#     while True:
#         if frame_push_idx[0] + 1 >= 30:
#             frame_rotation_idx[0] += 1
#         frame_push_idx[0] = (frame_push_idx[0] + 1) % 30
#
#         print(f"func2: rot_{frame_rotation_idx[0]}, pop_{frame_pop_idx[0]}, push_{frame_push_idx[0]}")
#         time.sleep(0.1)
#
# if __name__ == "__main__":
#     print("start")
#     try:
#         shared_frame_push_idx = sm.SharedMemory(name="shared_frame_push_idx")
#     except FileNotFoundError:
#         shared_frame_push_idx = sm.SharedMemory(create=True, name="shared_frame_push_idx", size=1)
#
#     # 이미지 pop index 공유 메모리 생성
#     try:
#         shared_frame_pop_idx = sm.SharedMemory(name="shared_frame_pop_idx")
#     except FileNotFoundError:
#         shared_frame_pop_idx = sm.SharedMemory(create=True, name="shared_frame_pop_idx", size=1)
#
#     # 이미지 push/pop index 상태 공유 메모리 생성
#     try:
#         shared_frame_rotation_idx = sm.SharedMemory(name="shared_frame_rotation_idx")
#     except FileNotFoundError:
#         shared_frame_rotation_idx = sm.SharedMemory(create=True, name="shared_frame_rotation_idx", size=1)
#
#     p1 = Process(target=func1, args=("shared_frame_push_idx", "shared_frame_pop_idx", "shared_frame_rotation_idx"))
#     p2 = Process(target=func2, args=("shared_frame_push_idx", "shared_frame_pop_idx", "shared_frame_rotation_idx"))
#
#     p1.start()
#     p2.start()
#
#     p1.join()
#     p2.join()

# def test(l):
#     return l
#
# li = [1, 2, 3]
# a, b, c = test(li)
#
# print(a)


d = {'a': [(1, 2, 3, 4), "test"]}
for idx, val in enumerate(d.items()):
    print(idx, "+", val[1][0])