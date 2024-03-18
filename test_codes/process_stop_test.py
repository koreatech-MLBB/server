from multiprocessing import Process
import time

def func1(a):
    try:
        for i in range(a):
            print(i)
            time.sleep(0.1)
    except BaseException as e:
        print(f"test: {e}")
        exit()

if __name__ == "__main__":
    try:
        p = Process(target=func1, args=(20000, ), name="test", daemon=True)
        p.start()
        p.join()
    except BaseException as e:
        print(f"main: {e}")
        p.terminate()