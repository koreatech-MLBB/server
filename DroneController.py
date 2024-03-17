from djitellopy import Tello
import numpy as np
from multiprocessing import shared_memory as sm

TOLERANCE_X = 5
TOLERANCE_Y = 5
SLOWDOWN_THRESHOLD_X = 20
SLOWDOWN_THRESHOLD_Y = 20
DRONE_SPEED_X = 20
DRONE_SPEED_Y = 20
SET_POINT_X = 640/2
SET_POINT_Y = 480/2

# def DroneController:
def DroneController(shared_memories: dict):
    drone = Tello(host="192.168.4.219")
    drone.connect()
    print(drone.get_battery())
    drone.takeoff()
    # standard_box = 300
    # img_size = (640, 480)  # frame
    # subject_height = 1.6
    # sensor_size = (3.6, 3.6)
    # drone_v = 10

    def cal_velocity(cx, cy):
        distanceX = cx - SET_POINT_X
        distanceY = cy - SET_POINT_Y
        up_down_velocity, front_back_velocity = 0, 0

        if distanceX < -TOLERANCE_X:
            front_back_velocity = - DRONE_SPEED_X
        elif distanceX > TOLERANCE_X:
            front_back_velocity = DRONE_SPEED_X
        else:
            front_back_velocity = 0

        if distanceY < -TOLERANCE_Y:
            up_down_velocity = DRONE_SPEED_Y
        elif distanceY > TOLERANCE_Y:
            up_down_velocity = -DRONE_SPEED_Y
        else:
            up_down_velocity = 0

        if abs(distanceX) < SLOWDOWN_THRESHOLD_X:
            front_back_velocity = front_back_velocity // 2
        if abs(distanceY) < SLOWDOWN_THRESHOLD_Y:
            up_down_velocity = up_down_velocity // 2

        return front_back_velocity, up_down_velocity

    def make_shared_memory(memories: dict):
        result = []
        for name, val in memories.items():
            mem = sm.SharedMemory(name=name)
            result.append(np.ndarray(shape=val[0], dtype=val[1], buffer=mem.buf))
        return result

    def stop():
        drone.land()
        # drone.emergency()

    try:
    # def run(cx, cy):
        while True:
            shared_position, shared_box = make_shared_memory(memories=shared_memories)
            front_back_velocity, up_down_velocity = cal_velocity(shared_box[0], shared_box[1])
            print(f"drone_controller: {front_back_velocity}, {up_down_velocity}")
            drone.send_rc_control(0, front_back_velocity, up_down_velocity, 0)
    except BaseException as e:
        print(e.__str__())
        stop()

