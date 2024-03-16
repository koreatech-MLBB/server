from djitellopy import Tello

TOLERANCE_X = 5
TOLERANCE_Y = 5
SLOWDOWN_THRESHOLD_X = 20
SLOWDOWN_THRESHOLD_Y = 20
DRONE_SPEED_X = 20
DRONE_SPEED_Y = 20
SET_POINT_X = 640/2
SET_POINT_Y = 480/2


class DroneController:
    def __init__(self):
        self.drone = Tello()
        self.drone.connect()
        print(self.drone.get_battery())
        self.drone.takeoff()
        self.standard_box = 300
        self.img_size = (640, 480)  # frame
        self.subject_height = 1.6
        self.sensor_size = (3.6, 3.6)
        self.drone_v = 10

    def cal_velocity(self, cx, cy):
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
            front_back_velocity = front_back_velocity//2
        if abs(distanceY) < SLOWDOWN_THRESHOLD_Y:
            up_down_velocity = up_down_velocity//2

        return front_back_velocity, up_down_velocity

    def run(self, cx, cy):
        front_back_velocity, up_down_velocity = self.cal_velocity(cx, cy)
        self.drone.send_rc_control(0, front_back_velocity, up_down_velocity, 0)

    def stop(self):
        self.drone.emergency()