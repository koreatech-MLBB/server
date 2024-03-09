from PoseVal import pose_val
from djitellopy import Tello
import numpy as np
import math
class DroneController:
    def __init__(self, ssid: str, password: str, shared_position: np.ndarray, shared_box: np.ndarray, standard_box: int, critical_value: float, img_size: tuple, sensor_size: tuple, subject_height: float, k_yaw: float):
        # self.drone = Tello(ssid=ssid, password=password)
        self.drone = Tello()
        self.drone.connect()
        self.shared_position = shared_position
        self.shared_box = shared_box
        self.standard_box = standard_box
        self.critical_value = critical_value
        self.img_size = img_size
        self.subject_height = subject_height
        self.sensor_size = sensor_size
        self.k_yaw = k_yaw
        self.t0 = 0
        self.d0 = 0

    def calc_move(self):
        size_x = (((self.img_size[0] // 2) - self.shared_box[0]) * ((self.shared_box[3] * self.subject_height) / (self.sensor_size[0] * self.sensor_size[1]))) / self.img_size[0]
        size_y = (self.subject_height * (self.standard_box - self.shared_box[3])) / (self.sensor_size[0] * self.sensor_size[1])
        size_z = (((self.img_size[1] // 2) - self.shared_box[1]) * ((self.shared_box[3] * self.subject_height) / (self.sensor_size[0] * self.sensor_size[1]))) / self.img_size[1]

        rc_pitch = (math.sqrt((size_x**2) + (size_y**2)) * ((self.shared_box[3] * self.subject_height) / (self.sensor_size[0] * self.sensor_size[1])) * math.sin(math.atan(size_y/size_x))) / (self.drone.t)

    def run(self):
        while True:
            # if self.shared_box[0] < 0.5 - self.critical_value:
            #     self.drone.send_command
            pass