from DBConnection import *
from PoseEstimation import *
from DroneController import *
from ESPConnection import *
from multiprocessing import Process, shared_memory
import numpy as np
import time

class server:
    def __init__(self):