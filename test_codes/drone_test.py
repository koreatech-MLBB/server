from djitellopy import Tello

drone = Tello()

drone.connect()
print(drone.get_battery())
drone.takeoff()

drone.land()