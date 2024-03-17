from djitellopy import Tello
import keyboard

drone = Tello(host="192.168.4.219")

try:
    drone.connect()
    print(drone.get_battery())
    drone.takeoff()

    # drone.send_rc_control(left_right_velocity=-5, forward_backward_velocity=10, up_down_velocity=10, yaw_velocity=0)

    while True:
        if keyboard.is_pressed('q'):
            drone.land()
            break
except KeyboardInterrupt as e:
    print(e)
    drone.land()
except Exception as e:
    print(e)
    drone.land()
