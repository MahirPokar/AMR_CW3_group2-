from djitellopy import tello
import cv2
import time 

def main():
    drone = tello.Tello(host="192.168.10.110")
    drone.connect(False) # If Tello does not send OK response you can try: drone.connect(False)
    drone.takeoff()
    # Write your motion control
    #using drone.rc

    drone.send_rc_control(0, 50, 0, 0) 
    time.sleep(2)
    drone.send_rc_control(0,0,0,0)
    time.sleep(2)
    drone.send_rc_control(50, 0, 0, 0)
    time.sleep(2)
    drone.send_rc_control(0,0,0,0)
    time.sleep(2)
    drone.send_rc_control(0, -50, 0, 0)
    time.sleep(2)
    drone.send_rc_control(0,0,0,0)
    time.sleep(2)
    drone.send_rc_control(-50, 0, 0, 0)
    time.sleep(2)
    drone.send_rc_control(0,0,0,0)
    time.sleep(2)

    #using drone.move
    # drone.move("forward",100)
    # drone.move("right", 100)
    # drone.move("back", 100)
    # drone.move("left", 100)

    drone.land()
    drone.end()

    key = cv2.waitKey(1) & 0xFF 

    if key == 27:
        print("Interrupted, drone is landing...")
        drone.land()
        drone.end()
        return 0
    drone.land()
    drone.end()
    return 0
if __name__ == "__main__":
    main()


