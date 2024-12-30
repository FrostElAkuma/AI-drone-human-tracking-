from utils import *
import cv2

w,h = 640, 360
#kp,kd,ki
#these are random values that I tried, tracking can be cmothened more with diffirent values
#these values ar just for rotation
pid = [0.5,0.5,0]
pError = 0
startCounter = 1 #for testing drone without flying set the value to 1 and for flight set it to 0 


drone1 = initializaTello()

#while loop to iterate over the frames 
while True:

    #Initial take off, then startCounter will be 1 again after the initial first loop so this If condition will only get called once
    if startCounter == 0:
        print("Taking off...")
        drone1.takeoff()
        target_height = 29  # Set target height (in cm or Tello's unit)
        drone1.send_rc_control(0, 0, target_height, 0)  # Move to target height
        startCounter = 1

    #step 1, getting and displlaying video stream
    img = telloGetFrame(drone1, w, h)

    #step 2, running our detection algorathim with the help of YOLO
    #We have an issue here is what if there are multiple people who will the drone follow?
    #That's why we will use our own model instead of the coco pre trained model
    #img= findPerson(img)
    img, info = findPerson(img, drone1)
    #this will print the cx
    print("This is cx",info[0][0])

    #step 3
    pError = trackPerson(drone1, info, w, pid, pError)

    cv2.imshow('Image', img)
    #command to stop drone. Saftey is important I already injured my fingures 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        drone1.land()
        break