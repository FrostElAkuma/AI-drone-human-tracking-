from djitellopy import Tello    
import cv2 
from ultralytics import YOLO
import numpy as np

custom = 1 #TO use custom data set use 1, 0 for coco data set

# Function to load class names from a file
def load_class_names(file_path):
    with open(file_path, 'r') as f:
        class_names = f.read().strip().split('\n')
    return class_names


if custom == 1:
  class_names = load_class_names('custom.names')
else:
  class_names = load_class_names('coco.names')



# Load the YOLO model
if custom == 1:
  model = YOLO("Z_LAST.pt")
else:
  model = YOLO("yolov8s.pt")

model2 = YOLO("Z_HandSignsBest.pt")

def initializaTello():
    drone1 = Tello()
    drone1.connect()

    #We need to set all velocities to 0. Rotation is yaw 
    drone1.for_back_velocity = 0
    drone1.left_right_velocity = 0
    drone1.up_down_velocity = 0
    drone1.yaw_velocity = 0
    drone1.speed = 0
    print("Drone battery is",drone1.get_battery())
    #We are turning the stream off here so just in case if in the previous session it did not turn off properly, we do it here.
    drone1.streamoff()
    drone1.streamon()
    return drone1
    
def telloGetFrame(drone1, w = 360, h=240):
    #getting video from the drone 
    myFrame = drone1.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame,(w,h))
    return img

def findPerson(img, drone1):
    results = model(img)
    results2 = model2(img)

    #Accesses the first detection result in results, 
    #which contains the bounding boxes and class predictions, and stores it in result
    result = results[0]
    result2 = results2[0]


    #Converts the bounding box coordinates to an integer array (int). 
    #This array bboxes contains x, y, x2, and y2 coordinates for each detected object
    bboxes1 = np.array(result.boxes.xyxy.cpu(), dtype="int")
    classes1 = np.array(result.boxes.cls.cpu(), dtype="int")

    bboxes2 = np.array(result2.boxes.xyxy.cpu(), dtype="int")
    classes2 = np.array(result2.boxes.cls.cpu(), dtype="int")

    myPersonList = []
    myPersonListArea = []

    # Get the battery level
    battery_level = drone1.get_battery()

    # Display battery status on the image
    cv2.putText(img, f"Battery: {battery_level}%", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

    for bboxes, classes in [(bboxes1, classes1), (bboxes2, classes2)]:
        for cls, bbox in zip(classes, bboxes):
            #Extracts x, y, x2, and y2 coordinates from bbox, 
            #defining the corners of the bounding box for the detected object
            (x, y, x2, y2) = bbox
            cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 225), 2)
            # Get the class name from the class index
            class_name = class_names[cls] if cls < len(class_names) else "Unknown"

            # Calculate the width and height of the bounding box
            width = x2 - x
            height = y2 - y
            area = width * height

            #center for x and y 
            cx = x + width // 2
            cy = y + height // 2

            if class_name == 'AZ' or class_name == 'person':
                myPersonListArea.append(area)
                myPersonList.append([cx, cy])
            
            if class_name == "B":
                print("Landing...")
                drone1.land()

            #I tried implementing controls using hand commands but the PID has total control over the drones movemetns
            #if class_name == "O":
                #print("Moving up...")
                #drone1.send_rc_control(0, 0, 30, 0)  # Up velocity of 20 units
                    
            # Display the class name instead of the index
            cv2.putText(img, class_name, (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 225), 2)

            #making sure the list is not empty
    if len(myPersonListArea) != 0:
        #getting the largest boxed area of a person which means he is the closest to the drone
        #and we will track the closest person for now
        i = myPersonListArea.index(max(myPersonListArea))
        #We will only return the person we want to track 
        return img, [myPersonList[i], myPersonListArea[i]]
    else:
        return img, [[0,0],0]

def trackPerson(drone1, info, w, pid , pError, target_area=7000, pid_area=[0.5, 0.5, 0]):

    #PID for rotation
    #if width is 640 then our middle point is 320. Our position should always be 320
    #if above 320 we have a posotive error, if below 320 we have a negative error
    #to see how far away are we from the center of the frame
    #This is like cx - center
    error = info[0][0] - w//2
    speed = pid[0]*error + pid[1]*(error-pError)
    #making the speed stay between -100 and 100. So it does not exceed the limit of the drone
    speed = int(np.clip(speed, -60, 60))

    #PID Implementation for going backwards and forward based on area
    area_error = target_area - info[1]
    speed_fb = pid_area[0] * area_error + pid_area[1] * (area_error - pError)
    speed_fb = int(np.clip(speed_fb, -30, 30))  # Limit forward/backward speed

    # PID for up/down
    #up_down_error = target_area - info[1]
    #speed_ud = pid[0] * up_down_error + pid[1] * (up_down_error - pError)
    #speed_ud = int(np.clip(speed_ud, -20, 20))  # Limit up/down speed

    print(f"Yaw Speed: {speed}, Forward/Backward Speed: {speed_fb}, Area {info[1]}")

    if info[0][0] !=0:
        drone1.yaw_velocity = speed
        drone1.for_back_velocity = speed_fb
        #drone1.up_down_velocity = speed_ud
    else:
        drone1.for_back_velocity = 0
        drone1.left_right_velocity = 0
        drone1.up_down_velocity = 0
        drone1.yaw_velocity = 0
        error = 0
    
    if drone1.send_rc_control:
        drone1.send_rc_control(drone1.left_right_velocity,
                               drone1.for_back_velocity,
                               drone1.up_down_velocity,
                               drone1.yaw_velocity)
    #this will be the previous error next time we do the calculation
    return error 