import cv2 as cv
from ultralytics import YOLO
import time 
from dronekit import connect, VehicleMode 
from pymavlink import mavutil 

#load and capture

def model(model_path, video_path) :

    #initialsied variables which get tweaked during testing
    tolerance_x, tolerance_y =  10, 10 #dead zones

prev_time = time.time()

def PID(kp, ki, kd, integral, error, prev_error, dt) : 
    proportional =kp * error #=INSTANT error
    integral += error * dt # = accumulated error
    derivative = (error - prev_error) / dt if dt> 0 else 0 #no errors on the first frame as we filter dt ; deriv = rate of change 
    output = proportional + (ki * integral) + (kd * derivative)
    return output, integral

#velocity sent to drone

def velocity(vehicle, vx,vy,vz) : 
    vehicle.simple_takeoff(10)
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0, #arg 1 : time_boot_ms ; 0= autopilot
        1, #target system : Target ID , 0 or 1 usually. For 0 , all vehicles which can connect to mav link, will. For 1 it maps the vehicle(drone in our case). Broadcast vs 1 drone
        1, #target component for arg 3 : 1 for autopilot, the flight controller. 154 for gimbal. 191 for raspbery pi
        coordinate_frame = mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED #look through the documentaiton. This one is the relative frame, taking context into account

    )

'''In frames, _OFFSET_ means “relative to vehicle position” while _LOCAL_ is “relative to home position” (these have no impact on velocity directions).
 _BODY_ means that velocity components are relative to the heading of the vehicle rather than the NED frame.'''

def aim(model_path, camera_index = 0, takeoff_alt = 0.2) : #add takeoff altitude maybe(?)
    model = YOLO("yolov8n.pt") #to do later : switch to model_path for generalisability  
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened() :
        raise IOError("Camera not accessible")
    
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    #now i gotta track the state 
    
    integral_x , integral_y = 0 , 0  #integrals remember the accumulated error over time
    prev_error_x, prev_error_y = 0, 0 
    
    frame_cx = width / 2 
    frame_cy = height/2 
    integral_X , integral_y = 0,0
    kp_x, ki_x, kd_x = 0.001, 0.002, 0.003
    kp_y, ki_y, kd_y = 0.001, 0.002, 0.003
    current_time = time.time() 
    dt = current_time - prev_time
    prev_time = current_time #update prev so the new prv is the current time, for the next loop
    if (len(results[0] > 0 )) : 

        box = results[0].boxes.xyxy[0]
        target_x = (box[0] + box[2]) / 2
        target_y = (box[1] + box[3]) / 2
        
        #error will be frame - target = expected
       
        error_x = frame_cx - target_x
        error_y = frame_cy - target_y

#now that everything is defined, gotta conect 
#next up : check for armabla, set guided mode, arm, takeoff 

    print(f'Connecting to vehicle on: {connection_string}') 
    vehicle = connect(connection_string, wait_ready=True)

    while not vehicle.is_armable : 
        print("Waiting for vehicle to be armable...")
        vehicle.armed = True  #checked armable

    #now arm by mentioning it is guided.
    print("Setting vehicle type...")
    vehicle.mode = VehicleMode("GUIDED")
    if vehicle.mode != "GUIDED" : 
        print("Waiting for guided mode...")
        time.sleep(1) #guided mode set

    #now guide it
    print("Arming...") #arm
    vehicle.armed = True        
    if vehicle.armed != True : 
        print("Waiting for arming..."0)
        time.sleep(1) 

    #start controlling

    print ("Taking off!")
    vehicle.simple_takeoff(takeoff_alt)

    while True:
        print (f'Altitude: ", {vehicle.location.global_relative_frame.alt}')
        #Break and return from function just below target altitude.
        if vehicle.location.global_relative_frame.alt>=takeoff_alt*0.95: 
            print ("Reached target altitude")
            break
        time.sleep(1)

    alt = vehicle.location.global_relative_frame.alt

    #computer vision boxes.
    while(True) : 
       ret, frame = cap.read()
       if not ret : #ret is a boolean which checks if it read correctly 
           break 
       results = model(frame , verbose = True ) #run yolo on it

    if (len(results[0].boxes) > 0 ): 

        boxes = results[0].boxes
        boxes_max = boxes.conf.argmax().item() # find the best box
        box = boxes.xyxy[boxes_max] #take the best box and use it ; the logic is, we take objects we are sure of. 
        #xyxy = top left for x and y and bottom right. Mathematically thats all u need to know the centre 

    # compute centre
        centre_x = (box[0] + box[2] ) /2 
        centre_y = (box[1] + box[3]) / 2

    error_x = centre_x - frame_cx
    error_y = centre_y - frame_cy 

    pid_x , integral_x = PID(kp,kd,ki,integral_x,error_x,prev_error_x, dt)
    pid_y , integral_y = PID(kp,kd,ki, integral_y, error_y, prev_error_y, dt)

    prev_error_x = error_x  
    prev_error_y = error_y 


        
    '''
    NOTES 
    use local roboflow dataset for local training 
    '''


def main() :

