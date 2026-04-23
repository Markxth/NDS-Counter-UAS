import cv2 as cv
from ultralytics import YOLO
import time

def aim(model_path, camera_index=0, takeoff_alt=0.2):
    model = YOLO("yolov8n.pt")
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("Camera not accessible")

    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    frame_cx = width  / 2
    frame_cy = height / 2

    integral_x,   integral_y   = 0, 0
    prev_error_x, prev_error_y = 0, 0
    kp_x, ki_x, kd_x = 0.002, 0.0001, 0.001
    kp_y, ki_y, kd_y = 0.002, 0.0001, 0.001
    prev_time = time.time()

    # -- drone connection, arming, takeoff commented out for CV testing --
    # print(f'Connecting to vehicle on: {connection_string}')
    # vehicle = connect(connection_string, wait_ready=True)
    # while not vehicle.is_armable:
    #     print("Waiting for vehicle to be armable...")
    #     time.sleep(1)
    # vehicle.mode = VehicleMode("GUIDED")
    # while vehicle.mode.name != "GUIDED":
    #     print("Waiting for guided mode...")
    #     time.sleep(1)
    # vehicle.armed = True
    # while not vehicle.armed:
    #     print("Waiting for arming...")
    #     time.sleep(1)
    # vehicle.simple_takeoff(takeoff_alt)
    # while True:
    #     if vehicle.location.global_relative_frame.alt >= takeoff_alt * 0.95:
    #         print("Reached target altitude")
    #         break
    #     time.sleep(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        frame =results[0].plot()

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        if len(results[0].boxes) > 0:
            boxes     = results[0].boxes
            #quick human detectio only
            '''
            mask = results[0].boxes.cls == 0 #mask 
            humans = results[0].boxes[mask] #get only humans
            for human in humans :
                x1, y1, x2, y2 = human.xyxy[0]
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
                confidence = human.conf[0].item() #item() convers from tensor to float
                cv.putText(frame, f'human : confidence : {confidence:.2f}', (int(x1),int(y1)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
'''
            #-------------------------------------------------
            #boxes_max = boxes.conf.argmax().item()
            #box       = boxes.xyxy[boxes_max]
            #where boxes are all the objects detected, boxes.conf is the confidence of each box, and boxes.xyxy is the coordinates of each box. 
            # I want the box with the highest confidence, so i get the index of that box with argmax and the coords with xyxy
            
            #-------------------------------------------------

            #centre_x = (box[0] + box[2]) / 2
            #centre_y = (box[1] + box[3]) / 2

            #error_x = centre_x - frame_cx
            #error_y = centre_y - frame_cy

            # pid_x, integral_x = PID(kp_x, ki_x, kd_x, integral_x, error_x, prev_error_x, dt)
            # pid_y, integral_y = PID(kp_y, ki_y, kd_y, integral_y, error_y, prev_error_y, dt)
            # max_vel = 0.5
            # vx = max(-max_vel, min(max_vel, float(pid_x)))
            # vy = max(-max_vel, min(max_vel, float(pid_y)))
            # send_ned_velocity(vehicle, vy, vx, 0.0)

            #prev_error_x = error_x
            #prev_error_y = error_y

        else:
            pass
            # send_ned_velocity(vehicle, 0, 0, 0)

        cv.imshow("tracking", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

aim("yolov8n.pt")