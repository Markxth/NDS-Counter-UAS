import cv2 as cv
from ultralytics import YOLO
import time

def aim(model_path, camera_index=0, takeoff_alt=0.2):
    model = YOLO(model_path)
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        raise IOError("Camera not accessible")

    width  = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))


    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)

        #current_time = time.time()
        #prev_time = current_time
        #dt = current_time - prev_time

        if len(results[0].boxes) > 0:
            boxes     = results[0].boxes
            #quick human detectio only
        
            mask = results[0].boxes.cls == 0 #mask 
            humans = results[0].boxes[mask] #get only humans
            for human in humans :
                x1, y1, x2, y2 = human.xyxy[0]
                cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 1)
                confidence = human.conf[0].item() #item() convers from tensor to float
                cv.putText(frame, f'human : confidence : {confidence:.2f}', (int(x1),int(y1)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        else:
            pass

        cv.imshow("tracking", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

aim("yolo26n.pt")