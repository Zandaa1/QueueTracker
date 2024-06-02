import cv2
from ultralytics import YOLO

# YOLOv8 MODEL USED
model = YOLO("yolov8n.pt")

# What Camera to use
cap = cv2.VideoCapture(0)

#While Camera is open
while cap.isOpened():

# Read Camera
    ret, frame = cap.read()

# frame was not read break the loop
    
    if not ret:
        break

# RESULTS ARE FROM WHAT YOLOV8/ULTRALYTICS ARE SUPPOSED TO TRACK
    results = model.track(source=frame, classes=[0], tracker="botsort.yaml")

    # Extract the number of people detected
    num_people = len([det for det in results[0].boxes if det.cls == 0])

    # CONSOLE OUTPUT HOW MANY PEOPLE ARE DETECTED
    print(f"Number of people detected: {num_people}")

    # ADD FRAME TO DETECTED OBJECTS
    annotated_frame = results[0].plot()  

    # OVERLAY TEXT ON THE FRAME, HOW MANY PEOPLE?
    cv2.putText(annotated_frame, 
                f'Number of people detected: {num_people}', 
                (10, 30),  # Position (x, y)
                cv2.FONT_HERSHEY_SIMPLEX,  # Font
                1,  # Font scale
                (0, 255, 0),  # Color (BGR - Green)
                2,  # Thickness
                cv2.LINE_AA)  # Line type

    # DISPLAY THESE THINGS PROGRAM NAME/FRAME
    cv2.imshow("PROTOTYPE", annotated_frame)

    # CLOSE PROGRAM WITH Q KEY
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
