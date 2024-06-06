import cv2
from ultralytics import YOLO
import object_counter as solutions

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)
assert cap.isOpened(), "Error reading video file"
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
w = 1280
h = 720
fps = cap.get(cv2.CAP_PROP_FPS)

line_points = [(650, 0), (650, 720)]  # line or region points
classes_to_count = [0]  # person and car classes for count

# Video writer
##video_writer = cv2.VideoWriter("object_counting_output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Init Object Counter
counter = solutions.ObjectCounter(
    view_img=True,
    reg_pts=line_points,
    classes_names=model.names,
    draw_tracks=True,
    line_thickness=2,
)

while cap.isOpened():
    success, im0 = cap.read()
    ##if not success:
       ## print("Video frame is empty or video processing has been successfully completed.")
        ##break
    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count, tracker="bytetrack.yaml")

    im0 = counter.start_counting(im0, tracks)
    ##video_writer.write(im0)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
##video_writer.release()
cv2.destroyAllWindows()
