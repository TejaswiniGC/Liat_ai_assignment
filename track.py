from ultralytics import YOLO
import cv2
import numpy as np
from norfair import Detection, Tracker, draw_tracked_objects

model = YOLO("model/best.pt")

tracker = Tracker(distance_function="euclidean", distance_threshold=30)

cap = cv2.VideoCapture("video/15sec_input_720p.mp4")
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("output_tracked.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4, iou=0.6, verbose=False)
    boxes = results[0].boxes

    detections = []
    if boxes is not None:
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            x_center = (xyxy[0] + xyxy[2]) / 2
            y_center = (xyxy[1] + xyxy[3]) / 2
            detections.append(Detection(points=np.array([[x_center, y_center]])))

    tracked_objects = tracker.update(detections=detections)

    draw_tracked_objects(frame, tracked_objects)

    annotated_frame = results[0].plot()

    out.write(annotated_frame)

    cv2.imshow("Tracked", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Tracking complete. Saved to 'output_tracked.mp4'")
