import cv2
import numpy as np
from ultralytics import YOLO
from yolox.tracker.byte_tracker import BYTETracker
from yolox.tracker.track import STrack

tracker_args = {
    "track_thresh": 0.5,
    "match_thresh": 0.8,
    "track_buffer": 30,
    "frame_rate": 30,
    "mot20": False,
}

model = YOLO("model/best.pt")
tracker = BYTETracker(tracker_args, frame_rate=30)

cap = cv2.VideoCapture("video/15sec_input_720p.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter("output_bytetrack.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

class_names = model.model.names  

frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1

    results = model.predict(frame, conf=0.4, iou=0.6, verbose=False)[0]

    if results.boxes is None:
        out.write(frame)
        continue

    detections = []

    for box in results.boxes:
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = xyxy
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        w = x2 - x1
        h = y2 - y1
        detections.append([x1, y1, w, h, conf, cls])

    detections = np.array(detections)
    online_targets = tracker.update(detections, [height, width], (height, width))

    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        cls_id = int(t.score[1]) if len(t.score) > 1 else 0  
        x1, y1, w, h = tlwh
        x2, y2 = int(x1 + w), int(y1 + h)
        cv2.rectangle(frame, (int(x1), int(y1)), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[cls_id]} #{tid}"
        cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("ByteTrack", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print("Tracking complete. Output saved as 'output_bytetrack.mp4'")
