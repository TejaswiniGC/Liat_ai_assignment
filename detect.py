from ultralytics import YOLO
import cv2

model = YOLO('model/best.pt')  

video_path = 'video/15sec_input_720p.mp4'
cap = cv2.VideoCapture(video_path)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter('output_detected.mp4',
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.3, save=False, verbose=False)

    annotated_frame = results[0].plot() 

    out.write(annotated_frame)
    cv2.imshow('Detection', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
print("Detection video saved as 'output_detected.mp4'")
