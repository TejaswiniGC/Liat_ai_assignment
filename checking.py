import cv2

cap = cv2.VideoCapture("video/15sec_input_720p.mp4")
if not cap.isOpened():
    print("OpenCV failed to open the video.")
    exit()

print("OpenCV opened the video.")

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count == 1:
        cv2.imwrite("first_frame.jpg", frame) 
        print("Saved first frame as 'first_frame.jpg'")
        break

cap.release()
