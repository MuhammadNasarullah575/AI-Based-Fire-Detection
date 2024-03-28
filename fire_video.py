from ultralytics import YOLO
import cvzone
import cv2
import math

# Running real-time from webcam
cap = cv2.VideoCapture('fire2.mp4')
model = YOLO('best.pt')

# Reading the classes
classnames = ['Smoke', 'Spark', 'fire', 'flame']

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640, 480))
    result = model(frame, stream=True)

    # Getting bbox, confidence, and class names information to work with
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            Class = int(box.cls[0])
            if confidence > 50:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Set color based on class
                if classnames[Class] == 'Smoke':
                    color = (0, 255, 255)  # Yellow for smoke
                elif classnames[Class] == 'Spark':
                    color = (0, 165, 255)  # Orange for spark
                elif classnames[Class] == 'fire':
                    color = (255, 0, 0)  # Blue for fire
                elif classnames[Class] == 'flame':
                    color = (0, 0, 255)  # Red for flame

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 5)
                cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],
                                   scale=1.5, thickness=2)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
