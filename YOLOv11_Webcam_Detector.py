#Import YOLOv11 package from ultralytics and OpenCV
from ultralytics import YOLO
import cv2

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize YOLOv11 object detector
model_path = "yolo11n.pt"
yolov11_detector = YOLO(model_path)
class_names = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush', 'Goggle', 'sunglass', 'pen'
]

cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
while cap.isOpened():
    # Capture Frames from webcam
    ret, frame = cap.read()

    if not ret:
        break

    # Make predictions
    results = yolov11_detector(frame, conf=0.6, iou=0.5)
    
    # Process results
    for result in results:
        boxes = result.boxes.xyxy
        scores = result.boxes.conf
        class_ids = result.boxes.cls

        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = map(int, box)
            
            # Detect Object Names
            class_name = class_names[int(class_id)]
            
            # Percentage of Detected Objects 
            percentage = int(score * 100)
            
            # Object name and percentage
            label = f"{class_name} {percentage}%"
            
            # Object borderbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
            
            # Label for Object name
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detected Objects", frame)

    # Press 'Q button' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()