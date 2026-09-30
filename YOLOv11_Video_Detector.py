#Import YOLOv11 package from ultralytics and OpenCV
from ultralytics import YOLO
import cv2

def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results

def predict_and_detect(chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),(int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), rectangle_thickness)
            
            # Percentage of Detected Objects 
            confidence = int(box.conf[0] * 100)  
            label = f"{result.names[int(box.cls[0])]} {confidence}%"
            cv2.putText(img, label,(int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), text_thickness)
    return img, results

def create_video_writer(video_cap, output_filename):
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 90
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    return writer

model = YOLO("yolo11x.pt")

output_filename = "Result_Video/result.mp4"
video_path = r"Videos/speed.mp4"
cap = cv2.VideoCapture(video_path)

# Check whether video is present or not
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

writer = create_video_writer(cap, output_filename)

while True:
    success, img = cap.read()
    if not success:
        break
    
    result_img, _ = predict_and_detect(model, img, classes=[], conf=0.5)
    writer.write(result_img)
    cv2.imshow("Result", result_img)

    # Press 'Q button' to stop
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
writer.release()
cv2.destroyAllWindows()
