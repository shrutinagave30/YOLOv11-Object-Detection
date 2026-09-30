#Import YOLOv11 package from ultralytics and OpenCV
from ultralytics import YOLO
import cv2

# Upload images
filename = "Images/test1.png"
image = cv2.imread(filename)

# import & load a YOLO11x model
model = YOLO("yolo11x.pt")

results = model(image)
for result in results:
    boxes = result.boxes 
    result.show()
    result.save(filename="Result_img/rs1.jpg")  # Save your output to some image files
