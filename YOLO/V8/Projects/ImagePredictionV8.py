from ultralytics import YOLO
import cv2
import numpy as np

# Load YOLOv8 model
model = YOLO("V8\CONFI FILES\yolov8m.pt")

# Load image for detection
image = cv2.imread(r"Sample\test2.jpeg")

# Perform object detection
results = model.predict(image)

# Process detection results
for box in results[0].boxes:
    class_id = results[0].names[box.cls[0].item()]
    cords = [round(x) for x in box.xyxy[0].tolist()]
    conf = round(box.conf[0].item(), 2)
    
    # Filter out detections based on confidence level
    if conf > 0.5:
        # Draw bounding box on the image
        color = (0, 255, 0)  # Green color for bounding box
        cv2.rectangle(image, (cords[0], cords[1]), (cords[2], cords[3]), color, 2)
        
        # Display class label and confidence
        label = f"{class_id} {conf}"
        cv2.putText(image, label, (cords[0], cords[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image with detected objects
cv2.imshow("Object Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
