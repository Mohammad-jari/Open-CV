import cv2
import numpy as np

# Load YOLO model
yolo = cv2.dnn.readNet(r"YOLO V3\CONFI FILES\yolov3.weights", r"YOLO V3\CONFI FILES\yolov3.cfg")
classes = []

with open(r"YOLO V3\CONFI FILES\coco.names") as f:
    classes = f.read().splitlines()

# Open a video capture object (0 for default camera, or provide a video file path)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    frame = cv2.resize(frame, (416, 416), interpolation=cv2.INTER_AREA)

    
    # If the frame was not read, break from the loop (end of video stream)
    if not ret:
        break

    # Resize the frame
    height, width, _ = frame.shape
    frame = cv2.resize(frame, (500, 700), interpolation=cv2.INTER_AREA)

    # Preprocess the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 1/255, (320, 320), (0, 0, 0), swapRB=True, crop=False)
    yolo.setInput(blob)
    output_layers_name = yolo.getUnconnectedOutLayersNames()
    layeroutput = yolo.forward(output_layers_name)

    # Process YOLO output and draw bounding boxes
    boxs = []
    confidences = []
    class_ids = []

    for output in layeroutput:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxs.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxs, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxs), 3))

    for i in indexes:
        x, y, w, h = boxs[i]
        label = str(classes[class_ids[i]])
        confi = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label + " " + confi, (x, y + 20), font, 2, (255, 255, 255), 2)

    # Display the resulting frame
    cv2.imshow("Live Video", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
