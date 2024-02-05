import cv2
import numpy as np 
yolo = cv2.dnn.readNet(r"V3\CONFI FILES\yolov3.weights",r"V3\CONFI FILES\yolov3.cfg")
classes = []
with open(r"V3\CONFI FILES\coco.names") as f:
    classes = f.read().splitlines()

img = cv2.imread(r"Sample\test3.jpg")
height, width, _ = img.shape
blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320), (0,0,0), swapRB=True, crop = False)
yolo.setInput(blob)
output_layes_name = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layes_name)

boxs = []
confidences = []
class_ids = []
for output in layeroutput :
    for detection in output:
        score=detection [5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence>0.6 :
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
colors = np.random.uniform(0, 255, size= (len(boxs), 3))

for i in indexes.flatten():
    x,y,w,h = boxs[i]
    label = str(classes[class_ids[i]])
    confi = str(round(confidences[i],2))
    color = colors[i]
    cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
    cv2.putText(img, label +" "+ confi, (x,y+20), font, 2, (255,255,255),2)
cv2.imshow("Image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()