import os
import cv2
import numpy as np
from PIL import Image


#directory='YOLO/yolo_frames'
#file_name=0

net = cv2.dnn.readNet("YOLO/yolov3.weights", "YOLO/yolov3.cfg")
classes = []
with open("YOLO/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


cap=cv2.VideoCapture('walking.avi')



while True:
    for i in range(5):
        _,frame= cap.read()
    frame = cv2.resize(frame, (500,500))
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN


    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            
            if label=='person':
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0,0), 2)
                cv2.putText(frame, label, (x, y), font, 1, (0,0,255), 2)

    """  The following commented code is to save the frames of the object detected video """
    
    # im = Image.fromarray(frame, 'RGB')
    # im.save(os.path.join(directory, str(file_name)+".jpg"), "JPEG")
    # file_name = file_name + 1        
            

    cv2.imshow("Capturing", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()