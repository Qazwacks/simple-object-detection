import cv2

#OpenCV DNN
net = cv2.dnn.readNet("dnn_model-220107-114215/dnn_model/yolov4-tiny.weights", "dnn_model-220107-114215/dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320, 320), scale=1/255)

#Load class list
classes = []
with open("dnn_model-220107-114215/dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

#Initialize Camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    #Get frames
    ret, frame = cap.read()

    #Object detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        x, y, w, h = bbox
        #Draw the rectange (where, point1, point2, color, thickness)
        cv2.rectangle(frame, (x,y), (x + w, y + h), (200, 0, 50), 3)
        #tell us what object (where, what type, position, font, size, color, thickness )
        class_name = classes[class_id]
        cv2.putText(frame, str(class_name), (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (200, 0, 50), 2)

    #Checking
    print("Class ids", class_ids)
    print("scores", scores)
    print("bboxes", bboxes)

    #load frames
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)