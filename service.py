#!/usr/bin/env python

from flask import Flask, render_template, Response
import numpy as np
import time
import cv2

app = Flask(__name__)

camera = cv2.VideoCapture(0)


def detection():
    # Load Yolo
    net = cv2.dnn.readNet("cfg/yolov3-tiny.weights", "cfg/yolov3-tiny.cfg")
    classes = []
    with open("cfg/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [
        layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()
    ]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Loading camera
    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0

    while True:
        success, frame = camera.read()
        frame_id += 1
        height, width, channels = frame.shape

        # Detecting objects
        blob = cv2.dnn.blobFromImage(frame,
                                     0.00392, (416, 416), (0, 0, 0),
                                     True,
                                     crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.2:
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
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.3)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y), (x + w, y + 30), color, -1)
                cv2.putText(frame, label + " " + str(round(confidence, 2)),
                            (x, y + 30), font, 3, (255, 255, 255), 3)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (10, 50), font, 3,
                    (0, 0, 0), 3)
        # cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break

        if not success:
            break

        try:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, -1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            pass


@app.route("/")
def index():
    return render_template('source.html')


@app.route("/video_feed")
def video_feed():
    return Response(detection(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run()

camera.release()
cv2.destroyAllWindows()
