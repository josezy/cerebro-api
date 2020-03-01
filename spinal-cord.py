import cv2
import time
import json
import tensorflow as tf

from flask import Flask

from cerebro import load_model, clean_boxes


app = Flask(__name__)

model = load_model()
graph = tf.get_default_graph()

CAM_URL = 'http://75.147.0.206/mjpg/video.mjpg'


@app.route('/')
def detect():
    cap = cv2.VideoCapture(CAM_URL)
    success, frame = cap.read()
    if not success:
        raise f'Couldnt read frame from camera {CAM_URL}'

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    print(f"[+] Performing detection for cam '{CAM_URL}'")
    detection_start = time.time()
    with graph.as_default():
        results = model.detect([rgb_frame], verbose=0)
    print(f"[!] Detection took {time.time() - detection_start} secs")

    car_boxes = clean_boxes(results[0], ['car', 'bus', 'truck'], 0.8)
    return json.dumps(car_boxes.tolist())


if __name__ == '__main__':
    app.run(host='localhost', port='5000')
