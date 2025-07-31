from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image
import io
import numpy as np
import cv2
import base64

app = Flask(__name__)

# Load YOLO model once
MODEL = YOLO('best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_brain_tumor', methods=['POST'])
def detect_brain_tumor():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert('RGB')
    np_image = np.array(image)

    # Run detection
    results = MODEL(image)
    if not results or not results[0].boxes:
        return jsonify({
            'tumor_detected': False,
            'message': 'No brain tumor detected.',
            'total_tumors': 0,
            'confidence': 0,
            'detections': [],
            'annotated_image': None
        })

    result = results[0]
    boxes = result.boxes
    detections = []
    max_confidence = 0

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence = float(box.conf)
        if confidence <0.3:
            continue
        tumor_type = str(result.names[int(box.cls)]) if hasattr(result, 'names') else 'Tumor'

        width = x2 - x1
        height = y2 - y1
        area = width * height
        center_x = x1 + width // 2
        center_y = y1 + height // 2

        detections.append({
            'tumor_type': tumor_type,
            'confidence': confidence,
            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
            'dimensions': {'width': width, 'height': height, 'area': area},
            'center': {'x': center_x, 'y': center_y}
        })
        

        if confidence > max_confidence:
            max_confidence = confidence

        # Draw rectangle and label
        cv2.rectangle(np_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f'{tumor_type}: {confidence:.2f}'
        cv2.putText(np_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Convert annotated image to base64
    _, buffer = cv2.imencode('.jpg', np_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')
    annotated_image_uri = f'data:image/jpeg;base64,{encoded_image}'

    return jsonify({
        'tumor_detected': True,
        'message': f'Brain tumor detected: {len(detections)} region(s) found.',
        'total_tumors': len(detections),
        'confidence': max_confidence,
        'detections': detections,
        'annotated_image': annotated_image_uri
    })

if __name__ == '__main__':
    app.run(debug=True, use_reloader = False)
