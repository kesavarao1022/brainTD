from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import io
import base64
import boto3
import os
from datetime import datetime

# Upload_DIR = 'uploads'  # Directory to save uploaded images
# if not os.path.exists(Upload_DIR):
#     os.makedirs(Upload_DIR)
# Result_folder = 'results'  # Directory to save result images
# if not os.path.exists(Result_folder):
#     os.makedirs(Result_folder)
app = Flask(__name__)
CORS(app)

# Initialize AWS S3 client
s3 = boto3.client( 's3',
    aws_access_key_id=os.getenv('key'),
    aws_secret_access_key=os.getenv('Secret'),
    region_name=os.getenv('region')
)
BRAIN_TUMOR_MODEL = YOLO('best.pt')  # Your custom trained model

@app.route('/')
def homepage():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/detect_brain_tumor', methods=['POST'])
def detect_brain_tumor():
    """Specific brain tumor detection endpoint"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read image file
        image_file = request.files['image']
        image_bytes = image_file.read()
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Run brain tumor detection
        results = BRAIN_TUMOR_MODEL(image)
        
        # Process results specifically for brain tumors
        annotated_image, tumor_detections = process_brain_tumor_results(image, results)
        
        # Convert annotated image to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Calculate overall assessment
        tumor_detected = len(tumor_detections) > 0
        max_confidence = max([det['confidence'] for det in tumor_detections]) if tumor_detections else 0

        return jsonify({
            'success': True,
            'tumor_detected': tumor_detected,
            'confidence': max_confidence,
            'detections': tumor_detections,
            'annotated_image': f"data:image/jpeg;base64,{img_base64}",
            'total_tumors': len(tumor_detections),
            'message': f"{'Tumor(s) detected' if tumor_detected else 'No tumors detected'}"
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_brain_tumor_results(image, results):
    """Process brain tumor detection results with medical-specific annotations"""
    tumor_detections = []
    annotated_image = image.copy()
    
    # Define colors for different tumor types (if your model detects different types)
    
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Get class name (tumor type)
                tumor_type = BRAIN_TUMOR_MODEL.names[class_id]
                
                # Only process if confidence is above threshold
                if confidence > 0.3:  # Lower threshold for medical detection
                    # Calculate tumor area (approximate)
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    
                    # Add to detections list
                    tumor_detections.append({
                        'tumor_type': tumor_type,
                        'confidence': float(confidence),
                        'bbox': {
                            'x1': int(x1), 'y1': int(y1),
                            'x2': int(x2), 'y2': int(y2)
                        },
                        'dimensions': {
                            'width': int(width),
                            'height': int(height),
                            'area': int(area)
                        },
                        'center': {
                            'x': int((x1 + x2) / 2),
                            'y': int((y1 + y2) / 2)
                        }
                    })
                    
                    
                    # Draw bounding box with thicker line for tumors
                    cv2.rectangle(annotated_image, (int(x1), int(y1)), (int(x2), int(y2)), 3)
                    
                    
                    # Draw detailed label
                    label = f"{tumor_type}: {confidence:.2f}"
                    size_label = f"Size: {int(width)}x{int(height)}"
                    
                    # Label background
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    size_label_size = cv2.getTextSize(size_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                    
                    # Draw label backgrounds
                    cv2.rectangle(annotated_image, (int(x1), int(y1) - 45),
                                (int(x1) + max(label_size[0], size_label_size[0]) + 10, int(y1)), color, -1)
                    
                    # Draw labels
                    cv2.putText(annotated_image, label, (int(x1) + 5, int(y1) - 25),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(annotated_image, size_label, (int(x1) + 5, int(y1) - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return annotated_image, tumor_detections

if __name__ == '__main__':
    app.run(debug=True)