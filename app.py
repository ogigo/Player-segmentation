from flask import Flask, render_template, request
import base64
import cv2
import numpy as np
from model import predict

app = Flask(__name__)

def convert_to_black_and_white(image_data):
    
    segmented_image=predict(image_data)

    # Convert the processed image back to base64 encoding
    _, encoded_image = cv2.imencode('.png', segmented_image)
    return base64.b64encode(encoded_image).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' in request.files:
        file = request.files['image']
        original_image = file.filename
        file.save(original_image)
        print(original_image)
        processed_image = convert_to_black_and_white(original_image)
        return render_template('index.html', original_image=original_image, processed_image=processed_image)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)