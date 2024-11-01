
import logging
import os
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from io import BytesIO
import requests
import gdown
import tensorflow as tf
import cv2
from tensorflow.lite.python.interpreter import Interpreter

# Load environment variables from .env file
load_dotenv()

# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)
CORS(app)  # Allow CORS for all origins

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Define the file name and output path
file_id = '1lu06HZsDf6wYdcyXjUu8C4xNAvmWZ5pM'
output_file = "model_quantized.tflite"  # Using a TFLite model

# Check if the model file already exists
if not os.path.isfile(output_file):
    # Construct the download URL
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    # Download the model file
    gdown.download(download_url, output_file, quiet=False)
else:
    logging.info(f"{output_file} already exists. No download needed.")

# Load the TFLite model using the TFLite Interpreter
interpreter = Interpreter(model_path=output_file)
interpreter.allocate_tensors()

# Get input and output details for the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def pencil_sketch_effect(img):
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)  # Convert from float (0-1) to uint8 (0-255)

    # Ensure the input image has 3 channels (BGR) before converting to grayscale
    if len(img.shape) == 2:  # Single channel image, already grayscale
        gray_img = img
    elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image (3 channels)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("Unexpected image format! Ensure the image has 1 or 3 channels.")

    # Invert the grayscale image
    inverted_img = cv2.bitwise_not(gray_img)
    
    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)
    
    # Invert the blurred image
    inverted_blur = cv2.bitwise_not(blurred_img)
    
    # Create the pencil sketch effect by blending
    sketch_img = cv2.divide(gray_img, inverted_blur, scale=256.0)

    return sketch_img

def make_prediction(image):
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Retrieve the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

@app.route("/", methods=["GET"])
def root():
    return jsonify({"message": "Hello World"})

@app.route("/predict", methods=["POST"])
def predict():
    max_upload_size = 16 * 1024 * 1024  # 16 MB
    content_length = request.headers.get("Content-Length")

    if content_length and int(content_length) > max_upload_size:
        return jsonify({"detail": "Payload Too Large"}), 413

    try:
        logging.debug("Received request for prediction.")
        
        data = request.get_json()
        logging.debug(f"Received data: {data}")

        if 'image' not in data:
            return jsonify({"detail": "No image data provided"}), 400
        
        image_data = data['image']['_streams'][1]
        logging.debug(f"Image data URL: {image_data}")

        # Check if the URL is an image file
        if not any(image_data.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            return jsonify({"detail": "Invalid image URL provided"}), 400
        
        # Extract file name and format
        p = image_data.split('/')[-1]
        a = p.split('.')[-1].upper()
        if a == 'JPG':
            a = 'JPEG'
            p = p.replace('JPG', 'JPEG')

        response = requests.get(image_data)
        logging.debug(f"Fetched image with status code: {response.status_code} and size: {len(response.content)}")

        # Check if the response is an image
        if response.status_code != 200 or 'image' not in response.headers['Content-Type']:
            return jsonify({"detail": "Failed to retrieve a valid image"}), 400
        
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = img.resize((256, 256))
        image = np.array(img) / 255.0

        if(p.split('-')[-1]=="canvasimg.jpg"):
            logging.debug("Pencil sketch effect")
            image = pencil_sketch_effect(image)
            if len(image.shape) == 3 and image.shape[2] == 1:  # Single channel grayscale
                image = np.repeat(image, 3, axis=2)
            elif len(image.shape) == 2:  # Grayscale without channel dimension
                image = np.stack((image,) * 3, axis=-1)  # Duplicate across 3 channels

            # Normalize and reshape to (1, 256, 256, 3) as expected by the model
            image = image.astype(np.float32) / 255.0 
        else:
            logging.debug(f"No pencil sketch effect {p}")
        
        # Prepare the input for TFLite model
        # image = np.expand_dims(image, axis=0).astype(np.float32)  # Shape it for model (1, 256, 256, 3)
        
        # # Make a prediction using the TFLite Interpreter
        # prediction = make_prediction(image)
        # logging.debug(f"Prediction shape: {prediction.shape}")
        image = np.expand_dims(image, axis=0).astype(np.float32)  # Shape it for model (1, 256, 256, 3)
        logging.debug(f"Prepared input shape for prediction: {image.shape}")

        # Attempt to make a prediction using the TFLite Interpreter
        try:
            prediction = make_prediction(image)
            logging.debug(f"Prediction shape: {prediction.shape}")
        except Exception as e:
            logging.error(f"Error during model prediction: {str(e)}")
            return jsonify({"detail": "Error during model prediction"}), 500
        # Convert the prediction back to an image
        predicted_image = (prediction[0] * 255).astype(np.uint8)
        output_image = Image.fromarray(predicted_image)

        # Save the predicted image in memory (without writing to disk)
        image_io = BytesIO()
        output_image.save(image_io, format=a)
        image_io.seek(0)

        files = {
            'images': (f'{p}', image_io, f'image/{a.lower()}'),
            'name': (None, data['user']),
            'filename': (None, f'{p}'),
        }
        
        logging.debug(f"Sending image to Node.js server for user: {data['user']}")
        response = requests.post(f'https://backend-9oaz.onrender.com/vendor/sktvendor/{data["user"]}', files=files)

        if response.status_code != 200:
            logging.error(f"Failed to upload image to Node.js server, status code: {response.status_code}")
            return jsonify({"detail": "Failed to upload image to Node.js server"}), 500

        logging.debug('Image successfully processed and sent to Node.js server')
        return jsonify({"message": "Image processed successfully", "node_response": response.json()})
    
    except Exception as e:
        logging.error(f"Error processing the image: {str(e)}")
        logging.debug(traceback.format_exc())
        return jsonify({"detail": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))  # Use Render's PORT or default to 5000
    app.run(host="0.0.0.0", port=port)

