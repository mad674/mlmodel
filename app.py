import cv2
import numpy as np
from PIL import Image
import io
import gc

def is_background_white(image_np, threshold=0.9):
    # Calculate the percentage of pixels that are near white
    white_pixels = np.all(image_np >= [240, 240, 240], axis=2)  # Adjust sensitivity if needed
    white_ratio = np.sum(white_pixels) / white_pixels.size
    return white_ratio > threshold

def convert_background_to_white(image_data):
    # Read the image from bytes and convert to numpy array
    img = Image.open(io.BytesIO(image_data)).convert('RGB')
    img_np = np.array(img)
    
    # Check if the background is already white
    if is_background_white(img_np):
        # Return original image if background is predominantly white
        return image_data
    
    # Convert to HSV for better color isolation
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # Define a mask to select the background color range
    lower_background = np.array([0, 0, 0])      # Lower bound for dark shades
    upper_background = np.array([180, 255, 100]) # Upper bound for light shades

    # Create a mask where non-white pixels are set to white
    mask = cv2.inRange(hsv, lower_background, upper_background)
    img_np[mask == 0] = [255, 255, 255] # Replace non-white background with white
    
    # Convert back to PIL format and then to bytes for further processing
    white_bg_img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    white_bg_img.save(buffer, format="PNG")
    logging.debug("Background converted to white")
    return buffer.getvalue()
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
import logging
import os
import traceback
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import gdown
import tensorflow as tf
import cv2
# Load environment variables from .env file
load_dotenv()

# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = FastAPI()

# Allow CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.DEBUG)

# Define the file name and output path
file_id = "1KL_E5JdasiXJkMuACFT5U3DHCB5zs1fM"
output_file = "p2p.tflite"  
# Check if the model file already exists
if not os.path.isfile(output_file):
    # Construct the download URL
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    # Download the model file
    gdown.download(download_url, output_file, quiet=False)
else:
    logging.info(f"{output_file} already exists. No download needed.")

# # Convert the model to TensorFlow Lite format with quantization
# tflite_model_file = 'model_quantized.tflite'
# if not os.path.isfile(tflite_model_file):
#     model = tf.keras.models.load_model(output_file)
    
#     # Set up the TFLite converter with quantization
#     converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
#     # Enable full integer quantization
#     converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
#     # Optionally, set a representative dataset for better accuracy during quantization
#     def representative_data_gen():
#         for _ in range(100):
#             # Yield random data as an example (adjust to your real input)
#             yield [np.random.rand(1, 256, 256, 3).astype(np.float32)]
    
#     converter.representative_dataset = representative_data_gen
#     converter.target_spec.supported_types = [tf.float16]  # Use float16 precision
    
#     # Convert the model
#     tflite_model = converter.convert()
    
#     # Save the quantized model
#     with open(tflite_model_file, 'wb') as f:
#         f.write(tflite_model)
# else:
#     logging.info(f"{tflite_model_file} already exists. No conversion needed.")

# Load TensorFlow Lite model into an interpreter
interpreter = tf.lite.Interpreter(model_path=output_file)
# interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()

# Get input and output details for the mode
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class ImageRequest(BaseModel):
    image: dict
    user: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(request: Request):
    # Set maximum upload size to 16 MB
    max_upload_size = 16 * 1024 * 1024  # 16 MB
    content_length = request.headers.get("content-length")

    if content_length and int(content_length) > max_upload_size:
        raise HTTPException(status_code=413, detail="Payload Too Large")

    try:
        logging.debug("Received request for prediction.")
        
        # Parse JSON body
        data = await request.json()
        
        # Check for image data
        if 'image' not in data:
            raise HTTPException(status_code=400, detail="No image data provided")
        
        # Extract image URL from the JSON structure
        image_data = data['image']['_streams'][1]  # Adjust according to your structure
        a = image_data.split('uploads/')[1]
        p = a
        a = a[len(a)-(len(a)-a.find('.'))+1:].upper()
        if a == 'JPG':
            a = 'JPEG'
            p = p.replace('JPG', 'JPEG')
        
        # Fetch the image from the URL
        response = requests.get(image_data)
        res=convert_background_to_white(response.content)
        img = Image.open(BytesIO(res)).convert('RGB')  # Ensure it's in RGB format
        
        # Preprocess the image for the model
        img = img.resize((1024,1024))  # Resize image for the model
        image = (np.array(img) / 127.5)-1  # Normalize the image
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
        
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))
        logging.debug("1.settensor")
        # Run the inference
        interpreter.invoke()
        logging.debug("2.invoke inter")
        # Get the prediction result
        prediction = interpreter.get_tensor(output_details[0]['index'])
        logging.debug("3.predict")
        # Convert the prediction back to an image
        predicted_image = np.clip((prediction[0] + 1) * 127.5, 0, 255).astype(np.uint8)
        output_image = Image.fromarray(predicted_image)

        # Save the predicted image in memory (without writing to disk)
        image_io = BytesIO()
        output_image.save(image_io, format=a)  # Save as JPEG or other format
        image_io.seek(0)  # Move to the beginning of the BytesIO buffer

        # Send the image to the Node.js server
        files = {
            'images': (f'{p}', image_io, f'image/{a.lower()}'),
            'name': (None, data['user']),
            'filename': (None, f'{p}'),
        }
        
        logging.debug(f"Sending image to Node.js server for user: {data['user']}")
        response = requests.post(f'https://backend-9oaz.onrender.com/vendor/sktvendor/{data["user"]}', files=files)

        # Check if the response from Node.js server is successful
        if response.status_code != 200:
            logging.error(f"Failed to upload image to Node.js server, status code: {response.status_code}")
            raise HTTPException(status_code=500, detail="Failed to upload image to Node.js server")

        logging.debug('Image successfully processed and sent to Node.js server')
        del prediction, predicted_image, image,output_image,image_data,data  # if applicable
        gc.collect()
        return {"message": "Image processed successfully", "node_response": response.json()}
    
    except Exception as e:
        logging.error(f"Error processing the image: {str(e)}")
        logging.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))  # Use Render's PORT or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
