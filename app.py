# import logging
# import os
# import traceback
# from fastapi import FastAPI, HTTPException, Request
# from pydantic import BaseModel
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import requests
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# import gdown
# import tensorflow as tf

# # Load environment variables from .env file
# load_dotenv()

# # Disable oneDNN optimizations
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# app = FastAPI()

# # Allow CORS for all origins
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Setup logging
# logging.basicConfig(level=logging.DEBUG)

# # Define the file name and output path
# # file_id = '1-Bsdmj9TK8DFM-HSwipckPW5a0RNNKVA'
# # output_file = 'generator_epoch_10.h5'

# # # Check if the model file already exists
# # if not os.path.isfile(output_file):
# #     # Construct the download URL
# #     download_url = f"https://drive.google.com/uc?id={file_id}"
    
# #     # Download the model file
# #     gdown.download(download_url, output_file, quiet=False)
# # else:
# #     logging.info(f"{output_file} already exists. No download needed.")

# # # Convert the model to TensorFlow Lite format with quantization
# # tflite_model_file = 'model_quantized.tflite'
# # if not os.path.isfile(tflite_model_file):
# #     model = tf.keras.models.load_model(output_file)
    
# #     # Set up the TFLite converter with quantization
# #     converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
# #     # Enable full integer quantization
# #     converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
# #     # Optionally, set a representative dataset for better accuracy during quantization
# #     def representative_data_gen():
# #         for _ in range(100):
# #             # Yield random data as an example (adjust to your real input)
# #             yield [np.random.rand(1, 256, 256, 3).astype(np.float32)]
    
# #     converter.representative_dataset = representative_data_gen
# #     converter.target_spec.supported_types = [tf.float16]  # Use float16 precision
    
# #     # Convert the model
# #     tflite_model = converter.convert()
    
# #     # Save the quantized model
# #     with open(tflite_model_file, 'wb') as f:
# #         f.write(tflite_model)
# # else:
# #     logging.info(f"{tflite_model_file} already exists. No conversion needed.")

# # Load TensorFlow Lite model into an interpreter
# interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
# interpreter.allocate_tensors()

# # Get input and output details for the mode
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# class ImageRequest(BaseModel):
#     image: dict
#     user: str

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.post("/predict")
# async def predict(request: Request):
#     # Set maximum upload size to 16 MB
#     max_upload_size = 16 * 1024 * 1024  # 16 MB
#     content_length = request.headers.get("content-length")

#     if content_length and int(content_length) > max_upload_size:
#         raise HTTPException(status_code=413, detail="Payload Too Large")

#     try:
#         logging.debug("Received request for prediction.")
        
#         # Parse JSON body
#         data = await request.json()
        
#         # Check for image data
#         if 'image' not in data:
#             raise HTTPException(status_code=400, detail="No image data provided")
        
#         # Extract image URL from the JSON structure
#         image_data = data['image']['_streams'][1]  # Adjust according to your structure
#         a = image_data.split('uploads/')[1]
#         p = a
#         a = a[len(a)-(len(a)-a.find('.'))+1:].upper()
#         if a == 'JPG':
#             a = 'JPEG'
#             p = p.replace('JPG', 'JPEG')
        
#         # Fetch the image from the URL
#         response = requests.get(image_data)
#         img = Image.open(BytesIO(response.content)).convert('RGB')  # Ensure it's in RGB format
        
#         # Preprocess the image for the model
#         img = img.resize((256, 256))  # Resize image for the model
#         image = np.array(img) / 255.0  # Normalize the image
#         image = np.expand_dims(image, axis=0)  # Add batch dimension
        
#         # Set the input tensor
#         interpreter.set_tensor(input_details[0]['index'], image.astype(np.float32))
        
#         # Run the inference
#         interpreter.invoke()

#         # Get the prediction result
#         prediction = interpreter.get_tensor(output_details[0]['index'])

#         # Convert the prediction back to an image
#         predicted_image = (prediction[0] * 255).astype(np.uint8)
#         output_image = Image.fromarray(predicted_image)

#         # Save the predicted image in memory (without writing to disk)
#         image_io = BytesIO()
#         output_image.save(image_io, format=a)  # Save as JPEG or other format
#         image_io.seek(0)  # Move to the beginning of the BytesIO buffer

#         # Send the image to the Node.js server
#         files = {
#             'images': (f'{p}', image_io, f'image/{a.lower()}'),
#             'name': (None, data['user']),
#             'filename': (None, f'{p}'),
#         }
        
#         logging.debug(f"Sending image to Node.js server for user: {data['user']}")
#         response = requests.post(f'http://localhost:4000/vendor/sktvendor/{data["user"]}', files=files)

#         # Check if the response from Node.js server is successful
#         if response.status_code != 200:
#             logging.error(f"Failed to upload image to Node.js server, status code: {response.status_code}")
#             raise HTTPException(status_code=500, detail="Failed to upload image to Node.js server")

#         logging.debug('Image successfully processed and sent to Node.js server')
#         return {"message": "Image processed successfully", "node_response": response.json()}
    
#     except Exception as e:
#         logging.error(f"Error processing the image: {str(e)}")
#         logging.debug(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(e))

# # Run the FastAPI app
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8000))  # Use Render's PORT or default to 8000
#     uvicorn.run(app, host="0.0.0.0", port=port)






# import logging
# import os
# import traceback
# from fastapi import FastAPI, HTTPException, Request
# from pydantic import BaseModel
# from tensorflow.keras.models import load_model
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import requests
# from fastapi.middleware.cors import CORSMiddleware
# from dotenv import load_dotenv
# import gdown

# # Load environment variables from .env file
# load_dotenv()

# # Disable oneDNN optimizations
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# app = FastAPI()

# # Allow CORS for all origins
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Setup logging
# logging.basicConfig(level=logging.DEBUG)

# # Define the file name and output path
# file_id = '1wUrE3eSfu0CSXCa1Xs_2SgvRQx59RZN9'
# output_file ='new70.h5'
# # Check if the model fle already exists
# if not os.path.isfile(output_file):
#     # Construct the download URL
#     download_url = f"https://drive.google.com/uc?id={file_id}"
    
#     # Download the model file
#     gdown.download(download_url, output_file, quiet=False)
# else:
#     logging.info(f"{output_file} already exists. No download needed.")

# # Load the mode
# model = load_model(output_file)

# class ImageRequest(BaseModel):
#     image: dict
#     user: str

# @app.get("/")
# async def root():
#     return {"message": "Hello World"}

# @app.post("/predict")
# async def predict(request: Request):
#     # Set maximum upload size to 16 MB
#     max_upload_size = 16 * 1024 * 1024  # 16 MB
#     content_length = request.headers.get("content-length")

#     if content_length and int(content_length) > max_upload_size:
#         raise HTTPException(status_code=413, detail="Payload Too Large")

#     try:
#         logging.debug("Received request for prediction.")
        
#         # Parse JSON body
#         data = await request.json()
        
#         # Check for image data
#         if 'image' not in data:
#             raise HTTPException(status_code=400, detail="No image data provided")
        
#         # Extract image URL from the JSON structure
#         image_data = data['image']['_streams'][1]  # Adjust according to your structure
#         a = image_data.split('uploads/')[1]
#         p = a
#         a = a[len(a)-(len(a)-a.find('.'))+1:].upper()
#         if a == 'JPG':
#             a = 'JPEG'
#             p = p.replace('JPG', 'JPEG')
        
#         # Fetch the image from the URL
#         response = requests.get(image_data)
#         img = Image.open(BytesIO(response.content)).convert('RGB')  # Ensure it's in RGB format
        
#         # Preprocess the image for the model
#         img = img.resize((256, 256))  # Resize image for the model
#         image = np.array(img) / 255.0  # Normalize the image
#         image = np.expand_dims(image, axis=0)  # Add batch dimension
        
#         # Make the prediction
#         prediction = model.predict(image)

#         # Convert the prediction back to an image
#         predicted_image = (prediction[0] * 255).astype(np.uint8)
#         output_image = Image.fromarray(predicted_image)

#         # Save the predicted image in memory (without writing to disk)
#         image_io = BytesIO()
#         output_image.save(image_io, format=a)  # Save as JPEG or other format
#         image_io.seek(0)  # Move to the beginning of the BytesIO buffer

#         # Send the image to the Node.js server
#         files = {
#             'images': (f'{p}', image_io, f'image/{a.lower()}'),
#             'name': (None, data['user']),
#             'filename': (None, f'{p}'),
#         }
        
#         logging.debug(f"Sending image to Node.js server for user: {data['user']}")
#         response = requests.post(f'http://localhost:4000/vendor/sktvendor/{data["user"]}', files=files)

#         # Check if the response from Node.js server is successful
#         if response.status_code != 200:
#             logging.error(f"Failed to upload image to Node.js server, status code: {response.status_code}")
#             raise HTTPException(status_code=500, detail="Failed to upload image to Node.js server")

#         logging.debug('Image successfully processed and sent to Node.js server')
#         return {"message": "Image processed successfully", "node_response": response.json()}
    
#     except Exception as e:
#         logging.error(f"Error processing the image: {str(e)}")
#         logging.debug(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(e))

# # Run the FastAPI app
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8000))  # Use Render's PORT or default to 8000
#     uvicorn.run(app, host="0.0.0.0", port=port)


# flask






# import logging
# import os
# import traceback
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from dotenv import load_dotenv
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import requests
# import gdown
# import tensorflow as tf
# import cv2
# # Load environment variables from .env file
# load_dotenv()

# # Disable oneDNN optimizations
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# app = Flask(__name__)
# CORS(app)  # Allow CORS for all origins

# # Setup logging

# logging.basicConfig(level=logging.DEBUG)

# # Define the file name and output path
# file_id = '1sT4PUj-ww3KzSRLLSeis7OmKpgWvBB9j'
# output_file = "a21g100.h5"  # Using a standard TensorFlow model

# # Check if the model file already exists
# if not os.path.isfile(output_file):
#     # Construct the download URL
#     download_url = f"https://drive.google.com/uc?id={file_id}"
    
#     # Download the model file
#     gdown.download(download_url, output_file, quiet=False)
# else:
#     logging.info(f"{output_file} already exists. No download needed.")

# # Load the standard TensorFlow model
# model = tf.keras.models.load_model(output_file)
# def pencil_sketch_effect(img):
#     if img.dtype != np.uint8:
#         img = (img * 255).astype(np.uint8)  # Convert from float (0-1) to uint8 (0-255)

#     # Ensure the input image has 3 channels (BGR) before converting to grayscale
#     if len(img.shape) == 2:  # Single channel image, already grayscale
#         gray_img = img
#     elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image (3 channels)
#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         raise ValueError("Unexpected image format! Ensure the image has 1 or 3 channels.")

#     # Invert the grayscale image
#     inverted_img = cv2.bitwise_not(gray_img)
    
#     # Apply Gaussian blur
#     blurred_img = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)
    
#     # Invert the blurred image
#     inverted_blur = cv2.bitwise_not(blurred_img)
    
#     # Create the pencil sketch effect by blending
#     sketch_img = cv2.divide(gray_img, inverted_blur, scale=256.0)

#     return sketch_img
# @app.route("/", methods=["GET"])
# def root():
#     return jsonify({"message": "Hello World"})

# @app.route("/predict", methods=["POST"])
# def predict():
#     max_upload_size = 16 * 1024 * 1024  # 16 MB
#     content_length = request.headers.get("Content-Length")

#     if content_length and int(content_length) > max_upload_size:
#         return jsonify({"detail": "Payload Too Large"}), 413

#     try:
#         logging.debug("Received request for prediction.")
        
#         data = request.get_json()
#         logging.debug(f"Received data: {data}")

#         if 'image' not in data:
#             return jsonify({"detail": "No image data provided"}), 400
        
#         image_data = data['image']['_streams'][1]
#         logging.debug(f"Image data URL: {image_data}")

#         # Extract file name and format
#         p = image_data.split('/')[-1]
#         a = p.split('.')[-1].upper()
#         if a == 'JPG':
#             a = 'JPEG'
#             p = p.replace('JPG', 'JPEG')

#         response = requests.get(image_data)
#         logging.debug(f"Fetched image with status code: {response.status_code} and size: {len(response.content)}")

#         img = Image.open(BytesIO(response.content)).convert('RGB')
#         img = img.resize((256, 256))
#         image = np.array(img) / 255.0
#         if(p.split('-')[-1]=="canvasimg.png"):
#             logging.debug("Pencil sketch effect mnnn")
#             image = pencil_sketch_effect(image)
#             if len(image.shape) == 3 and image.shape[2] == 1:  # Single channel grayscale
#                 image = np.repeat(image, 3, axis=2)
#             elif len(image.shape) == 2:  # Grayscale without channel dimension
#                 image = np.stack((image,) * 3, axis=-1)  # Duplicate across 3 channels

#             # Normalize and reshape to (1, 256, 256, 3) as expected by the model
#             image = image.astype(np.float32) / 255.0 
#         else:
#             logging.debug(f"No pencil sketch effect {p}")
#         # image = np.array(img) / 255.0
#         image = np.expand_dims(image, axis=0)
#         logging.debug(f"Input tensor shape: {image.shape}")
#         # Use the standard TensorFlow model to make a prediction
#         prediction = model.predict(image)
#         logging.debug(f"Prediction shape: {prediction.shape}")

#         # Convert the prediction back to an image
#         predicted_image = (prediction[0] * 255).astype(np.uint8)
#         output_image = Image.fromarray(predicted_image)

#         # Save the predicted image in memory (without writing to disk)
#         image_io = BytesIO()
#         output_image.save(image_io, format=a)
#         image_io.seek(0)

#         files = {
#             'images': (f'{p}', image_io, f'image/{a.lower()}'),
#             'name': (None, data['user']),
#             'filename': (None, f'{p}'),
#         }
        
#         logging.debug(f"Sending image to Node.js server for user: {data['user']}")
#         response = requests.post(f'http://localhost:4000/vendor/sktvendor/{data["user"]}', files=files)

#         if response.status_code != 200:
#             logging.error(f"Failed to upload image to Node.js server, status code: {response.status_code}")
#             return jsonify({"detail": "Failed to upload image to Node.js server"}), 500

#         logging.debug('Image successfully processed and sent to Node.js server')
#         return jsonify({"message": "Image processed successfully", "node_response": response.json()})
    
#     except Exception as e:
#         logging.error(f"Error processing the image: {str(e)}")
#         logging.debug(traceback.format_exc())
#         return jsonify({"detail": str(e)}), 500
# # Run the Flask app
# if __name__ == "__main__":
#     port = int(os.getenv("PORT", 5000))  # Use Render's PORT or default to 5000
#     app.run(host="localhost", port=port)








# import logging
# import os
# import traceback
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from dotenv import load_dotenv
# import numpy as np
# from PIL import Image
# from io import BytesIO
# import requests
# import gdown
# import torch
# import cv2
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision.transforms as transforms
# import torchvision.models as models
# # Load environment variables from .env file
# load_dotenv()

# # Setup logging
# logging.basicConfig(level=logging.DEBUG)

# # Define the file name and output path
# file_id = '1-FOXJcbmICLea_0_Xpz9rMQ8JDl8ZNny'
# output_file = "generator_epoch_50.pth"

# # Check if the model file already exists
# if not os.path.isfile(output_file):
#     # Construct the download URL
#     download_url = f"https://drive.google.com/uc?id={file_id}"

#     # Download the model file
#     gdown.download(download_url, output_file, quiet=False)
# else:
#     logging.info(f"{output_file} already exists. No download needed.")

# # Define the Generator model class
# class DownSample(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super(DownSample, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.LeakyReLU(0.2)
#         )

#     def forward(self, x):
#         return self.model(x)

# class Upsample(nn.Module):
#     def __init__(self, input_channels, output_channels):
#         super(Upsample, self).__init__()
#         self.model = nn.Sequential(
#             nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.InstanceNorm2d(output_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, x, skip_input):
#         x = self.model(x)
#         return torch.cat((x, skip_input), 1)

# class Generator(nn.Module):
#     def __init__(self, in_channels=3, out_channels=3):
#         super(Generator, self).__init__()
#         self.down1 = DownSample(in_channels, 64)
#         self.down2 = DownSample(64, 128)
#         self.down3 = DownSample(128, 256)
#         self.down4 = DownSample(256, 512)
#         self.down5 = DownSample(512, 512)
#         self.down6 = DownSample(512, 512)
#         self.down7 = DownSample(512, 512)
#         self.down8 = DownSample(512, 512)

#         self.up1 = Upsample(512, 512)
#         self.up2 = Upsample(1024, 512)
#         self.up3 = Upsample(1024, 512)
#         self.up4 = Upsample(1024, 512)
#         self.up5 = Upsample(1024, 256)
#         self.up6 = Upsample(512, 128)
#         self.up7 = Upsample(256, 64)

#         self.final = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(128, 3, kernel_size=4, padding=1),
#             nn.Tanh(),
#         )

#     def forward(self, x):
#         d1 = self.down1(x)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         d4 = self.down4(d3)
#         d5 = self.down5(d4)
#         d6 = self.down6(d5)
#         d7 = self.down7(d6)
#         d8 = self.down8(d7)
#         u1 = self.up1(d8, d7)
#         u2 = self.up2(u1, d6)
#         u3 = self.up3(u2, d5)
#         u4 = self.up4(u3, d4)
#         u5 = self.up5(u4, d3)
#         u6 = self.up6(u5, d2)
#         u7 = self.up7(u6, d1)
#         return self.final(u7)

# # Instantiate the model
# model = Generator()

# # Load the state dictionary
# try:
#     state_dict = torch.load(output_file, map_location=torch.device('cpu'))
#     # Create a new state dict for only the relevant keys
#     new_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()}
#     model.load_state_dict(new_state_dict)
#     model.eval()  # Set model to evaluation mode
#     logging.info("Model loaded successfully.")
# except RuntimeError as e:
#     logging.error(f"Error loading model state_dict: {e}")
#     raise
# except Exception as e:
#     logging.error(f"An unexpected error occurred: {e}")
#     raise

# def pencil_sketch_effect(img):
#     if img.dtype != np.uint8:
#         img = (img * 255).astype(np.uint8)  # Convert from float (0-1) to uint8 (0-255)

#     # Ensure the input image has 3 channels (BGR) before converting to grayscale
#     if len(img.shape) == 2:  # Single channel image, already grayscale
#         gray_img = img
#     elif len(img.shape) == 3 and img.shape[2] == 3:  # Color image (3 channels)
#         gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     else:
#         raise ValueError("Unexpected image format! Ensure the image has 1 or 3 channels.")

#     # Invert the grayscale image
#     inverted_img = cv2.bitwise_not(gray_img)
    
#     # Apply Gaussian blur
#     blurred_img = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)
    
#     # Invert the blurred image
#     inverted_blur = cv2.bitwise_not(blurred_img)
    
#     # Create the pencil sketch effect by blending
#     sketch_img = cv2.divide(gray_img, inverted_blur, scale=256.0)

#     return sketch_img
# # Create Flask app
# app = Flask(__name__)
# CORS(app)  # Allow CORS fo

# @app.route("/", methods=["GET"])
# def root():
#     return jsonify({"message": "Hello World"})

# @app.route("/predict", methods=["POST"])
# def predict():
#     max_upload_size = 16 * 1024 * 1024  # 16 MB
#     content_length = request.headers.get("Content-Length")

#     if content_length and int(content_length) > max_upload_size:
#         return jsonify({"detail": "Payload Too Large"}), 413

#     try:
#         logging.debug("Received request for prediction.")
        
#         data = request.get_json()
#         logging.debug(f"Received data: {data}")

#         if 'image' not in data:
#             return jsonify({"detail": "No image data provided"}), 400
        
#         image_data = data['image']['_streams'][1]
#         logging.debug(f"Image data URL: {image_data}")

#         # Extract file name and format
#         p = image_data.split('/')[-1]
#         a = p.split('.')[-1].upper()
#         if a == 'JPG':
#             a = 'JPEG'
#             p = p.replace('JPG', 'JPEG')

#         response = requests.get(image_data)
#         logging.debug(f"Fetched image with status code: {response.status_code} and size: {len(response.content)}")

#         img = Image.open(BytesIO(response.content)).convert('RGB')
#         img = img.resize((256, 256))
#         image = np.array(img) / 255.0
#         if(p.split('-')[-1]=="canvasimg.jpg"):
#             logging.debug("Pencil sketch effect mnnn")
#             image = pencil_sketch_effect(image)
#             if len(image.shape) == 3 and image.shape[2] == 1:  # Single channel grayscale
#                 image = np.repeat(image, 3, axis=2)
#             elif len(image.shape) == 2:  # Grayscale without channel dimension
#                 image = np.stack((image,) * 3, axis=-1)  # Duplicate across 3 channels
#             # Normalize and reshape to (1, 256, 256, 3) as expected by the model
#             image = image.astype(np.float32) / 255.0 
#         else:
#             logging.debug(f"No pencil sketch effect {p}")
        
#         image = np.expand_dims(image, axis=0)
#         image = torch.tensor(image, dtype=torch.float32).permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)

#         logging.debug(f"Input tensor shape: {image.shape}")

#         # Use the Generator model to make a prediction
#         with torch.no_grad():
#             prediction = model(image)
#         logging.debug(f"Prediction shape: {prediction.shape}")

#         # Convert the prediction back to an image
#         predicted_image = (prediction[0].numpy() * 255).astype(np.uint8)

#         # Check the shape and ensure it's in the right format
#         if predicted_image.shape[0] == 1:  # If there's a single channel
#             predicted_image = np.squeeze(predicted_image, axis=0)  # Remove the channel dimension if needed
#         if predicted_image.shape[0] == 3:  # Ensure it has three channels
#             predicted_image = np.transpose(predicted_image, (1, 2, 0))  # Change shape to (H, W, C)

#         output_image = Image.fromarray(predicted_image)
#         # Save the predicted image in memory (without writing to disk)
#         image_io = BytesIO()
#         output_image.save(image_io, format=a)
#         image_io.seek(0)

#         files = {
#             'images': (f'{p}', image_io, f'image/{a.lower()}'),
#             'name': (None, data['user']),
#             'filename': (None, f'{p}'),
#         }
        
#         logging.debug(f"Sending image to Node.js server for user: {data['user']}")
#         response = requests.post(f'http://localhost:4000/vendor/sktvendor/{data["user"]}', files=files)

#         if response.status_code != 200:
#             logging.error(f"Failed to upload image to Node.js server, status code: {response.status_code}")
#             return jsonify({"detail": "Failed to upload image to Node.js server"}), 500

#         logging.debug('Image successfully processed and sent to Node.js server')
#         return jsonify({"message": "Image processed successfully", "node_response": response.json()})
    
#     except Exception as e:
#         logging.error(f"Error processing the image: {str(e)}")
#         logging.debug(traceback.format_exc())
#         return jsonify({"detail": str(e)}), 500

# # Run the Flask app
# if __name__ == "__main__":
#     port = int(os.getenv("PORT", 5000))  # Use Render's PORT or default to 5000
#     app.run(host="localhost", port=port)


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
        image = np.expand_dims(image, axis=0).astype(np.float32)  # Shape it for model (1, 256, 256, 3)
        
        # Make a prediction using the TFLite Interpreter
        prediction = make_prediction(image)
        logging.debug(f"Prediction shape: {prediction.shape}")

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

