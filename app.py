
import logging
import os
import traceback
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from io import BytesIO
import requests
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import gdown
import tensorflow as tf
import warnings
from fastapi.responses import JSONResponse
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
import io
import cv2



warnings.filterwarnings("ignore")

# Load environment variables from .env file
load_dotenv()

# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
#please provide valid backend url or path **************************************
backend_url="http://localhost:4000"
#loading model
# file_id = ['1-OucVvEiaEqqKNx0IIDdjMNLYTZa4LAZ','1-A7KFVxO66TcgmGG-klWKCBVLb8uGSuL','10y0PBs-N9CCWFIzoMsHGN5MR9nasUyQH','1cGqzuPbH5WGDZ1wS5ATNcFPHSmZ_7UgC','1-05P0iR-eCWVadVL-gBM2qsxebMmX8Mf','1--WUD_oz7gwZy8d1hT6X0es362t0-LeJ','1IPUJj4d4fNeimhTeFfFCILXwGiTIOjTL','1-lGYX59mO7daC50OaMXcvQ_HoW4Vb-jj']
# output_file =['n1024p2p410.h5','dia105.h5','tepoch50.h5','jepoch10.h5','ear20.h5','ring36.h5','necklace5.h5','necklaces50.h5']
file_id = ['1-OucVvEiaEqqKNx0IIDdjMNLYTZa4LAZ','1-A7KFVxO66TcgmGG-klWKCBVLb8uGSuL','10y0PBs-N9CCWFIzoMsHGN5MR9nasUyQH','1cGqzuPbH5WGDZ1wS5ATNcFPHSmZ_7UgC']
output_file =['n1024p2p410.h5','dia105.h5','tepoch50.h5','jepoch10.h5']
for(i,j) in zip(file_id,output_file):
    if not os.path.isfile(j):
        # Construct the download URL
        download_url = f"https://drive.google.com/uc?id={i}"
        # Download the model file
        gdown.download(download_url, j, quiet=False)
    else:
        logging.info(f"{j} already exists. No download needed.")
# Load the models
types=['n1024p2p410.h5']*4
premodel =[]
# Modify config dictionary before loading the model
for i in types:
    premodel.append(load_model(i))
#silver model loadinng
sm="dia105.h5"
silvermodel=load_model(sm)
#classification model loading
saved_model_path = "tepoch50.h5" 
model = load_model(saved_model_path, compile=False)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])  # Update with the actual loss function
#segmentation model jewelry or not
jewelry_model_path = "jepoch10.h5"  # Update with correct path
jewelrymodel = load_model(jewelry_model_path, compile=False)
jewelrymodel.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define class names
class_names = ['bracelet', 'earring', 'necklace', 'ring']  # Update with your actual class names
jewelryclass_names=['jewelry','notjewelry']
#preprocessing
#pencil sketch conversion *****
def pencil_sketch_effect(img):
    # Ensure the input image has 3 channels (BGR) before converting to grayscale
    # Unexpected image format! Ensure the image has 1 or 3 channels.")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Invert the grayscale image
    inverted_img = cv2.bitwise_not(gray_img)
    
    # Apply Gaussian blur
    blurred_img = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)
    
    # Invert the blurred image
    inverted_blur = cv2.bitwise_not(blurred_img)
    
    # Create the pencil sketch effect by blending
    sketch_img = cv2.divide(gray_img, inverted_blur, scale=256.0)

    return sketch_img
def is_background_white(img_np):
    """Check if the background is predominantly white."""
    white_pixels = np.sum(np.all(img_np > [200, 200, 200], axis=-1))
    total_pixels = img_np.shape[0] * img_np.shape[1]
    return (white_pixels / total_pixels) > 0.9
# Function to convert the background of an image to white
def convert_background_to_white(image_data):
    # Decode the image from bytes and convert to numpy array
    img = Image.open(BytesIO(image_data)).convert('RGB')
    img_np = np.array(img)
    
    # If background is predominantly white, return original image data
    if is_background_white(img_np):
        return image_data
    
    # Convert to HSV for better background isolation
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    
    # Define a mask to isolate the background (non-white)
    lower_background = np.array([0, 0, 0])      # Lower bound for dark shades
    upper_background = np.array([180, 255, 100]) # Upper bound for light shades
    mask = cv2.inRange(hsv, lower_background, upper_background)
    
    # Replace non-white background with white
    img_np[mask == 0] = [255, 255, 255]
    
    # Convert back to PIL image, then to bytes
    white_bg_img = Image.fromarray(img_np)
    buffer = BytesIO()
    white_bg_img.save(buffer, format="PNG")
    return buffer.getvalue()
class ImageRequest(BaseModel):
    image: dict
    user: str

# THis predict function for classification passing required model and form data as parameter********
async def predict(image_data,model,user):
    try:
        # logging.debug("gold received request for prediction.")# Adjust according to your structure
        a = image_data.split('uploads/')[1]
        p = a
        a = a[len(a)-(len(a)-a.find('.'))+1:].upper()
        if a == 'JPG':
            a = 'JPEG'
            p = p.replace('JPG', 'JPEG')
        
        # Fetch the image from the URL
        response = requests.get(image_data)
        # image_data_with_white_bg = convert_background_to_white(response.content) remove complusory 
        imge = Image.open(BytesIO(response.content)).convert('RGB')

        # Resize the image to 1024x1024 for uniform processing
        imge = imge.resize((1024, 1024))

        # Convert the PIL Image to a NumPy array (OpenCV format)
        img_array = np.array(imge)

        # Apply the pencil sketch effect
        sketched_img = pencil_sketch_effect(img_array)

        # Convert the pencil sketch image back to a NumPy array
        sketched_img = np.expand_dims(sketched_img, axis=-1)  # Add channel dimension for grayscale
        sketched_img = np.repeat(sketched_img, 3, axis=-1)  # Convert to 3 channels (RGB equivalent)

        # Normalize and prepare for model input
        image = (sketched_img / 127.5) - 1  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension (batch_size, height, width, channels)
        # Add batch dimension
        # Make the prediction
        prediction = model.predict(image)
        # Convert the prediction back to an image
        # predicted_image = np.clip((prediction[0] + 1) * 127.5, 0, 255).astype(np.uint8)
        # Clip first, then round to ensure all values are within range and integers
        predicted_image = np.clip((prediction[0] + 1) * 127.5, 0, 255).astype(np.uint8)
        output_image = Image.fromarray(predicted_image)

        # output_image = Image.fromarray(predicted_image)
        
        # Save the predicted image in memory (without writing to disk)
        image_io = BytesIO()
        output_image.save(image_io, format=a)  # Save as JPEG or other format
        image_io.seek(0)  # Move to the beginning of the BytesIO buffer

        # Send the image to the Node.js server
        files = {
            'images': (f'{p}', image_io, f'image/{a.lower()}'),
            'name': (None, user),
            'filename': (None, f'{p}'),
        }
        
        # logging.debug(f"Sending image to Node.js server for user: {user}")
        response = requests.post(f'{backend_url}/vendor/sktvendor/{user}', files=files)

        # Check if the response from Node.js server is successful
        if response.status_code != 200:
            logging.error(f"Failed to upload image to Node.js server, status code: {response.status_code}")
            raise HTTPException(status_code=500, detail="Failed to upload image to Node.js server")

        logging.debug('Image successfully processed and sent to Node.js server')
        return {"message": "Image processed successfully", "node_response": response.json()}
    
    except Exception as e:
        logging.error(f"Error processing the image: {str(e)}")
        logging.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Hello World"}

# this is  for classification giving their respected model function ********
@app.post("/gold")
async def predict_image(request:Request):
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
        image_data = data['image']['_streams'][1]  # Adjust according to your structur
        response = requests.get(image_data)
        # whitebg=convert_background_to_white(response.content)
        # logging.debug("Preprocessing image...1")
        pimage = Image.open(BytesIO(response.content)).convert('RGB')
        # Preprocess the image
        image = pimage.resize((224, 224))  # Resize image to match input size
        classimg_array = np.array(image)
        classsketched_img = pencil_sketch_effect(classimg_array)
        # logging.debug("Preprocessing image...2")
        # Convert the pencil sketch image back to a NumPy array
        classsketched_img = np.expand_dims(classsketched_img, axis=-1)  # Add channel dimension for grayscale
        classsketched_img = np.repeat(classsketched_img, 3, axis=-1)
        # logging.debug("Preprocessing image...3")
        img_array = (img_to_array(classsketched_img) / 127.5) -1  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        predictions = model.predict(img_array)
        # logging.debug("Preprocessing image...4")
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_idx]
        
        for i in range(len(premodel)):
            if(predicted_class_name is class_names[i]):
                genmodel=premodel[i]
                logging.debug("Preprocessing image...4"+str(genmodel)+predicted_class_name+class_names[i])
                break
        # logging.debug("Preprocessing image...5"+str(genmodel)+predicted_class_name)
        output= await predict(image_data,genmodel,data['user'])
        # logging.debug(output)
        # Return the predicted class
        return {"predicted_class": predicted_class_name, "output": output}
    except Exception as e:
        logging.error(f"Error processing the models: {str(e)}")
        logging.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# classification for jewelry or not 
@app.post("/predict")
async def predict_jewelry(request: Request):
    try:
        # Parse JSON body
        data = await request.json()
        # Check for image data
        if 'image' not in data:
            raise HTTPException(status_code=400, detail="No image data provided")
        # Extract image URL from the JSON structure
        image_data = data['image']['_streams'][1]  # Adjust according to your structure
        response = requests.get(image_data)
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        image = image.resize((224, 224))  # Resize image to match input size
        img_array = np.array(image)
        sketched_img = pencil_sketch_effect(img_array)
        sketched_img = np.expand_dims(sketched_img, axis=-1)  # Add channel dimension for grayscale
        sketched_img = np.repeat(sketched_img, 3, axis=-1)
        img_array = (img_to_array(sketched_img) / 127.5) -1  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  
        # Perform prediction
        predictions = jewelrymodel.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        predicted_class_name = jewelryclass_names[predicted_class_idx]
        # Return the predicted class
        if predicted_class_name == 'jewelry':
            return {"success": True, "predicted_class": predicted_class_name}
        else:
            return {"success": False, "predicted_class": predicted_class_name}
    except Exception as e:
        logging.error(f"Error processing the image: {str(e)}")

# silver model process here 
@app.post("/silver")
async def silverpredict(request: Request):
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
        image_data_with_white_bg = convert_background_to_white(response.content)
        imge = Image.open(BytesIO(image_data_with_white_bg)).convert('RGB')

        # Resize the image to 1024x1024 for uniform processing
        imge = imge.resize((1024, 1024))

        # Convert the PIL Image to a NumPy array (OpenCV format)
        img_array = np.array(imge)

        # Apply the pencil sketch effect
        sketched_img = pencil_sketch_effect(img_array)

        # Convert the pencil sketch image back to a NumPy array
        sketched_img = np.expand_dims(sketched_img, axis=-1)  # Add channel dimension for grayscale
        sketched_img = np.repeat(sketched_img, 3, axis=-1)  # Convert to 3 channels (RGB equivalent)

        # Normalize and prepare for model input
        image = (sketched_img / 127.5) - 1  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension (batch_size, height, width, channels)
        # Fetch the image from the URL
        prediction = silvermodel.predict(image)

        # Convert the prediction back to an image
        # predicted_image = np.clip((prediction[0] + 1) * 127.5, 0, 255).astype(np.uint8)
        # Clip first, then round to ensure all values are within range and integers
        predicted_image = np.clip((prediction[0] + 1) * 127.5, 0, 255).astype(np.uint8)
        output_image = Image.fromarray(predicted_image)

        # output_image = Image.fromarray(predicted_image)
        
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
        response = requests.post(f'{backend_url}/vendor/sktvendor/{data["user"]}', files=files)

        # Check if the response from Node.js server is successful
        if response.status_code != 200:
            logging.error(f"Failed to upload image to Node.js server, status code: {response.status_code}")
            raise HTTPException(status_code=500, detail="Failed to upload image to Node.js server")

        logging.debug('Image successfully processed and sent to Node.js server')
        return {"message": "Image processed successfully", "node_response": response.json()}
    
    except Exception as e:
        logging.error(f"Error processing the image: {str(e)}")
        logging.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 5000))  # Use Render's PORT or default to 8000
    uvicorn.run(app, host="localhost", port=port)
