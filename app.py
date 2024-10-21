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

# Load environment variables from .env file
load_dotenv()

# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

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
file_id = '1-Bsdmj9TK8DFM-HSwipckPW5a0RNNKVA'
output_file = 'generator_epoch_10.h5'

# Check if the model file already exists
if not os.path.isfile(output_file):
    # Construct the download URL
    download_url = f"https://drive.google.com/uc?id={file_id}"
    
    # Download the model file
    gdown.download(download_url, output_file, quiet=False)
else:
    logging.info(f"{output_file} already exists. No download needed.")

# Load the model
model = load_model(output_file)

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
        img = Image.open(BytesIO(response.content)).convert('RGB')  # Ensure it's in RGB format
        
        # Preprocess the image for the model
        img = img.resize((256, 256))  # Resize image for the model
        image = np.array(img) / 255.0  # Normalize the image
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Make the prediction
        prediction = model.predict(image)

        # Convert the prediction back to an image
        predicted_image = (prediction[0] * 255).astype(np.uint8)
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
        return {"message": "Image processed successfully", "node_response": response.json()}
    
    except Exception as e:
        logging.error(f"Error processing the image: {str(e)}")
        logging.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))  # Use Render's PORT or default to 8000
    uvicorn.run(app, host="0.0.0.0", port=port)
