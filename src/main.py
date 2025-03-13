import os
import openai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from fastai.vision.all import *
import pathlib
from dotenv import load_dotenv
import platform
import cv2
import numpy as np

# Load environment variables
load_dotenv()
PORT=os.getenv("PORT")
URL1=os.getenv("WEB_APP_URL")

# Initialize FastAPI App
app = FastAPI()

start=time.time()
print("server starting.... ")

# Enable CORS for frontend
origins = [URL1]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI()

# Configure platform-specific path handling
plt = platform.system()
if plt == "Linux":
    pathlib.WindowsPath = pathlib.PosixPath

# Load the pre-trained model
learn = load_learner("export.pkl")
labels = learn.dls.vocab

# Directory for temporary image uploads
UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Face detection configuration
FACE_DETECTION_ENABLED = os.getenv('FACE_DETECTION_ENABLED', 'true').lower() == 'true'
MIN_FACE_CONFIDENCE = float(os.getenv('MIN_FACE_CONFIDENCE', '0.8'))

# Initialize face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_face(image_path):
    """
    Detect if there is a face in the image and return True if found
    """
    try:
        # Read the image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Return True if at least one face is detected
        return len(faces) > 0
    except Exception as e:
        print(f"Error in face detection: {str(e)}")
        return False

# Prediction function
def predict(image_path):
    try:
        # Load and process the image
        img = PILImage.create(image_path)

        # Make prediction
        pred, pred_idx, probs = learn.predict(img)

        # Get predictions with confidence scores
        predictions = []
        for i, (label, prob) in enumerate(zip(labels, probs)):
            if prob > 0.1:  # Only include predictions with >10% confidence
                predictions.append({
                    "disease": str(label),
                    "confidence": float(prob),
                    "description": get_disease_description(str(label))
                })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x["confidence"], reverse=True)
        return predictions[:3]  # Return top 3 predictions

    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return []

def get_disease_description(disease_name):
    # Add basic descriptions for each disease
    descriptions = {
        "Acne": "A skin condition characterized by red pimples on the skin, especially on the face, due to inflamed or infected sebaceous glands.",
        "Eczema": "A medical condition in which patches of skin become rough and inflamed with blisters causing itching and bleeding.",
        "Rosacea": "A condition that causes redness and visible blood vessels in your face, sometimes with small, red, pus-filled bumps.",
        "Psoriasis": "A skin disorder that causes skin cells to multiply up to 10 times faster than normal, resulting in bumpy red patches covered with white scales.",
        # Add more descriptions as needed
    }
    return descriptions.get(disease_name, "No description available.")

# ChatGPT function to get suggestions
def get_suggestions_from_chatgpt(predictions):
    try:
        if not predictions:
            return "No conditions detected to provide suggestions for."

        conditions = ", ".join([p["disease"] for p in predictions])
        prompt = f"""The following skin conditions were detected: {conditions}. 
        For each condition, please provide:
        1. Detailed description
        2. Common causes
        3. Treatment options
        4. Prevention tips
        Please format the response in a clear, organized way."""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"ChatGPT error: {str(e)}")
        return "Unable to generate suggestions at this time."

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.post("/predict")
async def handle_predict(file: UploadFile = File(...)):
    start=time.time()   
    try:
        if file.filename == "":
            raise HTTPException(status_code=400, detail="No file selected")

        if not allowed_file(file.filename):
            raise HTTPException(status_code=400, detail="Invalid file type")

        # Create a temporary file
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        
        # Save the uploaded file
        with open(filepath, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Check for face in image if enabled
        if FACE_DETECTION_ENABLED:
            has_face = detect_face(filepath)
            if not has_face:
                os.remove(filepath)  # Clean up the uploaded file
                raise HTTPException(status_code=400, detail="No face detected in the image. Please upload a clear image of a face.")

        # Continue with existing prediction logic
        predictions = predict(filepath)
        
        # Get treatment suggestions
        suggestions = get_suggestions_from_chatgpt(predictions) if predictions else ""

        # Cleanup
        if os.path.exists(filepath):
            os.remove(filepath)
        end=time.time()-start
        print(f"/predict time: {end:.2f}")
        return {
            "success": True,
            "predictions": predictions,
            "suggestions": suggestions
        }

    except Exception as e:
        print(f"Error in handle_predict: {str(e)}")
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(status_code=500, detail=str(e))

# Model for Chat request
class ChatRequest(BaseModel):
    question: str
    predictions: List[dict]

@app.post("/chat")
async def handle_chat(request: ChatRequest):
    start=time.time()
    try:
        question = request.question
        predictions = request.predictions
        
        # Format the context from predictions
        conditions = ", ".join([p["disease"] for p in predictions])
        descriptions = "\n".join([f"{p['disease']}: {p['description']}" for p in predictions])
        
        # Create a detailed prompt for ChatGPT
        prompt = f"""Context: The following skin conditions were detected:
{descriptions}

User Question: {question}

Please provide a detailed, medically-informed response addressing the user's question about these conditions. 
Include relevant medical information, but remind the user to consult a healthcare professional for proper diagnosis and treatment."""

        # Get response from ChatGPT        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        
        end=time.time()-start
        print(f"/chat time:{end:.2f}")

        return {
            "success": True,
            "response": response.choices[0].message.content
        }

    except Exception as e:
        print(f"Error in handle_chat: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail="Failed to process your question. Please try again."
        )

end=time.time()-start
print(f"end of server.... {end:.2f}")

# If running the file directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)