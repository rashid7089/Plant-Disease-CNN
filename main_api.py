from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import numpy as np
import io
import tensorflow as tf

# --- Configuration (must match training exactly) ---
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Healthy", "Powdery", "Rust"]

app = FastAPI(title="Plant Disease Classifier")

# Load the trained model once, when the server starts
model = tf.keras.models.load_model("./models/best_model.keras")


@app.get("/")
async def root():
    return {"message": "Plant Disease Classifier API is running."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Step 1: Validate the uploaded file is actually an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid image file.")

    # Step 2: Read and preprocess the image (must match training pipeline)
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(IMG_SIZE)
    array = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)

    # Step 3: Run prediction
    # Note: no manual rescaling here — the model already includes
    # a Rescaling(1./255) layer internally, applied automatically.
    probs = model.predict(array, verbose=0)[0]
    best_idx = int(np.argmax(probs))

    # Step 4: Return a clean, structured response
    return {
        "predicted_class": CLASS_NAMES[best_idx],
        "confidence": float(probs[best_idx]),
        "all_probabilities": {
            CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
        }
    }