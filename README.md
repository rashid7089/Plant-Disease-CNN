# Methodology

[THE METHODOLOGY.md FILE!!](Methodology.md)

# Plant Disease Classifier

Classifies plant leaf images into three categories — **Healthy**, **Powdery**, and **Rust** — using a custom CNN. Achieves **0.95 test accuracy** and **0.95 macro F1-score** with the final Model D architecture.

## Dataset Cleaning & Checking

- **Source:** Plant Disease Recognition Dataset from Kaggle (`rashikrahmanpritom/plant-disease-recognition-dataset`).
- **Structure:** Already split into `Train`, `Validation`, and `Test` folders with class subfolders.
- **Loading:** Images are loaded with `tf.keras.utils.image_dataset_from_directory` and resized to the input size required by each model.
- **Pipeline:** Training data is cached, shuffled, and prefetched with `tf.data.AUTOTUNE` for faster training.

| Checkpoint | Value |
|---|---|
| Total training samples | 1,322 |
| Training Healthy | 458 |
| Training Powdery | 430 |
| Training Rust | 434 |
| Validation samples | 60 (20 per class) |
| Test samples | 150 (50 per class) |
| Classes | 3 |

## Model & Final Configuration

**Final model:** Model D Custom CNN (`best_model.keras`)

**Input size:** 224 × 224 RGB

**Architecture:**

```
Input (224, 224, 3)
  → RandomFlip
  → RandomRotation(0.1)
  → RandomZoom(0.1)
  → Rescaling(1.0 / 255)
  → Conv2D(32) → BatchNorm → ReLU → MaxPool
  → Conv2D(64) → BatchNorm → ReLU → MaxPool
  → Conv2D(128) → BatchNorm → ReLU → MaxPool
  → Conv2D(64) → BatchNorm → ReLU → MaxPool
  → GlobalAveragePooling2D
  → Dropout(0.4)
  → Dense(3, softmax)
```

**Training settings:**

- Optimizer: Adam
- Loss: categorical crossentropy
- Metric: accuracy
- Callbacks: `EarlyStopping` and `ModelCheckpoint` monitoring validation loss
- Best validation result: **96.67% accuracy** at epoch 32

**Test-set performance:**

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Healthy | 0.88 | 1.00 | 0.93 |
| Powdery | 1.00 | 0.88 | 0.94 |
| Rust | 1.00 | 0.98 | 0.99 |
| **Accuracy** | | | **0.95** |
| **Macro F1** | | | **0.95** |

## API

A FastAPI app is provided in `main_api.py`. It loads `best_model.keras` and exposes a `/predict` endpoint for image classification.

Run it with:

```bash
uvicorn main_api:app --reload
```

Example request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@leaf_image.jpg"
```

Example response:

```json
{
  "predicted_class": "Rust",
  "confidence": 0.9876,
  "all_probabilities": {
    "Healthy": 0.0089,
    "Powdery": 0.0035,
    "Rust": 0.9876
  }
}
```
