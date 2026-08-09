# Methodology

<!-- [THE METHODOLOGY.md FILE!!](Methodology.md) -->

# Plant Disease CNN

Classifies leaf images as **Healthy**, **Powdery**, or **Rust** using a custom CNN trained on a Kaggle plant-disease dataset. Achieves **0.95 test accuracy** and **0.96 macro F1** with a batch-normalized 4-block CNN plus data augmentation.

## Usage

### Training
The experiments live in the Jupyter notebooks:
- `model_a_and_d.ipynb` — baseline + final custom model
- `model_c.ipynb` — regularized CNN learning-rate sweep
- `Main_2.ipynb` — deeper-layers experiment (Model B)
- `testing_models.ipynb` — loads and evaluates `best_model.keras` on the test set

### Serving the model (FastAPI)
```bash
uvicorn main_api:app --reload
```
Post an image to `http://localhost:8000/predict` and get back:
```json
{
  "predicted_class": "Rust",
  "confidence": 0.99,
  "all_probabilities": {"Healthy": 0.0, "Powdery": 0.01, "Rust": 0.99}
}
```

## Dataset Cleaning & Checking

- **Source:** `plant-disease-recognition-dataset` pulled via `kagglehub`, pre-split into `Train` (1,322), `Validation` (60), and `Test` (150).
- **No null handling:** images load directly from class folders; no nulls or missing labels exist.
- **Balanced classes:** Healthy (458), Powdery (430), Rust (434) — nearly even, no resampling needed.
- **Preprocessing:** resized to `224×224`, rescaled to `[0,1]`, augmented (flip / rotate ±10° / zoom 10%) during training.

| Checkpoint | Value |
|---|---|
| Total training samples | 1,322 |
| Validation samples | 60 |
| Test samples | 150 |
| Classes | 3 (Healthy, Powdery, Rust) |
| Features | 224×224×3 RGB pixels |

## Model & Final Configuration

**Algorithm:** Custom Convolutional Neural Network.

**Final architecture:**
```
Conv2D(32) → BN(0.9) → ReLU → MaxPool
Conv2D(64) → BN(0.9) → ReLU → MaxPool
Conv2D(128) → BN(0.9) → ReLU → MaxPool
Conv2D(64)  → BN(0.9) → ReLU → MaxPool
GlobalAveragePooling2D → Dropout(0.4) → Dense(3, softmax)
```

**Training configuration:**
- `Image size = (224, 224, 3)`, `Batch size = 32`
- `Optimizer = Adam(learning_rate = 0.05)`
- `Loss = categorical_crossentropy`, metric = `accuracy`
- `EarlyStopping` / `ModelCheckpoint`, epochs = 100

**Test results (150 images):** precision `0.96`, recall `0.95`, F1 `0.95`, accuracy `0.95`.