# Introduction

This is a Convolutional Neural Network (CNN) project that classifies plant leaf images into one of three categories: **Healthy**, **Powdery** (powdery mildew), and **Rust**.

# Methodology

I picked the **Plant Disease Recognition Dataset** from Kaggle. The dataset is already split into Train, Validation, and Test folders, so I loaded it directly with `tf.keras.utils.image_dataset_from_directory`. I then trained and compared four different CNN architectures, one notebook per model:

- **Model A** is defined in `model_a.ipynb`
- **Model B** is defined in `model_b.ipynb`
- **Model C** is defined in `model_c.ipynb`
- **Model D** is defined in `model_d.ipynb`

All trained Keras models are stored in the `models/` directory, and `testing_models.ipynb` evaluates the final model.

Model A served as a simple baseline, Model B added depth, Model C added regularization, and Model D combined the best ideas into a deeper, regularized custom network. Each model was evaluated with a classification report on the held-out test set. Finally, I created a FastAPI application (`main_api.py`) that uses the best trained model for inference.

# Dataset & Cleaning Process

The dataset is `plant-disease-recognition-dataset` from Kaggle (`rashikrahmanpritom/plant-disease-recognition-dataset`). It contains RGB leaf images organized into three classes:

- `Healthy`
- `Powdery`
- `Rust`

No manual cleaning was required because the dataset is already folder-structured. I only:

1. Verified the folder structure and class counts.
2. Loaded images with `image_dataset_from_directory`, resizing every image to the input size required by the model being trained.
3. Applied `cache()`, `shuffle()` (training set only), and `prefetch()` to speed up the data pipeline.
4. Used categorical labels (`label_mode="categorical"`) for all models.

Final Table

| Checkpoint | Value |
|---|---|
| Total training samples | 1,322 |
| Training Healthy | 458 |
| Training Powdery | 430 |
| Training Rust | 434 |
| Validation samples | 60 (20 per class) |
| Test samples | 150 (50 per class) |
| Classes | 3 |

# Training

I developed 4 different CNN strategies, referred to by letters as requested.

## Model A: Baseline

**Notebook:** `model_a.ipynb`

Model A is a small baseline CNN with two convolutional blocks. It has no batch normalization and only a single dense layer at the end.

```python
config = {
    "name": "Model_A_Baseline",
    "filters": [32, 64],
    "kernel_size": (3, 3),
    "batch_norm": False,
    "dropout": 0.3,
    "dense": [32]
}
```

This model failed on the test set: it predicted almost everything as class 0 (Healthy), achieving only **33% accuracy** and an F1-score near zero for Powdery and Rust.

## Model B: Deeper

**Notebook:** `model_b.ipynb`

Model B keeps the same simple structure as Model A but adds a third convolutional block with 128 filters, making the network deeper.

```python
config_model_b = {
    "name": "Model_B_Deeper",
    "filters": [32, 64, 128],
    "kernel_size": (3, 3),
    "batch_norm": False,
    "dropout": 0.0
}
```

Trained for 10 epochs with the Adam optimizer and sparse categorical crossentropy, it reached a validation accuracy of about **77%** and a validation loss of **0.55**. While better than Model A, it still lacked regularization and data augmentation.

## Model C: Regularized

**Notebook:** `model_c.ipynb`

Model C introduces batch normalization, dropout, and data augmentation (random flip, rotation, and zoom). It uses three convolutional blocks with 32, 64, and 128 filters, and an input size of 160x160.

Key architecture points:

- `RandomFlip`, `RandomRotation`, `RandomZoom` augmentation layers
- `BatchNormalization` after each convolution
- `GlobalAveragePooling2D` + `Dropout(0.4)`
- `Dense(3, activation="softmax")`

I ran two learning-rate experiments:

| Experiment | Best Epoch | Best Val Loss | Best Val Accuracy |
|---|---|---|---|
| lr = 0.001 | 10 | 0.3434 | 0.8833 |
| lr = 3e-4 | 15 | 0.4833 | 0.8167 |

The `lr = 0.001` run was saved as `models/model_final.keras`.

## Model D: Custom (Final / Best Model)

**Notebook:** `model_d.ipynb`

Model D is the deepest and most regularized architecture. It uses an input size of 224x224, four convolutional blocks (filters 32, 64, 128, 64), batch normalization after every convolution, data augmentation, and 40% dropout.

Architecture summary:

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

Total parameters: ~168K.

It was trained for up to 100 epochs with `EarlyStopping` and `ModelCheckpoint` monitoring validation loss. The `ModelCheckpoint` writes the best checkpoint to `model_d_best.keras` inside the notebook; the final selected model is committed as `models/best_model.keras`.

Best validation result:

- **Best epoch:** 32
- **Best validation loss:** 0.0914
- **Best validation accuracy:** 0.9667

# Evaluation

Although Model C showed solid validation performance, Model D achieved the best generalization on the held-out test set, so Model D was selected as the final production model.

### Test-Set Classification Report (Model D — `best_model.keras`)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Healthy | 0.88 | 1.00 | 0.93 | 50 |
| Powdery | 1.00 | 0.88 | 0.94 | 50 |
| Rust | 1.00 | 0.98 | 0.99 | 50 |
| **Accuracy** | | | **0.95** | 150 |
| Macro Avg | 0.96 | 0.95 | 0.95 | 150 |
| Weighted Avg | 0.96 | 0.95 | 0.95 | 150 |

### Model Comparison Summary

| Model | Test Accuracy | Notes |
|---|---|---|
| Model A Baseline | 0.33 | Collapsed to predicting mostly Healthy |
| Model B Deeper | ~0.77 val acc | Better, but no regularization |
| Model C Regularized | 0.88 val acc | Good, saved as `models/model_final.keras` |
| **Model D Custom** | **0.95** | **Best overall, saved as `models/best_model.keras`** |

# Deployment

I created a simple FastAPI application in `main_api.py` that loads `models/best_model.keras` and exposes a `/predict` endpoint. The API:

- Accepts an image upload
- Resizes it to 224x224
- Runs it through Model D
- Returns the predicted class, confidence, and per-class probabilities

Run the API locally with:

```bash
uvicorn main_api:app --reload
```

Then send an image to `http://localhost:8000/predict`.
