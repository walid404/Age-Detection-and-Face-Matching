# Age Detection and Face Matching

A comprehensive deep learning project for **age prediction** and **face identity matching** using the FG-NET dataset. This project combines age estimation with face recognition to enable age-aware identity verification.

## 🎯 Project Overview

This project implements two main pipelines:

1. **Age Prediction**: Predicts the age of a person in an image using various CNN architectures (ResNet, MobileNet, EfficientNet, etc.)
2. **Face Matching**: Performs face identity verification using ArcFace embeddings with adjustable similarity thresholds

### Key Features

- ✅ **Multiple Model Architectures**: ResNet18, ResNet34, ResNet50, MobileNet, AlexNet, EfficientNet
- ✅ **Identity-Aware Data Splits**: Prevents identity leakage between train/val/test sets
- ✅ **Early Stopping**: Prevents overfitting with patience-based early stopping
- ✅ **MLflow Tracking**: Comprehensive experiment logging and comparison
- ✅ **Threshold Optimization**: Automatic face matching threshold selection
- ✅ **Data Visualization**: EDA plots, identity distributions, loss curves, and predictions

## 📁 Project Structure

```
Age-Detection-and-Face-Matching/
├── src/
│   ├── config/
│   │   └── config.yaml                 # Configuration file
│   ├── controller/                     # Business logic
│   │   ├── age_inference_controller.py
│   │   ├── age_face_inference_controller.py
│   │   ├── face_match_inference_controller.py
│   │   ├── face_match_evaluation_controller.py
│   │   ├── face_match_threshold_*.py
│   │   └── train_controller.py
│   ├── model/
│   │   ├── datasets/
│   │   │   ├── age_dataset.py          # PyTorch Dataset
│   │   │   ├── data_preparation.py     # Download & prepare data
│   │   │   ├── identity_split.py       # Identity-aware splitting
│   │   │   └── split_factory.py
│   │   ├── networks/
│   │   │   ├── age_models.py           # Age prediction models
│   │   │   ├── arcface_model.py        # Face embedding extraction
│   │   │   └── face_matcher.py         # Face matching logic
│   │   ├── training/
│   │   │   ├── trainer.py              # Training loop
│   │   │   ├── losses.py               # Loss functions
│   │   │   └── early_stopping.py       # Early stopping logic
│   │   └── utils/
│   │       ├── load.py                 # Image/model loading utilities
│   │       └── age_gap.py              # Age gap metrics
│   ├── scripts/
│   │   ├── prepare_data.py             # Dataset preparation
│   │   ├── age_inference.py            # Age prediction CLI
│   │   ├── face_age_inference.py       # Combined age + face matching CLI
│   │   ├── face_match_threshold_selection.py
│   │   ├── run_eda.py                  # EDA visualization
│   │   ├── run_full_experiment.py      # End-to-end pipeline
│   │   ├── compare_splits.py           # Split strategy comparison
│   │   ├── generate_mlflow_dashboard.py
│   │   └── generate_comparison_tables.py
│   ├── view/
│   │   ├── eda_plots.py                # EDA visualizations
│   │   ├── identity_distribution.py    # Identity distribution plots
│   │   ├── loss_plot.py
│   │   └── visualize_predictions.py    # Prediction samples
│   ├── Dataset/                        # (Downloaded during setup)
│   ├── saved_models/                   # Pre-trained model weights
│   ├── reports/
│   │   ├── plots/                      # Generated visualizations
│   │   └── tables/                     # Results tables (CSV)
│   └── requirements.txt                # Python dependencies
├── .gitignore
└── README.md
```

## 🔧 Environment Setup

### Prerequisites

- Python 3.9+ 
- pip or conda
- CUDA 11.8+ (optional, for GPU acceleration)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/walid404/Age-Detection-and-Face-Matching.git
cd Age-Detection-and-Face-Matching
```

#### 2. Create Virtual Environment

```bash
# Using venv
python -m venv my_env
source my_env/bin/activate  # On Windows: my_env\Scripts\activate

# OR using conda
conda create -n age-detection python=3.9
conda activate age-detection
```

#### 3. Install Dependencies

```bash
cd src
pip install -r requirements.txt
```

**Dependencies**:
- `torch==2.9.1` - Deep learning framework
- `torchvision==0.24.1` - Computer vision utilities
- `deepface==0.0.97` - Face recognition (ArcFace)
- `mlflow==3.8.1` - Experiment tracking
- `scikit-learn==1.8.0` - Metrics & utilities
- `pillow==12.1.0` - Image processing
- `PyYAML==6.0.3` - Configuration files
- `tqdm==4.67.1` - Progress bars
- `requests==2.32.5` - HTTP library

## 📊 Data Preparation

### Automatic Dataset Download

The FG-NET dataset will be automatically downloaded and extracted during the first run:

```bash
cd src
python scripts/prepare_data.py \
  --dataset_root Dataset \
  --dataset_name FGNET \
  --images_dir_name images \
  --labels_csv_name labels.csv
```

This will:
1. Download the FG-NET dataset (~500MB)
2. Extract images to `src/Dataset/FGNET/images/`
3. Generate `labels.csv` with person_id and age annotations

**Dataset Format**:
- **Images**: JPEG files with naming convention `{person_id:03d}A{age:02d}.jpg`
- **Labels CSV**: Columns `[image_name, person_id, age]`

### Generated Pair Data

For face matching tasks, face pair datasets are generated:

```bash
python scripts/face_match_threshold_selection.py
```

This creates:
- `train_pairs.csv` - Training face pairs (positive & negative)
- `test_pairs.csv` - Test face pairs (positive & negative)

## 🚀 Quick Start

### 1. Run Full Experiment Pipeline

Execute the complete end-to-end pipeline:

```bash
cd src
python scripts/run_full_experiment.py
```

This will:
- Run EDA and generate plots
- Train multiple age prediction models with different configurations
- Compare random vs identity-aware data splits
- Generate MLflow dashboards and comparison tables

### 2. Age Prediction (Single Image)

```bash
cd src
python scripts/age_inference.py \
  --image_path path/to/image.jpg \
  --model mobilenet \
  --checkpoint saved_models/mobilenet_identity_bs16_lr0.0005_ep60.pt \
  --img_size 224
```

**Output**: Predicted age as integer

### 3. Age-Aware Face Matching (Two Images)

```bash
cd src
python scripts/face_age_inference.py \
  --image1 path/to/person1.jpg \
  --image2 path/to/person2.jpg \
  --model mobilenet \
  --weights saved_models/mobilenet_identity_bs16_lr0.0005_ep60.pt \
  --threshold 0.45
```

**Output**:
```json
{
  "image_1": {
    "path": "...",
    "predicted_age": 25
  },
  "image_2": {
    "path": "...",
    "predicted_age": 28
  },
  "match": 1,
  "similarity": 0.67
}
```

### 4. Run EDA Only

```bash
cd src
python scripts/run_eda.py \
  --plot_dir reports/plots \
  --dataset_root Dataset \
  --dataset_name FGNET \
  --labels_csv_name labels.csv
```

## ⚙️ Configuration

Edit `src/config/config.yaml` to customize training:

```yaml
dataset:
  dataset_root: src/Dataset
  dataset_name: FGNET
  img_size: 224
  train_split: 0.7
  val_split: 0.1
  split_strategies: ["random", "identity"]

training:
  batch_size: [16, 32]              # Multiple values for grid search
  epochs: [20, 40]
  learning_rate: [0.0001, 0.0005]
  loss: mse                          # or "mae"
  patience: 5                        # Early stopping patience

models:
  names:
    - resnet18
    - mobilenet
    - efficientnet
    # Add more: resnet34, resnet50, alexnet

mlflow:
  experiment_name: Age_Prediction_Full_Comparison
  model_dir: src/saved_models
```

## 📈 Training & Evaluation

### Model Architectures

Supported age prediction models:
- **ResNet18/34/50**: Residual networks with different depths
- **MobileNet**: Lightweight for edge deployment
- **EfficientNet**: State-of-the-art efficiency
- **AlexNet**: Classic deep CNN

All use **pretrained ImageNet weights** and are fine-tuned on FG-NET.

### Training Details

- **Loss Function**: MSE or MAE
- **Optimizer**: Adam with learning rate scheduling
- **LR Scheduler**: ReduceLROnPlateau (reduces LR on validation loss plateau)
- **Early Stopping**: Stops training if validation loss doesn't improve for N epochs

### Data Splitting Strategies

#### Random Split
- Simple image-level split (train/val/test ratios: 70/10/20)
- **Risk**: Identity leakage (same person in multiple splits)

#### Identity-Aware Split ⭐ (Recommended)
- Ensures each person appears in only ONE split
- Prevents identity information leakage
- More realistic evaluation of model generalization

## 📊 Results & Evaluation

### Metrics

**Age Prediction**:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- R² Score

**Face Matching**:
- Accuracy
- Precision
- Recall
- F1-Score
- Average Similarity (for match/non-match pairs)

### View Results

```bash
cd src

# Generate MLflow dashboard
python scripts/generate_mlflow_dashboard.py \
  --experiment_name Age_Prediction_Full_Comparison \
  --plots_dir reports/plots \
  --tables_dir reports/tables

# View MLflow UI
mlflow ui --host 127.0.0.1 --port 5000
# Open http://localhost:5000 in browser
```

### Exported Artifacts

- `reports/plots/` - Visualizations (loss curves, predictions, distributions)
- `reports/tables/` - Comparison tables (CSV format)
- `saved_models/` - Trained model weights (.pt files)
- `mlruns/` - MLflow tracking data

## 🔍 Advanced Usage

### Face Matching Threshold Selection

Automatically find the optimal similarity threshold:

```bash
cd src
python scripts/face_match_threshold_selection.py \
  --dataset_root Dataset \
  --dataset_name FGNET \
  --thresholds 0.3 0.35 0.4 0.45 0.5 0.55 0.6 \
  --optimize_metric f1 \
  --save_results True
```

### Compare Split Strategies

Analyze performance difference between random and identity-aware splits:

```bash
cd src
python scripts/compare_splits.py \
  --experiment_name Age_Prediction_Full_Comparison \
  --plots_dir reports/plots
```

### Batch Inference

Process multiple images for age prediction or face matching using the controller classes:

```python
from src.controller.age_face_inference_controller import AgeFaceMatchingInference

pipeline = AgeFaceMatchingInference(
    age_model_name="mobilenet",
    age_model_weights="src/saved_models/mobilenet_identity_bs16_lr0.0005_ep60.pt",
    match_threshold=0.45
)

result = pipeline.infer("image1.jpg", "image2.jpg")
print(result)
```

## 🔗 Key Classes & Functions

### Model Loading
- [`load_model()`](src/model/utils/load.py) - Load pretrained age models
- [`get_age_model()`](src/model/networks/age_models.py) - Create model architecture

### Datasets
- [`AgeDataset`](src/model/datasets/age_dataset.py) - PyTorch Dataset for age data
- [`identity_aware_split()`](src/model/datasets/identity_split.py) - Identity-aware splitting
- [`generate_face_matching_pairs()`](src/model/datasets/data_preparation.py) - Create pair datasets

### Face Matching
- [`ArcFaceExtractor`](src/model/networks/arcface_model.py) - Extract face embeddings
- [`FaceMatcher`](src/model/networks/face_matcher.py) - Match faces by similarity

### Training
- [`train_epoch()`](src/model/training/trainer.py) - Single training epoch
- [`evaluate_full()`](src/model/training/trainer.py) - Full evaluation metrics
- [`EarlyStopping`](src/model/training/early_stopping.py) - Early stopping logic

## 🐛 Troubleshooting

### CUDA/GPU Issues
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CPU-only PyTorch:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### DeepFace Model Download
The first face matching call downloads pre-trained models (~300MB). Ensure internet connectivity.

### Memory Issues
- Reduce `batch_size` in config.yaml
- Use `--model mobilenet` (lighter than ResNet50)
- Process images one-at-a-time instead of batches

### Dataset Not Found
```bash
# Re-download and prepare dataset
cd src
python scripts/prepare_data.py
```

## 📚 Citation & References

- **FG-NET Dataset**: [Aging Database](https://yanweifu.github.io/FG_NET_data/)
- **ArcFace**: [Deep Insight](https://github.com/deepinsight/insightface)
- **MLflow**: [MLflow Documentation](https://mlflow.org/docs)

## 📝 License

This project is part of CYSheild Tasks. 

## ✉️ Contact

For issues or questions, please open an issue on the GitHub repository.

---

**Happy experimenting!** 🚀