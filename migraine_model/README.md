# Migraine Prediction Model

Production-ready XGBoost model for predicting migraine probability from comprehensive feature sets.

## Features

- **Target**: Binary classification (migraine: True/False)
- **Output**: Probability of migraine (0.0 to 1.0)
- **Model**: XGBoost with hyperparameter optimization
- **Robust Design**: Stratified splitting, early stopping, regularization, comprehensive evaluation

## Installation

```bash
pip install -r requirements.txt
```

Required packages:
- xgboost (>=2.0.0)
- optuna (for hyperparameter optimization)
- scikit-learn
- pandas, numpy
- matplotlib, seaborn (for visualizations)

## Usage

### Basic Training

```bash
python train_migraine_model.py --input combined_data.csv --output models/
```

### With Custom Options

```bash
python train_migraine_model.py \
    --input combined_data.csv \
    --output models/ \
    --trials 200 \
    --random-state 42
```

### Arguments

- `--input, -i`: Input CSV file path (default: combined_data.csv)
- `--output, -o`: Output directory for model artifacts (default: models)
- `--trials`: Number of hyperparameter optimization trials (default: 100)
- `--random-state`: Random seed for reproducibility (default: 42)

## Input Data Format

The CSV file should contain the following columns:

**Features (21):**
- `gender` (categorical: male/female)
- `migraine_days_per_month` (int)
- `stress_intensity` (int: 0-5)
- `temp_mean`, `wind_mean`, `pressure_mean`, `sun_irr_mean`, `sun_time_mean`, `precip_total`, `cloud_mean` (float)
- `step_count_normalized` (float: 0-1)
- `mood_score` (float)
- `mood_category` (categorical: Very Low, Low, Moderate, Good, Very Good)
- `screen_brightness_normalized` (float: 0-1)
- `stress`, `hormonal`, `sleep`, `weather`, `food`, `sensory`, `physical` (float: 0-1)

**Target (1):**
- `migraine` (boolean: True/False)

## Output Files

After training, the following files are saved to the output directory:

1. **migraine_model.json** - Trained XGBoost model
2. **feature_engineer.pkl** - Feature preprocessing pipeline
3. **feature_names.json** - List of feature names
4. **feature_importance.csv** - Feature importance scores
5. **model_metadata.json** - Model hyperparameters and metrics
6. **model_evaluation_report.txt** - Comprehensive evaluation report
7. **roc_curve.png** - ROC curve visualization
8. **calibration_curve.png** - Probability calibration curve
9. **feature_importance.png** - Top 20 feature importance plot

## Model Architecture

- **Algorithm**: XGBoost (Gradient Boosting)
- **Objective**: binary:logistic (probability output)
- **Evaluation Metric**: logloss, AUC
- **Early Stopping**: 50 rounds patience
- **Class Imbalance**: Handled with scale_pos_weight
- **Regularization**: L1 (reg_alpha) and L2 (reg_lambda)

## Model Performance

The model is evaluated on three sets:
- **Training Set** (70%): Used for model training
- **Validation Set** (15%): Used for early stopping and hyperparameter tuning
- **Test Set** (15%): Final evaluation (untouched during training)

Metrics reported:
- ROC-AUC Score
- Log Loss
- Brier Score
- Precision, Recall, F1-Score
- Accuracy
- Confusion Matrix

## Making Predictions

To use the trained model for predictions:

```python
import xgboost as xgb
import pandas as pd
import pickle
import json

# Load model
model = xgb.Booster()
model.load_model('models/migraine_model.json')

# Load feature engineer
with open('models/feature_engineer.pkl', 'rb') as f:
    feature_engineer = pickle.load(f)

# Load feature names
with open('models/feature_names.json', 'r') as f:
    feature_names = json.load(f)

# Prepare new data
new_data = pd.DataFrame([{
    'gender': 'female',
    'migraine_days_per_month': 5,
    # ... other features
}])

# Transform features
X_processed = feature_engineer.transform(new_data)

# Create DMatrix
dtest = xgb.DMatrix(X_processed)

# Predict probability
probability = model.predict(dtest)[0]
print(f"Migraine probability: {probability:.4f}")
```

## Notes

- Constant features (hormonal, sleep) are automatically removed
- Categorical features are encoded automatically
- The model treats rows independently (no temporal dependencies)
- All random seeds are fixed for reproducibility

