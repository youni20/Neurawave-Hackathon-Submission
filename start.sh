#!/bin/bash

# Migraine Prediction Model Training Script
# This script trains the XGBoost model with all necessary steps
# Designed for cloud environments - assumes data files are already prepared

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
INPUT_FILE="combined_data.csv"
OUTPUT_DIR="models"
TRIALS=100
RANDOM_STATE=42

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Migraine Prediction Model Training${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Check Python
echo -e "${YELLOW}[1/5] Checking Python installation...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python is not installed or not in PATH${NC}"
    exit 1
fi
PYTHON_VERSION=$(python --version 2>&1)
echo -e "${GREEN}✓ Found: $PYTHON_VERSION${NC}"
echo ""

# Step 2: Check required packages
echo -e "${YELLOW}[2/5] Checking required packages...${NC}"
MISSING_PACKAGES=()

python -c "import pandas" 2>/dev/null || MISSING_PACKAGES+=("pandas")
python -c "import numpy" 2>/dev/null || MISSING_PACKAGES+=("numpy")
python -c "import xgboost" 2>/dev/null || MISSING_PACKAGES+=("xgboost")
python -c "import optuna" 2>/dev/null || MISSING_PACKAGES+=("optuna")
python -c "import sklearn" 2>/dev/null || MISSING_PACKAGES+=("scikit-learn")
python -c "import matplotlib" 2>/dev/null || MISSING_PACKAGES+=("matplotlib")
python -c "import seaborn" 2>/dev/null || MISSING_PACKAGES+=("seaborn")

if [ ${#MISSING_PACKAGES[@]} -gt 0 ]; then
    echo -e "${RED}Error: Missing required packages: ${MISSING_PACKAGES[*]}${NC}"
    echo -e "${YELLOW}Installing missing packages...${NC}"
    pip install ${MISSING_PACKAGES[@]}
    echo -e "${GREEN}✓ Packages installed${NC}"
else
    echo -e "${GREEN}✓ All required packages are installed${NC}"
fi
echo ""

# Step 3: Check input data file
echo -e "${YELLOW}[3/5] Checking input data file...${NC}"
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file '$INPUT_FILE' not found${NC}"
    echo -e "${YELLOW}Note: Data generation scripts are not available in cloud environment${NC}"
    echo -e "${YELLOW}Please ensure $INPUT_FILE is uploaded before training${NC}"
    exit 1
fi

# Check file is not empty
if [ ! -s "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file '$INPUT_FILE' is empty${NC}"
    exit 1
fi

# Verify data has required features
if ! head -1 "$INPUT_FILE" | grep -q "consecutive_migraine_days"; then
    echo -e "${YELLOW}  ⚠ Warning: $INPUT_FILE may be missing temporal features${NC}"
    echo -e "${YELLOW}  Expected: consecutive_migraine_days, days_since_last_migraine${NC}"
fi

ROW_COUNT=$(wc -l < "$INPUT_FILE" | tr -d ' ')
echo -e "${GREEN}✓ Found input file: $INPUT_FILE ($ROW_COUNT lines)${NC}"
echo ""

# Step 4: Create output directory
echo -e "${YELLOW}[4/5] Preparing output directory...${NC}"
mkdir -p "$OUTPUT_DIR"
echo -e "${GREEN}✓ Output directory: $OUTPUT_DIR${NC}"
echo ""

# Step 5: Train the model
echo -e "${YELLOW}[5/5] Training XGBoost model...${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo -e "  Input file: $INPUT_FILE"
echo -e "  Output directory: $OUTPUT_DIR"
echo -e "  Hyperparameter trials: $TRIALS"
echo -e "  Random state: $RANDOM_STATE"
echo ""

python train_migraine_model.py \
    --input "$INPUT_FILE" \
    --output "$OUTPUT_DIR" \
    --trials "$TRIALS" \
    --random-state "$RANDOM_STATE"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "${BLUE}Model artifacts saved to: $OUTPUT_DIR/${NC}"
    echo -e "  - migraine_model.json (trained model)"
    echo -e "  - feature_engineer.pkl (preprocessing)"
    echo -e "  - feature_importance.csv (feature rankings)"
    echo -e "  - model_metadata.json (model info)"
    echo -e "  - model_evaluation_report.txt (metrics)"
    echo -e "  - roc_curve.png (ROC curve plot)"
    echo -e "  - calibration_curve.png (calibration plot)"
    echo -e "  - feature_importance.png (importance plot)"
    echo ""
    
    # Check if model files exist
    if [ -f "$OUTPUT_DIR/migraine_model.json" ]; then
        echo -e "${GREEN}✓ Model file created successfully${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: Model file not found${NC}"
    fi
    
    exit 0
else
    echo ""
    echo -e "${RED}========================================${NC}"
    echo -e "${RED}Training failed with exit code: $TRAIN_EXIT_CODE${NC}"
    echo -e "${RED}========================================${NC}"
    exit $TRAIN_EXIT_CODE
fi
