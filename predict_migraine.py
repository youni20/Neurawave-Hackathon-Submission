#!/usr/bin/env python3
"""
Migraine Prediction Script
Predicts migraine probability from user-reported features using a trained model.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
import xgboost as xgb


def load_model_artifacts(model_dir: str) -> Dict[str, Any]:
    """
    Load all model artifacts from directory.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        Dictionary with 'model', 'feature_engineer', 'feature_names', 'metadata'
    """
    model_path = Path(model_dir)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    artifacts = {}
    
    # Load XGBoost model
    model_file = model_path / "migraine_model.json"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    model = xgb.Booster()
    model.load_model(str(model_file))
    artifacts['model'] = model
    print(f"✓ Loaded model from {model_file}")
    
    # Load feature engineer
    fe_file = model_path / "feature_engineer.pkl"
    if not fe_file.exists():
        raise FileNotFoundError(f"Feature engineer file not found: {fe_file}")
    
    with open(fe_file, 'rb') as f:
        artifacts['feature_engineer'] = pickle.load(f)
    print(f"✓ Loaded feature engineer from {fe_file}")
    
    # Load feature names (optional)
    fn_file = model_path / "feature_names.json"
    if fn_file.exists():
        with open(fn_file, 'r') as f:
            artifacts['feature_names'] = json.load(f)
        print(f"✓ Loaded feature names from {fn_file}")
    
    # Load metadata (optional)
    metadata_file = model_path / "model_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r') as f:
            artifacts['metadata'] = json.load(f)
        print(f"✓ Loaded metadata from {metadata_file}")
    
    return artifacts


def prepare_features_from_json(user_data: Dict[str, Any], feature_engineer) -> pd.DataFrame:
    """
    Convert user JSON data to DataFrame and apply feature engineering.
    
    Args:
        user_data: Dictionary with user-reported features
        feature_engineer: Fitted FeatureEngineer instance
        
    Returns:
        Transformed features DataFrame ready for prediction
    """
    # Convert to DataFrame (single row)
    df = pd.DataFrame([user_data])
    
    # Apply feature engineering transformations
    X_processed = feature_engineer.transform(df)
    
    return X_processed


def predict_migraine_probability(
    model: xgb.Booster,
    X: pd.DataFrame,
    feature_names: Optional[list] = None
) -> float:
    """
    Predict migraine probability from features.
    
    Args:
        model: Trained XGBoost model
        X: Transformed features DataFrame
        feature_names: Optional list of feature names (for DMatrix)
        
    Returns:
        Migraine probability (0.0 to 1.0)
    """
    # Use feature names from DataFrame if not provided
    if feature_names is None:
        feature_names = X.columns.tolist()
    
    # Create DMatrix
    dtest = xgb.DMatrix(X, feature_names=feature_names)
    
    # Predict probability
    probability = model.predict(dtest)[0]
    
    return float(probability)


def main():
    """Main prediction function."""
    parser = argparse.ArgumentParser(
        description='Predict migraine probability from user features',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from JSON file
  python predict_migraine.py --model-dir models/ --input user_features.json
  
  # Predict from JSON file and save output
  python predict_migraine.py --model-dir models/ --input user_features.json --output prediction.json
        """
    )
    
    parser.add_argument(
        '--model-dir', '-m',
        type=str,
        required=True,
        help='Directory containing trained model files (migraine_model.json, feature_engineer.pkl)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='JSON file with user-reported features'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Optional: JSON file to save prediction result'
    )
    
    args = parser.parse_args()
    
    try:
        # Load model artifacts
        print("=" * 80)
        print("LOADING MODEL")
        print("=" * 80)
        artifacts = load_model_artifacts(args.model_dir)
        model = artifacts['model']
        feature_engineer = artifacts['feature_engineer']
        feature_names = artifacts.get('feature_names')
        metadata = artifacts.get('metadata', {})
        
        # Load user features from JSON
        print("\n" + "=" * 80)
        print("LOADING USER FEATURES")
        print("=" * 80)
        input_path = Path(args.input)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        
        with open(input_path, 'r') as f:
            user_data = json.load(f)
        
        print(f"✓ Loaded features from {input_path}")
        print(f"  Features provided: {list(user_data.keys())}")
        
        # Prepare features
        print("\n" + "=" * 80)
        print("PREPARING FEATURES")
        print("=" * 80)
        X_processed = prepare_features_from_json(user_data, feature_engineer)
        print(f"✓ Transformed to {len(X_processed.columns)} features")
        print(f"  Feature names: {list(X_processed.columns)}")
        
        # Make prediction
        print("\n" + "=" * 80)
        print("PREDICTION")
        print("=" * 80)
        probability = predict_migraine_probability(
            model,
            X_processed,
            feature_names=feature_names
        )
        
        # Format result
        result = {
            'migraine_probability': probability,
            'migraine_risk': 'High' if probability >= 0.5 else 'Low',
            'input_features': user_data,
            'model_info': {
                'model_type': metadata.get('model_type', 'XGBoost'),
                'training_timestamp': metadata.get('training_timestamp', 'Unknown')
            }
        }
        
        # Print result
        print(f"\nMigraine Probability: {probability:.4f} ({probability*100:.2f}%)")
        print(f"Risk Level: {result['migraine_risk']}")
        
        # Save output if requested
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n✓ Prediction saved to {output_path}")
        
        print("=" * 80)
        
        return result
        
    except Exception as e:
        print(f"\nError during prediction: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

