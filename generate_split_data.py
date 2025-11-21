#!/usr/bin/env python3
"""
Generate separate train/val/test datasets with distribution shift.
This helps detect overfitting by ensuring validation and test data come from
slightly different distributions than training data.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np

from data_generation.synthetic_data_generator import SyntheticDataGenerator
from data_generation.distribution_config import (
    DistributionParams,
    get_train_distribution_params,
    get_val_test_distribution_params
)
from migraine_model.data_split import stratified_split


def generate_splits(
    persons: int = 1000,
    days: int = 50,
    train_output: str = 'train_data.csv',
    val_output: str = 'val_data.csv',
    test_output: str = 'test_data.csv',
    person_data: Optional[str] = None,
    seed: int = 42,
    train_params: Optional[DistributionParams] = None,
    val_test_params: Optional[DistributionParams] = None
):
    """Core implementation for generating split datasets with optional custom params."""
    print("=" * 80)
    print("GENERATING TRAIN/VAL/TEST DATA WITH DISTRIBUTION SHIFT")
    print("=" * 80)
    print(f"Persons: {persons}")
    print(f"Days per person: {days}")
    print(f"Total records per set: {persons * days:,}")
    print(f"Random seed: {seed}")
    print("=" * 80)
    
    # Calculate split sizes
    train_persons = int(persons * 0.60)
    val_persons = int(persons * 0.20)
    test_persons = persons - train_persons - val_persons
    
    print(f"\nSplit sizes:")
    print(f"  Train: {train_persons} persons ({train_persons * days:,} records)")
    print(f"  Val:   {val_persons} persons ({val_persons * days:,} records)")
    print(f"  Test:  {test_persons} persons ({test_persons * days:,} records)")
    
    # Get distribution parameters
    train_params = train_params or get_train_distribution_params()
    val_test_params = val_test_params or get_val_test_distribution_params()
    
    print(f"\nDistribution parameters:")
    print(f"  Train: mood_noise={train_params.mood_noise_scale}, base_prob={train_params.base_migraine_probability}")
    print(f"  Val/Test: mood_noise={val_test_params.mood_noise_scale}, base_prob={val_test_params.base_migraine_probability}")
    
    # Generate training data with train distribution
    print(f"\n[1/3] Generating TRAINING data (train distribution)...")
    train_generator = SyntheticDataGenerator(
        person_data_file=person_data,
        random_state=seed,
        distribution_params=train_params
    )
    train_data = train_generator.generate_dataset(
        n_persons=train_persons,
        n_days=days,
        start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    )
    train_generator.save_dataset(train_data, train_output)
    print(f"  ✓ Saved {len(train_data):,} training records to {train_output}")
    if 'migraine' in train_data.columns:
        print(f"  Migraine rate: {train_data['migraine'].mean()*100:.2f}%")
    
    # Generate validation data with val/test distribution
    print(f"\n[2/3] Generating VALIDATION data (val/test distribution)...")
    val_generator = SyntheticDataGenerator(
        person_data_file=person_data,
        random_state=None,  # Use global random state
        distribution_params=val_test_params
    )
    val_data = val_generator.generate_dataset(
        n_persons=val_persons,
        n_days=days,
        start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=100)
    )
    val_generator.save_dataset(val_data, val_output)
    print(f"  ✓ Saved {len(val_data):,} validation records to {val_output}")
    if 'migraine' in val_data.columns:
        print(f"  Migraine rate: {val_data['migraine'].mean()*100:.2f}%")
    
    # Generate test data with val/test distribution
    print(f"\n[3/3] Generating TEST data (val/test distribution)...")
    test_generator = SyntheticDataGenerator(
        person_data_file=person_data,
        random_state=None,  # Use global random state
        distribution_params=val_test_params
    )
    test_data = test_generator.generate_dataset(
        n_persons=test_persons,
        n_days=days,
        start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=200)
    )
    test_generator.save_dataset(test_data, test_output)
    print(f"  ✓ Saved {len(test_data):,} test records to {test_output}")
    if 'migraine' in test_data.columns:
        print(f"  Migraine rate: {test_data['migraine'].mean()*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("DATA GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  Train: {train_output}")
    print(f"  Val:   {val_output}")
    print(f"  Test:  {test_output}")
    print("\nNote: Validation and test data use different distribution parameters")
    print("      to better detect overfitting and test generalization.")
    print("=" * 80)


def main():
    """CLI entrypoint to generate train/val/test datasets with distribution shift."""
    parser = argparse.ArgumentParser(
        description='Generate train/val/test datasets with distribution shift',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--persons', '-p', type=int, default=1000, help='Number of persons (default: 1000)')
    parser.add_argument('--days', '-d', type=int, default=50, help='Number of days per person (default: 50)')
    parser.add_argument('--train-output', type=str, default='train_data.csv', help='Training output file')
    parser.add_argument('--val-output', type=str, default='val_data.csv', help='Validation output file')
    parser.add_argument('--test-output', type=str, default='test_data.csv', help='Test output file')
    parser.add_argument('--person-data', type=str, default=None, help='Path to person_data CSV file (optional)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    generate_splits(
        persons=args.persons,
        days=args.days,
        train_output=args.train_output,
        val_output=args.val_output,
        test_output=args.test_output,
        person_data=args.person_data,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

