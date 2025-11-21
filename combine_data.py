#!/usr/bin/env python3
"""
Script to combine synthetic_migraine_data.csv with person_data to create combined_data.csv

SOLUTION 3: synthetic_migraine_data.csv now includes predictor features (stress, triggers, weather)
that were used to determine migraine probabilistically. We only need to add person data.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

def main():
    """Combine synthetic_migraine_data.csv with person_data into combined_data.csv"""
    print("=" * 80)
    print("Combining Data Sources (Solution 3 Architecture)")
    print("=" * 80)
    
    # File paths
    synthetic_file = 'synthetic_migraine_data.csv'
    person_file = 'person_data_100000.csv'
    output_file = 'combined_data.csv'
    
    # Check if files exist
    for file in [synthetic_file, person_file]:
        if not Path(file).exists():
            print(f"Error: {file} not found!", file=sys.stderr)
            sys.exit(1)
    
    print(f"\n[1/2] Loading synthetic_migraine_data.csv...")
    synthetic_df = pd.read_csv(synthetic_file)
    n_rows = len(synthetic_df)
    print(f"  Loaded {n_rows:,} rows")
    print(f"  Columns: {list(synthetic_df.columns)}")
    
    print(f"\n[2/2] Loading person_data_100000.csv...")
    person_df = pd.read_csv(person_file)
    # Sample n_rows from person_data (or use first n_rows)
    if len(person_df) >= n_rows:
        person_df = person_df.head(n_rows).copy()
    else:
        # If not enough, repeat rows
        repeat_times = (n_rows // len(person_df)) + 1
        person_df = pd.concat([person_df] * repeat_times, ignore_index=True).head(n_rows)
    
    # Extract only gender and migraine_days_per_month from person_data
    person_subset = person_df[['gender', 'migraine_days_per_month']].copy()
    print(f"  Extracted gender and migraine_days_per_month for {len(person_subset):,} rows")
    
    print(f"\n[3/3] Combining data sources...")
    # Combine person data with synthetic data
    # Note: synthetic_migraine_data.csv already contains:
    #   - Predictor features: stress_intensity, temp_mean, wind_mean, pressure_mean, 
    #     sun_irr_mean, sun_time_mean, precip_total, cloud_mean, stress, hormonal, 
    #     sleep, weather, food, sensory, physical
    #   - Symptom features: step_count_normalized, mood_score, mood_category, 
    #     screen_brightness_normalized
    #   - Target: migraine
    combined_df = pd.concat([person_subset, synthetic_df], axis=1)
    
    # Ensure column order: person data, predictor features, temporal features, symptom features, target
    expected_order = [
        # Person data
        'gender', 'migraine_days_per_month',
        # Predictor features (cause migraines)
        'stress_intensity', 'temp_mean', 'wind_mean', 'pressure_mean', 
        'sun_irr_mean', 'sun_time_mean', 'precip_total', 'cloud_mean',
        'stress', 'hormonal', 'sleep', 'weather', 'food', 'sensory', 'physical',
        # Temporal features (PHASE 1: momentum tracking)
        'consecutive_migraine_days', 'days_since_last_migraine',
        # Symptom features (result from migraines)
        'step_count_normalized', 'mood_score', 'mood_category', 'screen_brightness_normalized',
        # Target
        'migraine'
    ]
    
    # Reorder columns (only include columns that exist)
    available_cols = [col for col in expected_order if col in combined_df.columns]
    # Add any remaining columns that weren't in expected_order
    remaining_cols = [col for col in combined_df.columns if col not in available_cols]
    combined_df = combined_df[available_cols + remaining_cols]
    
    # Shuffle to mix data
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"  Combined dataset: {len(combined_df):,} rows × {len(combined_df.columns)} columns")
    
    # Save combined data
    print(f"\nSaving to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 80)
    print("Data Combination Complete!")
    print("=" * 80)
    print(f"\nOutput file: {output_file}")
    print(f"  Rows: {len(combined_df):,}")
    print(f"  Columns: {len(combined_df.columns)}")
    if 'migraine' in combined_df.columns:
        migraine_count = combined_df['migraine'].sum()
        migraine_pct = (migraine_count / len(combined_df)) * 100
        print(f"  Migraine occurrences: {migraine_count:,} ({migraine_pct:.2f}%)")
    print("=" * 80)

if __name__ == '__main__':
    main()

