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
import pandas as pd
import numpy as np

from data_generation.synthetic_data_generator import SyntheticDataGenerator
from data_generation.distribution_config import (
    get_train_distribution_params,
    get_val_test_distribution_params
)
from migraine_model.data_split import stratified_split
import subprocess
import itertools
import json
import time


def main():
    """Generate train/val/test datasets with distribution shift."""
    parser = argparse.ArgumentParser(
        description='Generate train/val/test datasets with distribution shift',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--persons', '-p',
        type=int,
        default=1000,
        help='Number of persons (default: 1000)'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=50,
        help='Number of days per person (default: 50)'
    )
    
    parser.add_argument(
        '--train-output',
        type=str,
        default='train_data.csv',
        help='Output file for training data (default: train_data.csv)'
    )
    
    parser.add_argument(
        '--val-output',
        type=str,
        default='val_data.csv',
        help='Output file for validation data (default: val_data.csv)'
    )
    
    parser.add_argument(
        '--test-output',
        type=str,
        default='test_data.csv',
        help='Output file for test data (default: test_data.csv)'
    )
    
    parser.add_argument(
        '--person-data',
        type=str,
        default=None,
        help='Path to person_data CSV file (optional)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--grid',
        action='store_true',
        help='Run grid search over generation hyperparameters (will run training for each combo)'
    )
    parser.add_argument(
        '--train-mood-noise-list',
        type=str,
        default=None,
        help='Comma-separated list of train mood_noise_scale values to try (e.g. 1.2,1.5,2.0)'
    )
    parser.add_argument(
        '--val-mood-noise-list',
        type=str,
        default=None,
        help='Comma-separated list of val/test mood_noise_scale values to try (e.g. 1.5,1.8)'
    )
    parser.add_argument(
        '--val-base-prob-list',
        type=str,
        default=None,
        help='Comma-separated list of val/test base_migraine_probability values to try (e.g. 0.09,0.11)'
    )
    parser.add_argument(
        '--grid-trials',
        type=int,
        default=20,
        help='Number of hyperparameter optimization trials to pass to training during grid search (default: 20)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GENERATING TRAIN/VAL/TEST DATA WITH DISTRIBUTION SHIFT")
    print("=" * 80)
    print(f"Persons: {args.persons}")
    print(f"Days per person: {args.days}")
    print(f"Total records per set: {args.persons * args.days:,}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)
    
    # Calculate split sizes
    # OPTION 3: Generate more data for val/test to ensure proper migraine distribution
    # Train: 60%, Val: 20%, Test: 20% (increased from 70/15/15)
    train_persons = int(args.persons * 0.60)
    val_persons = int(args.persons * 0.20)
    test_persons = args.persons - train_persons - val_persons
    
    print(f"\nSplit sizes:")
    print(f"  Train: {train_persons} persons ({train_persons * args.days:,} records)")
    print(f"  Val:   {val_persons} persons ({val_persons * args.days:,} records)")
    print(f"  Test:  {test_persons} persons ({test_persons * args.days:,} records)")
    
    # Get distribution parameters
    train_params = get_train_distribution_params()
    val_test_params = get_val_test_distribution_params()

    def _parse_list(src: str):
        if not src:
            return None
        try:
            return [float(x.strip()) for x in src.split(',') if x.strip() != '']
        except Exception:
            return None

    train_mood_list = _parse_list(args.train_mood_noise_list)
    val_mood_list = _parse_list(args.val_mood_noise_list)
    val_base_prob_list = _parse_list(args.val_base_prob_list)
    
    print(f"\nDistribution parameters:")
    print(f"  Train: mood_noise={train_params.mood_noise_scale}, base_prob={train_params.base_migraine_probability}")
    print(f"  Val/Test: mood_noise={val_test_params.mood_noise_scale}, base_prob={val_test_params.base_migraine_probability}")
    # If grid search requested, iterate over parameter combinations
    if args.grid:
        print("\nRunning generation grid search...")
        # Default lists: use current params if lists not provided
        if train_mood_list is None:
            train_mood_list = [train_params.mood_noise_scale]
        if val_mood_list is None:
            val_mood_list = [val_test_params.mood_noise_scale]
        if val_base_prob_list is None:
            val_base_prob_list = [val_test_params.base_migraine_probability]

        combos = list(itertools.product(train_mood_list, val_mood_list, val_base_prob_list))
        print(f"Trying {len(combos)} combinations")

        results = []
        for idx, (t_mood, v_mood, v_base) in enumerate(combos, start=1):
            run_tag = f"grid_{idx:03d}"
            print(f"\n[{run_tag}] t_mood={t_mood}, v_mood={v_mood}, v_base={v_base}")

            # Prepare params for this combo (shallow copy)
            t_params = get_train_distribution_params()
            vt_params = get_val_test_distribution_params()
            t_params.mood_noise_scale = float(t_mood)
            vt_params.mood_noise_scale = float(v_mood)
            vt_params.base_migraine_probability = float(v_base)

            # Seed per-run for reproducibility but let generator manage internal randomness
            np.random.seed(args.seed + idx)

            # Generate datasets
            train_gen = SyntheticDataGenerator(person_data_file=args.person_data, random_state=None, distribution_params=t_params)
            train_df = train_gen.generate_dataset(
                n_persons=train_persons,
                n_days=args.days,
                start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            )
            train_path = f"{Path(args.train_output).stem}_{run_tag}.csv"
            train_gen.save_dataset(train_df, train_path)

            val_gen = SyntheticDataGenerator(person_data_file=args.person_data, random_state=None, distribution_params=vt_params)
            val_df = val_gen.generate_dataset(
                n_persons=val_persons,
                n_days=args.days,
                start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=100)
            )
            val_path = f"{Path(args.val_output).stem}_{run_tag}.csv"
            val_gen.save_dataset(val_df, val_path)

            test_gen = SyntheticDataGenerator(person_data_file=args.person_data, random_state=None, distribution_params=vt_params)
            test_df = test_gen.generate_dataset(
                n_persons=test_persons,
                n_days=args.days,
                start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=200)
            )
            test_path = f"{Path(args.test_output).stem}_{run_tag}.csv"
            test_gen.save_dataset(test_df, test_path)

            # Combine into single CSV for training script
            combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            combined_path = f"combined_{run_tag}.csv"
            combined_df.to_csv(combined_path, index=False)

            # Run training script on combined CSV
            out_dir = f"models/{run_tag}"
            cmd = [sys.executable, 'train_migraine_model.py', '--input', combined_path, '--output', out_dir, '--trials', str(args.grid_trials), '--random-state', str(args.seed)]
            print(f"Running training: {' '.join(cmd)}")
            start = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            duration = time.time() - start

            stdout = proc.stdout or ''
            stderr = proc.stderr or ''

            # Parse key metrics from stdout (fallback to reading saved evaluation report if needed)
            metrics = {'run': run_tag, 't_mood': t_mood, 'v_mood': v_mood, 'v_base': v_base, 'duration_s': duration}
            try:
                # Look for Final Test Set Performance block
                if 'Final Test Set Performance:' in stdout:
                    lines = stdout.splitlines()
                    for i, line in enumerate(lines):
                        if 'Final Test Set Performance:' in line:
                            # Parse next 4 lines for ROC-AUC, F1-Score, Precision, Recall
                            for j in range(1, 5):
                                if i + j < len(lines):
                                    l = lines[i + j].strip()
                                    if l.startswith('ROC-AUC:'):
                                        metrics['roc_auc'] = float(l.split(':')[1].strip())
                                    elif l.startswith('F1-Score:'):
                                        metrics['f1_score'] = float(l.split(':')[1].strip())
                                    elif l.startswith('Precision:'):
                                        metrics['precision'] = float(l.split(':')[1].strip())
                                    elif l.startswith('Recall:'):
                                        metrics['recall'] = float(l.split(':')[1].strip())
                            break
            except Exception as e:
                metrics['parse_error'] = str(e)

            # If parsing failed, attempt to read models/<run_tag>/model_evaluation_report.txt
            if 'roc_auc' not in metrics:
                report_path = Path(out_dir) / 'model_evaluation_report.txt'
                if report_path.exists():
                    try:
                        report_text = report_path.read_text()
                        for line in report_text.splitlines():
                            if line.strip().startswith('ROC-AUC:'):
                                metrics['roc_auc'] = float(line.split(':')[1].strip())
                            if line.strip().startswith('F1-Score:'):
                                metrics['f1_score'] = float(line.split(':')[1].strip())
                            if line.strip().startswith('Precision:'):
                                metrics['precision'] = float(line.split(':')[1].strip())
                            if line.strip().startswith('Recall:'):
                                metrics['recall'] = float(line.split(':')[1].strip())
                    except Exception:
                        pass

            metrics['stdout'] = stdout[:2000]
            metrics['stderr'] = stderr[:2000]
            results.append(metrics)

            # Save intermediate results
            results_df = pd.DataFrame(results)
            results_df.to_csv('generation_grid_search_results.csv', index=False)

        print('\nGrid search complete. Results saved to generation_grid_search_results.csv')
        return

    # Generate training data with train distribution
    print(f"\n[1/3] Generating TRAINING data (train distribution)...")
    train_generator = SyntheticDataGenerator(
        person_data_file=args.person_data,
        random_state=args.seed,
        distribution_params=train_params
    )
    train_data = train_generator.generate_dataset(
        n_persons=train_persons,
        n_days=args.days,
        start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    )
    train_generator.save_dataset(train_data, args.train_output)
    print(f"  ✓ Saved {len(train_data):,} training records to {args.train_output}")
    if 'migraine' in train_data.columns:
        print(f"  Migraine rate: {train_data['migraine'].mean()*100:.2f}%")

    # Generate validation data with val/test distribution
    print(f"\n[2/3] Generating VALIDATION data (val/test distribution)...")
    # OPTION 2: Don't seed val/test generators - let them use global random state
    # This prevents deterministic behavior that causes 0% migraine rate
    # The global state is already seeded in train generator, so we continue from there
    val_generator = SyntheticDataGenerator(
        person_data_file=args.person_data,
        random_state=None,  # OPTION 2: Use global random state (not fixed seed)
        distribution_params=val_test_params
    )
    val_data = val_generator.generate_dataset(
        n_persons=val_persons,
        n_days=args.days,
        start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=100)
    )
    val_generator.save_dataset(val_data, args.val_output)
    print(f"  ✓ Saved {len(val_data):,} validation records to {args.val_output}")
    if 'migraine' in val_data.columns:
        print(f"  Migraine rate: {val_data['migraine'].mean()*100:.2f}%")

    # Generate test data with val/test distribution
    print(f"\n[3/3] Generating TEST data (val/test distribution)...")
    # OPTION 2: Don't seed val/test generators - let them use global random state
    # This prevents deterministic behavior that causes 0% migraine rate
    test_generator = SyntheticDataGenerator(
        person_data_file=args.person_data,
        random_state=None,  # OPTION 2: Use global random state (not fixed seed)
        distribution_params=val_test_params
    )
    test_data = test_generator.generate_dataset(
        n_persons=test_persons,
        n_days=args.days,
        start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=200)
    )
    test_generator.save_dataset(test_data, args.test_output)
    print(f"  ✓ Saved {len(test_data):,} test records to {args.test_output}")
    if 'migraine' in test_data.columns:
        print(f"  Migraine rate: {test_data['migraine'].mean()*100:.2f}%")
    
    print("\n" + "=" * 80)
    print("DATA GENERATION COMPLETE!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  Train: {args.train_output}")
    print(f"  Val:   {args.val_output}")
    print(f"  Test:  {args.test_output}")
    print("\nNote: Validation and test data use different distribution parameters")
    print("      to better detect overfitting and test generalization.")
    print("=" * 80)


if __name__ == '__main__':
    from datetime import timedelta
    main()

