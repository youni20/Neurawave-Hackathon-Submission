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
import subprocess
import itertools
import json
import time
import optuna

from data_generation.synthetic_data_generator import SyntheticDataGenerator
from data_generation.distribution_config import (
    get_train_distribution_params,
    get_val_test_distribution_params
)
from migraine_model.data_split import stratified_split


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
        help='Run focused grid search over a small set of generation parameters'
    )
    parser.add_argument(
        '--bayes',
        action='store_true',
        help='Run Bayesian optimization (Optuna) over generation parameters'
    )
    parser.add_argument(
        '--bayes-trials',
        type=int,
        default=30,
        help='Number of Optuna trials for Bayesian optimization'
    )
    parser.add_argument(
        '--grid-trials',
        type=int,
        default=20,
        help='When running grid, number of training trials to pass to the training script'
    )
    parser.add_argument(
        '--train-mood-noise-list',
        type=str,
        default='1.2,1.5',
        help='Comma-separated list of train mood_noise_scale values to try when --grid is used'
    )
    parser.add_argument(
        '--val-mood-noise-list',
        type=str,
        default='1.6,1.8',
        help='Comma-separated list of val mood_noise_scale values to try when --grid is used'
    )
    parser.add_argument(
        '--val-base-prob-list',
        type=str,
        default='0.09,0.11',
        help='Comma-separated list of val base_migraine_probability values to try when --grid is used'
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
    
    print(f"\nDistribution parameters:")
    print(f"  Train: mood_noise={train_params.mood_noise_scale}, base_prob={train_params.base_migraine_probability}")
    print(f"  Val/Test: mood_noise={val_test_params.mood_noise_scale}, base_prob={val_test_params.base_migraine_probability}")

    def _parse_list(src: str):
        try:
            return [float(x.strip()) for x in src.split(',') if x.strip() != '']
        except Exception:
            return None

    train_mood_list = _parse_list(args.train_mood_noise_list) if hasattr(args, 'train_mood_noise_list') else None
    val_mood_list = _parse_list(args.val_mood_noise_list) if hasattr(args, 'val_mood_noise_list') else None
    val_base_prob_list = _parse_list(args.val_base_prob_list) if hasattr(args, 'val_base_prob_list') else None

    # Helper to run training and parse saved evaluation report
    def _run_training_on_combined(combined_path: str, out_dir: str, trials: int, seed: int):
        cmd = [sys.executable, 'train_migraine_model.py', '--input', combined_path, '--output', out_dir, '--trials', str(trials), '--random-state', str(seed)]
        print(f"Running training: {' '.join(cmd)}")
        start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        duration = time.time() - start
        stdout = proc.stdout or ''
        stderr = proc.stderr or ''

        # Try to read evaluation report
        metrics = {}
        report_path = Path(out_dir) / 'model_evaluation_report.txt'
        if report_path.exists():
            try:
                text = report_path.read_text()
                # parse training and validation sections
                def _parse_section(section_name: str):
                    res = {}
                    idx = text.find(section_name)
                    if idx == -1:
                        return res
                    sec = text[idx:idx+500]
                    for line in sec.splitlines():
                        if 'ROC-AUC' in line:
                            res['roc_auc'] = float(line.split(':')[1].strip())
                        if 'Precision:' in line and 'Precision:' in line:
                            # take first precision occurrence in section
                            res['precision'] = float(line.split(':')[1].strip())
                        if 'Accuracy:' in line:
                            res['accuracy'] = float(line.split(':')[1].strip())
                    return res

                train_sec = _parse_section('Training Set Metrics:')
                val_sec = _parse_section('Validation Set Metrics:')
                metrics['train_roc_auc'] = train_sec.get('roc_auc')
                metrics['train_precision'] = train_sec.get('precision')
                metrics['train_accuracy'] = train_sec.get('accuracy')
                metrics['val_roc_auc'] = val_sec.get('roc_auc')
                metrics['val_precision'] = val_sec.get('precision')
                metrics['val_accuracy'] = val_sec.get('accuracy')
            except Exception as e:
                print(f"Warning: failed to parse report {report_path}: {e}")

        # Fallback: try to parse stdout for final metrics if report not found
        if 'val_roc_auc' not in metrics or metrics.get('val_roc_auc') is None:
            if 'Final Test Set Performance:' in stdout:
                for i, line in enumerate(stdout.splitlines()):
                    if 'Final Test Set Performance:' in line:
                        # the block contains ROC-AUC, F1-Score, Precision, Recall
                        for j in range(1, 6):
                            if i + j < len(stdout.splitlines()):
                                l = stdout.splitlines()[i + j].strip()
                                if l.startswith('ROC-AUC:'):
                                    metrics['val_roc_auc'] = float(l.split(':')[1].strip())
                                if l.startswith('Precision:'):
                                    metrics['val_precision'] = float(l.split(':')[1].strip())
                        break

        metrics['stdout'] = stdout[:2000]
        metrics['stderr'] = stderr[:2000]
        metrics['duration_s'] = duration
        return metrics

    # Bayesian optimization over generation params
    if args.bayes:
        print("\nRunning Bayesian optimization over generation parameters...")
        results = []

        def objective(trial: optuna.trial.Trial):
            # Suggest a compact set of generation parameters that most affect overfitting
            t_mood = trial.suggest_float('train_mood_noise', 0.5, 4.0)
            v_mood = trial.suggest_float('val_mood_noise', 0.5, 4.5)
            v_base = trial.suggest_float('val_base_prob', 0.02, 0.20)
            t_step = trial.suggest_float('train_step_noise', 0.05, 0.4)
            v_step = trial.suggest_float('val_step_noise', 0.05, 0.4)
            migraine_mood_min = trial.suggest_float('migraine_mood_factor_min', 0.2, 0.6)

            # Prepare distribution params for this trial
            t_params = get_train_distribution_params()
            vt_params = get_val_test_distribution_params()
            t_params.mood_noise_scale = float(t_mood)
            vt_params.mood_noise_scale = float(v_mood)
            vt_params.base_migraine_probability = float(v_base)
            t_params.step_count_noise_scale = float(t_step)
            vt_params.step_count_noise_scale = float(v_step)
            t_params.migraine_mood_factor_min = float(migraine_mood_min)
            vt_params.migraine_mood_factor_min = float(migraine_mood_min)

            # Generate datasets for trial
            trial_tag = f"bayes_{trial.number:04d}"
            np.random.seed(args.seed + trial.number)

            train_gen = SyntheticDataGenerator(person_data_file=args.person_data, random_state=None, distribution_params=t_params)
            train_df = train_gen.generate_dataset(n_persons=train_persons, n_days=args.days, start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0))
            val_gen = SyntheticDataGenerator(person_data_file=args.person_data, random_state=None, distribution_params=vt_params)
            val_df = val_gen.generate_dataset(n_persons=val_persons, n_days=args.days, start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=100))
            test_gen = SyntheticDataGenerator(person_data_file=args.person_data, random_state=None, distribution_params=vt_params)
            test_df = test_gen.generate_dataset(n_persons=test_persons, n_days=args.days, start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=200))

            combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
            combined_path = f"combined_{trial_tag}.csv"
            combined_df.to_csv(combined_path, index=False)

            out_dir = f"models/{trial_tag}"
            metrics = _run_training_on_combined(combined_path, out_dir, args.bayes_trials, args.seed)

            # Compute objective: penalize train-val accuracy gap and low val ROC-AUC
            train_acc = metrics.get('train_accuracy') or 1.0
            val_acc = metrics.get('val_accuracy') or 0.0
            val_auc = metrics.get('val_roc_auc') or 0.0
            gap = max(0.0, float(train_acc) - float(val_acc))
            score = gap + (1.0 - float(val_auc))

            # Record trial result
            rec = {
                'trial': trial.number,
                'train_mood': t_mood,
                'val_mood': v_mood,
                'val_base_prob': v_base,
                'train_step_noise': t_step,
                'val_step_noise': v_step,
                'migraine_mood_min': migraine_mood_min,
                'train_accuracy': train_acc,
                'val_accuracy': val_acc,
                'val_roc_auc': val_auc,
                'objective': score
            }
            results.append(rec)
            pd.DataFrame(results).to_csv('generation_bayes_results.csv', index=False)

            # Report to optuna
            return float(score)

        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=args.seed))
        study.optimize(objective, n_trials=args.bayes_trials)

        print('\nBayesian optimization complete.')
        print(f"Best trial: #{study.best_trial.number} -> params: {study.best_trial.params} value={study.best_value}")
        # Save study results
        df_study = optuna.trial._study_direction_to_dataframe(study)
        try:
            # optuna API compatibility: try to save as csv via study.trials_dataframe
            df_trials = study.trials_dataframe()
            df_trials.to_csv('generation_bayes_trials.csv', index=False)
        except Exception:
            pass
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

