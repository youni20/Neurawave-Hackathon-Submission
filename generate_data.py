#!/usr/bin/env python3
"""
CLI script for generating synthetic migraine prediction datasets.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from data_generation.synthetic_data_generator import SyntheticDataGenerator


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description='Generate synthetic migraine prediction datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate dataset with 100 persons, 90 days each
  python generate_data.py --persons 100 --days 90 --output data.csv
  
  # Generate dataset using existing person data
  python generate_data.py --persons 1000 --days 365 --person-data person_data_100000.csv --output data.csv
  
  # Generate dataset with custom start date and random seed
  python generate_data.py --persons 50 --days 30 --start-date 2024-01-01 --seed 42 --output data.csv
        """
    )
    
    parser.add_argument(
        '--persons', '-p',
        type=int,
        default=100,
        help='Number of persons to generate (default: 100)'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=90,
        help='Number of days per person (default: 90)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='synthetic_migraine_data.csv',
        help='Output CSV file path (default: synthetic_migraine_data.csv)'
    )
    
    parser.add_argument(
        '--person-data',
        type=str,
        default=None,
        help='Path to existing person_data CSV file (optional)'
    )
    
    parser.add_argument(
        '--questionnaire',
        type=str,
        default=None,
        help='Path to questionnaire CSV file (optional)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date in YYYY-MM-DD format (default: today)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.persons < 1:
        print("Error: Number of persons must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    if args.days < 1:
        print("Error: Number of days must be at least 1", file=sys.stderr)
        sys.exit(1)
    
    # Parse start date
    start_date = None
    if args.start_date:
        try:
            start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
        except ValueError:
            print(f"Error: Invalid date format '{args.start_date}'. Use YYYY-MM-DD", file=sys.stderr)
            sys.exit(1)
    
    # Check if person data file exists
    if args.person_data and not Path(args.person_data).exists():
        print(f"Warning: Person data file '{args.person_data}' not found. Will generate synthetic person data.", file=sys.stderr)
        args.person_data = None
    
    # Check if questionnaire file exists
    if args.questionnaire and not Path(args.questionnaire).exists():
        print(f"Warning: Questionnaire file '{args.questionnaire}' not found. Will use default patterns.", file=sys.stderr)
        args.questionnaire = None
    
    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Synthetic Migraine Data Generator")
    print("=" * 60)
    print(f"Persons: {args.persons}")
    print(f"Days per person: {args.days}")
    print(f"Total records: {args.persons * args.days:,}")
    print(f"Output file: {args.output}")
    if args.person_data:
        print(f"Person data: {args.person_data}")
    if args.questionnaire:
        print(f"Questionnaire: {args.questionnaire}")
    if args.seed:
        print(f"Random seed: {args.seed}")
    print("=" * 60)
    print()
    
    try:
        # Initialize generator
        generator = SyntheticDataGenerator(
            person_data_file=args.person_data,
            questionnaire_file=args.questionnaire,
            random_state=args.seed
        )
        
        # Generate dataset
        print("Generating dataset...")
        dataset = generator.generate_dataset(
            n_persons=args.persons,
            n_days=args.days,
            start_date=start_date
        )
        
        # Save dataset
        print(f"\nSaving dataset to {args.output}...")
        generator.save_dataset(dataset, args.output)
        
        # Print summary statistics
        print("\n" + "=" * 60)
        print("Dataset Summary")
        print("=" * 60)
        print(f"Total records: {len(dataset):,}")
        if 'migraine' in dataset.columns:
            print(f"Migraine occurrences: {dataset['migraine'].sum():,} ({dataset['migraine'].mean()*100:.2f}%)")
        print("\nFeature Statistics:")
        if 'step_count' in dataset.columns:
            print(f"  Step count: mean={dataset['step_count'].mean():.0f}, std={dataset['step_count'].std():.0f}")
        if 'mood_score' in dataset.columns:
            print(f"  Mood score: mean={dataset['mood_score'].mean():.2f}, std={dataset['mood_score'].std():.2f}")
        if 'screen_brightness' in dataset.columns:
            print(f"  Screen brightness: mean={dataset['screen_brightness'].mean():.1f}, std={dataset['screen_brightness'].std():.1f}")
        print("=" * 60)
        print("\nDataset generation completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

