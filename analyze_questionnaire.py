#!/usr/bin/env python3
"""
Script to analyze questionnaire data using ML classifier and display statistics.
"""

import argparse
import sys
from pathlib import Path

from data_generation.ml_trigger_classifier import MLTriggerClassifier


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Analyze questionnaire data with ML classifier and display statistics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze questionnaire data and display statistics
  python analyze_questionnaire.py migraine_symptom_classification.csv
  
  # Train models and show feature importance
  python analyze_questionnaire.py migraine_symptom_classification.csv --train
  
  # Process dataset and save results
  python analyze_questionnaire.py migraine_symptom_classification.csv --train --output results.csv
        """
    )
    
    parser.add_argument(
        'questionnaire_file',
        type=str,
        help='Path to questionnaire CSV file'
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='Train ML models for trigger classification'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output CSV file for processed data (optional)'
    )
    
    parser.add_argument(
        '--feature-importance',
        type=int,
        default=10,
        help='Number of top features to show for importance (default: 10)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    if not Path(args.questionnaire_file).exists():
        print(f"Error: Questionnaire file '{args.questionnaire_file}' not found.", file=sys.stderr)
        sys.exit(1)
    
    print("=" * 80)
    print("Questionnaire Data Analysis with ML Classifier")
    print("=" * 80)
    print(f"Questionnaire file: {args.questionnaire_file}")
    print()
    
    try:
        # Initialize classifier
        classifier = MLTriggerClassifier(args.questionnaire_file)
        
        # Print statistics
        classifier.print_statistics()
        
        # Train models if requested
        if args.train:
            print("\n" + "=" * 80)
            print("Training ML Models")
            print("=" * 80)
            classifier.train_models()
            
            # Print feature importance
            classifier.print_feature_importance(top_n=args.feature_importance)
        
        # Process dataset if output is specified
        if args.output:
            print("\n" + "=" * 80)
            print("Processing Dataset")
            print("=" * 80)
            result_df = classifier.process_dataset()
            
            if len(result_df) > 0:
                result_df.to_csv(args.output, index=False)
                print(f"\nProcessed dataset saved to {args.output}")
                print(f"Dataset shape: {result_df.shape}")
                print(f"Added trigger columns: {classifier.TRIGGER_CATEGORIES}")
        
        print("\n" + "=" * 80)
        print("Analysis completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

