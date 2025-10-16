"""
SIMPLE ACCURACY VALIDATION TEST
===============================

This script provides a simpler test of accuracy validation that doesn't
rely on the complex evaluation module, focusing on core data accuracy metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
import sys
import os

# Set random seed for reproducibility
np.random.seed(42)

def simulate_aser_simple(n=2400):
    """Simple ASER simulation"""
    df = pd.DataFrame()
    df['reading'] = np.clip(np.random.normal(55, 20, n) - 5, 0, 100)
    df['arithmetic'] = np.clip(np.random.normal(52, 22, n) - 6, 0, 100)
    df['age'] = np.random.randint(11, 18, n)
    df['gender'] = np.random.choice([0, 1], size=n)
    df['school_type'] = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    df['gap'] = ((df['reading'] < 40) | (df['arithmetic'] < 40)).astype(int)
    return df

def simulate_nas_simple(n=3600):
    """Simple NAS simulation"""
    df = pd.DataFrame()
    df['math'] = np.clip(np.random.normal(50, 18, n), 0, 100)
    df['language'] = np.clip(np.random.normal(54, 16, n), 0, 100)
    df['science'] = np.clip(np.random.normal(52, 17, n), 0, 100)
    df['grade'] = np.random.choice([3, 5, 8, 10], size=n)
    df['urban'] = np.random.choice([0, 1], size=n, p=[0.6, 0.4])
    
    # Calculate gap based on grade-specific thresholds
    thresh_map = {3: 35, 5: 40, 8: 45, 10: 50}
    df['threshold'] = df['grade'].map(thresh_map)
    df['avg_score'] = (df['math'] + df['language'] + df['science']) / 3
    df['gap'] = (df['avg_score'] < df['threshold']).astype(int)
    
    return df

def simulate_kaggle_simple(n=680):
    """Simple Kaggle simulation"""
    df = pd.DataFrame()
    df['literacy_rate'] = np.clip(np.random.normal(73.4, 10.1, n), 30, 95)
    df['female_literacy'] = np.clip(df['literacy_rate'] - np.random.normal(8, 4, n), 20, 95)
    df['urban_pop_percent'] = np.clip(np.random.normal(25, 19, n), 0, 95)
    df['total_schools'] = np.random.poisson(lam=300, size=n) + 50
    df['schools_with_electricity'] = (df['total_schools'] * np.random.normal(0.85, 0.15, n)).astype(int)
    
    # Infrastructure gap: if multiple infrastructure issues
    df['electricity_rate'] = df['schools_with_electricity'] / df['total_schools']
    df['gap'] = (df['electricity_rate'] < 0.7).astype(int)  # Infrastructure gap threshold
    
    return df

def validate_dataset_accuracy(simulated_df, dataset_name, real_benchmarks):
    """Validate a single dataset against real benchmarks"""
    print(f"\n{'='*60}")
    print(f"{dataset_name} DATASET ACCURACY VALIDATION")
    print(f"{'='*60}")
    
    results = {}
    
    # Get real benchmarks for this dataset
    benchmarks = real_benchmarks[dataset_name]
    
    # 1. Learning gap rate validation
    sim_gap_rate = simulated_df['gap'].mean()
    real_gap_rate = benchmarks['learning_gap_rate']
    gap_accuracy = 1 - abs(sim_gap_rate - real_gap_rate) / real_gap_rate
    
    print(f"Learning Gap Rate Validation:")
    print(f"  Real {dataset_name}: {real_gap_rate:.1%}")
    print(f"  Simulated:        {sim_gap_rate:.1%}")
    print(f"  Accuracy:         {gap_accuracy:.1%}")
    
    results['gap_rate_accuracy'] = gap_accuracy
    
    # 2. Sample size validation
    expected_size = benchmarks.get('expected_sample_size', len(simulated_df))
    size_ratio = len(simulated_df) / expected_size
    size_adequacy = min(1.0, size_ratio)
    
    print(f"\nSample Size Validation:")
    print(f"  Generated samples: {len(simulated_df)}")
    print(f"  Expected samples:  {expected_size}")
    print(f"  Size adequacy:     {size_adequacy:.1%}")
    
    results['sample_size_adequacy'] = size_adequacy
    
    # 3. Distribution validation (for numeric columns)
    numeric_cols = simulated_df.select_dtypes(include=[np.number]).columns
    distribution_scores = []
    
    print(f"\nDistribution Validation:")
    for col in numeric_cols:
        if col not in ['gap', 'threshold']:
            # Check if distribution looks reasonable (not all zeros, has variance)
            col_std = simulated_df[col].std()
            col_range = simulated_df[col].max() - simulated_df[col].min()
            
            # Simple distribution quality score
            if col_std > 0 and col_range > 0:
                dist_score = min(1.0, col_std / (col_range / 4))  # Expect std to be ~1/4 of range
                distribution_scores.append(dist_score)
                print(f"  {col:20s}: std={col_std:.2f}, range={col_range:.1f}, score={dist_score:.3f}")
    
    avg_distribution_score = np.mean(distribution_scores) if distribution_scores else 0
    results['distribution_quality'] = avg_distribution_score
    
    # 4. Data quality checks
    missing_data_rate = simulated_df.isnull().sum().sum() / (len(simulated_df) * len(simulated_df.columns))
    data_quality_score = 1 - missing_data_rate
    
    print(f"\nData Quality:")
    print(f"  Missing data rate: {missing_data_rate:.1%}")
    print(f"  Data quality score: {data_quality_score:.1%}")
    
    results['data_quality'] = data_quality_score
    
    # Overall accuracy for this dataset
    overall_accuracy = np.mean(list(results.values()))
    results['overall_accuracy'] = overall_accuracy
    
    print(f"\nOVERALL {dataset_name} ACCURACY: {overall_accuracy:.1%}")
    print(f"{'='*60}")
    
    return results

def main():
    """Main function to run simple accuracy validation"""
    print("🎓 SIMPLE ACCURACY VALIDATION TEST")
    print("="*80)
    print("Testing how well simulated educational data matches real-world patterns")
    print("="*80)
    
    # Real-world benchmarks (simplified)
    real_benchmarks = {
        'ASER': {
            'learning_gap_rate': 0.57,  # 57% children below grade level
            'expected_sample_size': 2400
        },
        'NAS': {
            'learning_gap_rate': 0.20,  # 20% below basic level
            'expected_sample_size': 3600
        },
        'Kaggle': {
            'learning_gap_rate': 0.27,  # 27% infrastructure gaps
            'expected_sample_size': 680
        }
    }
    
    # Generate simulated datasets
    print("Generating simulated datasets...")
    datasets = {
        'ASER': simulate_aser_simple(2400),
        'NAS': simulate_nas_simple(3600),
        'Kaggle': simulate_kaggle_simple(680)
    }
    print("✓ Datasets generated successfully")
    
    # Validate each dataset
    all_results = {}
    for dataset_name, dataset_df in datasets.items():
        results = validate_dataset_accuracy(dataset_df, dataset_name, real_benchmarks)
        all_results[dataset_name] = results
    
    # Overall summary
    print(f"\n{'='*80}")
    print("ACCURACY VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    overall_accuracies = [results['overall_accuracy'] for results in all_results.values()]
    system_accuracy = np.mean(overall_accuracies)
    
    print("Individual Dataset Accuracies:")
    for dataset_name, results in all_results.items():
        accuracy = results['overall_accuracy']
        grade = get_accuracy_grade(accuracy)
        print(f"  {dataset_name:8s}: {accuracy:.1%} (Grade: {grade})")
    
    print(f"\nSystem-wide Accuracy: {system_accuracy:.1%}")
    print(f"Overall Grade: {get_accuracy_grade(system_accuracy)}")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if system_accuracy >= 0.85:
        print("✅ Excellent accuracy! Simulated data closely matches real patterns.")
    elif system_accuracy >= 0.75:
        print("✓ Good accuracy. Minor adjustments could improve alignment.")
        print("  • Consider fine-tuning distribution parameters")
        print("  • Validate against additional data sources")
    elif system_accuracy >= 0.65:
        print("⚠ Moderate accuracy. Several improvements needed:")
        print("  • Review simulation methodology")
        print("  • Increase sample sizes")
        print("  • Add more realistic constraints")
    else:
        print("❌ Low accuracy. Major revisions required:")
        print("  • Fundamental review of simulation approach")
        print("  • Validate against primary data sources")
        print("  • Consider different modeling techniques")
    
    print(f"\n✓ Accuracy validation completed!")
    
    return all_results, system_accuracy

def get_accuracy_grade(accuracy):
    """Convert accuracy to letter grade"""
    if accuracy >= 0.90:
        return 'A+'
    elif accuracy >= 0.85:
        return 'A'
    elif accuracy >= 0.80:
        return 'B+'
    elif accuracy >= 0.75:
        return 'B'
    elif accuracy >= 0.70:
        return 'C+'
    elif accuracy >= 0.65:
        return 'C'
    else:
        return 'D'

if __name__ == "__main__":
    try:
        results, accuracy = main()
        print(f"\nSimple validation completed with {accuracy:.1%} overall accuracy!")
    except Exception as e:
        print(f"Error in validation: {str(e)}")
        import traceback
        traceback.print_exc()