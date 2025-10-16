"""
DATA SIMULATION METHODOLOGY
===========================

This file documents how simulated data was created to match real ASER, NAS, and Kaggle datasets.
Each simulation function is designed to replicate the statistical properties and distributions 
found in the actual Indian education datasets.

Sources:
- ASER 2024: https://asercentre.org/wp-content/uploads/2022/12/ASER_2024_Final-Report_13_2_24.pdf
- NAS 2021: https://nas.gov.in/download-national-report  
- Kaggle: https://www.kaggle.com/datasets/rajanand/education-in-india?resource=download
"""

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# ================================================================================
# 1. ASER DATASET SIMULATION
# ================================================================================

def simulate_aser(n=2400):
    """
    Simulates ASER (Annual Status of Education Report) dataset based on real 2024 data.
    
    REAL ASER 2024 CHARACTERISTICS:
    - Rural focus with 80% government schools
    - Reading levels: Letters → Words → Paragraphs → Stories  
    - Arithmetic levels: Nothing → Numbers → Subtraction → Division
    - ~57% children below expected grade level
    - Age range: 5-16 years (focus on adolescents 11-17)
    - Strong emphasis on basic literacy and numeracy
    
    SIMULATION STRATEGY:
    - Use normal distributions with rural bias (-5/-6 points)
    - School type reflects rural government school dominance (85%)
    - Gap definition based on 40% threshold (matches ASER criteria)
    - Text responses correlate with performance levels
    """
    df = pd.DataFrame()
    
    # READING SCORES: Normal distribution biased lower for rural challenges
    # Real ASER shows significant rural learning gaps
    df['reading'] = np.clip(
        np.random.normal(55, 20, n) - 5,  # -5 bias for rural context
        0, 100
    )
    
    # ARITHMETIC SCORES: Slightly lower than reading (matches ASER pattern)
    # Math typically more challenging in rural areas
    df['arithmetic'] = np.clip(
        np.random.normal(52, 22, n) - 6,  # -6 bias for math difficulty
        0, 100
    )
    
    # AGE DISTRIBUTION: Focus on adolescent learners (11-17 years)
    # ASER covers broader range but this focuses on key transition ages
    df['age'] = np.random.randint(11, 18, n)
    
    # GENDER: Balanced distribution (real ASER shows near parity)
    df['gender'] = np.random.choice([0, 1], size=n)  # 0=male, 1=female
    
    # SCHOOL TYPE: 85% government schools (matches rural ASER context)
    # Real ASER 2024 shows ~80% government school enrollment in rural areas
    df['school_type'] = np.random.choice([0, 1], size=n, p=[0.85, 0.15])  # 0=govt, 1=private
    
    # TEXT RESPONSES: Synthetic responses based on performance
    # Higher scores → more "understands", lower scores → more "struggles"
    df['text'] = [
        ' '.join(
            ['understands'] * int(max(1, (r/25))) + 
            ['struggles'] * int(max(1, (100-r)/25))
        ) 
        for r in df['reading'].round().astype(int)
    ]
    
    # GAP LABEL: Learning gap if either subject below 40% (matches ASER threshold)
    # Real ASER defines learning gaps based on grade-appropriate skill levels
    df['gap'] = ((df['reading'] < 40) | (df['arithmetic'] < 40)).astype(int)
    
    print(f"ASER Simulation: {len(df)} students, {df['gap'].mean():.1%} gap rate")
    return df


# ================================================================================
# 2. NAS DATASET SIMULATION  
# ================================================================================

def simulate_nas(n=3600):
    """
    Simulates NAS (National Achievement Survey) dataset based on real 2021 data.
    
    REAL NAS 2021 CHARACTERISTICS:
    - National coverage across all states
    - Grades 3, 5, 8, 10 assessment
    - Subjects: Mathematics, Language, Science
    - Performance levels: Below Basic, Basic, Proficient, Advanced
    - ~20% students below basic level
    - Urban-rural split: ~65% rural, 35% urban
    - School type: ~75% government, 25% private
    
    SIMULATION STRATEGY:
    - Multi-subject normal distributions (Math, Language, Science)
    - Grade-specific performance thresholds
    - Balanced grade distribution (25% each grade)
    - Realistic urban-rural demographics (60% rural)
    - Teacher quality as additional factor
    """
    df = pd.DataFrame()
    
    # MATHEMATICS SCORES: Slightly below average (50th percentile baseline)
    # Real NAS shows math as challenging subject
    df['math'] = np.clip(np.random.normal(50, 18, n), 0, 100)
    
    # LANGUAGE SCORES: Slightly higher than math (real pattern)
    # Language typically performs better than math in NAS
    df['language'] = np.clip(np.random.normal(54, 16, n), 0, 100)
    
    # SCIENCE SCORES: Intermediate between math and language
    df['science'] = np.clip(np.random.normal(52, 17, n), 0, 100)
    
    # GRADE DISTRIBUTION: Equal representation across NAS target grades
    # Real NAS assesses these specific grades nationally
    df['grade'] = np.random.choice([3, 5, 8, 10], size=n, p=[0.25, 0.25, 0.25, 0.25])
    
    # URBAN-RURAL SPLIT: 60% rural (matches Indian demographic pattern)
    # Real NAS shows rural-urban performance gaps
    df['urban'] = np.random.choice([0, 1], size=n, p=[0.6, 0.4])  # 0=rural, 1=urban
    
    # TEACHER QUALIFICATION: 70% qualified (matches real teacher data)
    # Important factor in NAS performance analysis
    df['teacher_qual'] = np.random.choice([0, 1], size=n, p=[0.3, 0.7])  # 1=better qualified
    
    # TEXT SUMMARY: Weighted composite score as text
    # Simulates student performance summaries
    df['text'] = (
        df['math'] * 0.4 + 
        df['language'] * 0.3 + 
        df['science'] * 0.3
    ).round().astype(int).astype(str)
    
    # GAP DEFINITION: Grade-specific thresholds (higher expectations for higher grades)
    # Real NAS uses competency-based assessment with grade-appropriate expectations
    thresh = df['grade'].map({3: 35, 5: 40, 8: 45, 10: 50})
    df['avg'] = (df['math'] + df['language'] + df['science']) / 3.0
    df['gap'] = (df['avg'] < thresh).astype(int)
    df.drop(columns=['avg'], inplace=True)
    
    print(f"NAS Simulation: {len(df)} students, {df['gap'].mean():.1%} gap rate")
    return df


# ================================================================================
# 3. KAGGLE DISTRICT DATASET SIMULATION
# ================================================================================

def simulate_kaggle(n=680):
    """
    Simulates Kaggle Education in India dataset based on real 2015-16 district data.
    
    REAL KAGGLE 2015-16 CHARACTERISTICS:
    - 680+ districts across major Indian states
    - District-level infrastructure and demographic data
    - Variables: Population, literacy, schools, teachers, facilities
    - Government school dominance (~75%)
    - Wide variation in literacy rates (45-95%)
    - Infrastructure gaps in rural/tribal districts
    
    SIMULATION STRATEGY:
    - Lognormal population distribution (realistic district size variation)
    - State-wise proportional representation
    - Correlated infrastructure variables
    - Realistic facility ratios based on government data
    - Composite infrastructure gap score
    """
    df = pd.DataFrame()
    
    # BASIC DISTRICT INFO
    df['district_id'] = range(1, n+1)
    
    # STATE DISTRIBUTION: Proportional to real Indian state populations
    # UP has most districts, followed by Bihar, Maharashtra, etc.
    df['state'] = np.random.choice(
        ['UP', 'Bihar', 'Maharashtra', 'WB', 'MP', 'Rajasthan', 'Karnataka', 'Gujarat', 'AP', 'TN'],
        size=n, 
        p=[0.15, 0.12, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.10, 0.10]
    )
    
    # DEMOGRAPHICS: Lognormal distribution for realistic district size variation
    # Real districts range from small tribal areas to large urban districts
    df['total_population'] = np.random.lognormal(mean=12, sigma=0.8, size=n).astype(int)
    
    # URBAN POPULATION: Normal with realistic range (0-95%)
    # Most districts are predominantly rural
    df['urban_pop_percent'] = np.clip(np.random.normal(25, 19, n), 0, 95)
    
    # LITERACY RATE: Based on Census 2011 patterns (mean ~73.4%)
    # Wide variation from tribal areas (30%) to urban areas (95%)
    df['literacy_rate'] = np.clip(np.random.normal(73.4, 10.1, n), 30, 95)
    
    # FEMALE LITERACY: Typically 8±4% lower than overall literacy
    # Reflects real gender gaps in Indian education
    df['female_literacy'] = np.clip(
        df['literacy_rate'] - np.random.normal(8, 4, n), 
        20, 95
    )
    
    # SCHOOL INFRASTRUCTURE: Based on DISE (District Information System for Education) data
    # Total schools: Poisson distribution + baseline
    df['total_schools'] = np.random.poisson(lam=300, size=n) + 50
    
    # Government schools: ~74% of total (matches real RTE Act implementation)
    df['govt_schools'] = (df['total_schools'] * np.random.normal(0.74, 0.1, n)).astype(int)
    
    # Private schools: ~23% of total  
    df['private_schools'] = (df['total_schools'] * np.random.normal(0.23, 0.08, n)).astype(int)
    
    # INFRASTRUCTURE QUALITY: Based on real DISE facility data
    # Electricity: ~85% coverage (varies by state)
    df['schools_with_electricity'] = (df['total_schools'] * np.random.normal(0.85, 0.15, n)).astype(int)
    
    # Toilets: ~78% coverage (Swachh Bharat Mission impact)
    df['schools_with_toilets'] = (df['total_schools'] * np.random.normal(0.78, 0.18, n)).astype(int)
    
    # Water: ~82% coverage
    df['schools_with_water'] = (df['total_schools'] * np.random.normal(0.82, 0.16, n)).astype(int)
    
    # TEACHER STATISTICS: Based on real pupil-teacher ratios
    # Average 4.2 teachers per school
    df['total_teachers'] = (df['total_schools'] * np.random.normal(4.2, 1.5, n)).astype(int)
    
    # Trained teachers: ~68% (varies significantly by state)
    df['trained_teachers_percent'] = np.clip(np.random.normal(68, 12, n), 30, 95)
    
    # Female teachers: ~52% (slightly higher than population average)
    df['female_teachers_percent'] = np.clip(np.random.normal(52, 15, n), 20, 80)
    
    # ENROLLMENT AND OUTCOMES
    # Total enrollment: ~180 students per school average
    df['total_enrollment'] = (df['total_schools'] * np.random.normal(180, 80, n)).astype(int)
    
    # Dropout rate: ~12% average (higher in rural/tribal areas)
    df['dropout_rate'] = np.clip(np.random.normal(12, 6, n), 2, 40)
    
    # INFRASTRUCTURE GAP: Composite score based on facility availability
    # Gap = 1 if infrastructure score below 75% threshold
    infrastructure_score = (
        (df['schools_with_electricity'] / df['total_schools'] * 0.4) +
        (df['schools_with_water'] / df['total_schools'] * 0.3) +
        (df['schools_with_toilets'] / df['total_schools'] * 0.3)
    )
    df['infrastructure_gap'] = (infrastructure_score < 0.75).astype(int)
    
    # TEXT DESCRIPTION: District summary for NLP processing
    df['text'] = [
        f"District with {int(lit):.0f}% literacy, {int(urb):.0f}% urban population"
        for lit, urb in zip(df['literacy_rate'], df['urban_pop_percent'])
    ]
    
    # Rename for consistency
    df['gap'] = df['infrastructure_gap']
    df.drop('infrastructure_gap', axis=1, inplace=True)
    
    print(f"Kaggle Simulation: {len(df)} districts, {df['gap'].mean():.1%} infrastructure gap rate")
    return df


# ================================================================================
# VALIDATION FUNCTIONS
# ================================================================================

def validate_simulation_realism():
    """
    Validates that simulated datasets match real data characteristics.
    """
    print("SIMULATION VALIDATION REPORT")
    print("=" * 50)
    
    # Generate datasets
    aser_data = simulate_aser(2400)
    nas_data = simulate_nas(3600)
    kaggle_data = simulate_kaggle(680)
    
    print("\n1. ASER VALIDATION:")
    print(f"   Government schools: {(aser_data['school_type']==0).mean():.1%} (Target: 80-85%)")
    print(f"   Learning gap rate: {aser_data['gap'].mean():.1%} (Target: ~57%)")
    print(f"   Age range: {aser_data['age'].min()}-{aser_data['age'].max()} years (Target: 11-17)")
    
    print("\n2. NAS VALIDATION:")
    print(f"   Rural students: {(nas_data['urban']==0).mean():.1%} (Target: ~60%)")
    print(f"   Achievement gap: {nas_data['gap'].mean():.1%} (Target: ~20%)")
    print(f"   Grade distribution: {nas_data['grade'].value_counts().to_dict()}")
    
    print("\n3. KAGGLE VALIDATION:")
    print(f"   Districts: {len(kaggle_data)} (Target: 680)")
    print(f"   Literacy range: {kaggle_data['literacy_rate'].min():.1f}%-{kaggle_data['literacy_rate'].max():.1f}% (Target: 30-95%)")
    print(f"   Govt schools: {(kaggle_data['govt_schools'].sum()/kaggle_data['total_schools'].sum()):.1%} (Target: ~75%)")
    print(f"   Infrastructure gaps: {kaggle_data['gap'].mean():.1%}")


# ================================================================================
# MAIN EXECUTION
# ================================================================================

if __name__ == "__main__":
    print("INDIAN EDUCATION DATA SIMULATION")
    print("=" * 40)
    print("\nGenerating datasets based on real sources:")
    print("- ASER 2024: Rural education assessment")
    print("- NAS 2021: National achievement survey") 
    print("- Kaggle 2015-16: District infrastructure data")
    print()
    
    # Validate simulation realism
    validate_simulation_realism()
    
    print("\nSimulation complete. Datasets ready for AI model evaluation.")