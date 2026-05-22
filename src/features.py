import pandas as pd
import numpy as np

# A comprehensive list of major NCAA Power 5 athletic conference schools
# to segment athletes by level of competition.
POWER_5_SCHOOLS = {
    # SEC
    'Alabama', 'LSU', 'Georgia', 'Florida', 'Auburn', 'Arkansas', 'Texas A&M', 'Missouri', 
    'South Carolina', 'Ole Miss', 'Mississippi St.', 'Tennessee', 'Kentucky', 'Vanderbilt', 'Mississippi',
    # Big Ten
    'Ohio St.', 'Penn St.', 'Wisconsin', 'Michigan', 'Michigan St.', 'Iowa', 'Nebraska', 
    'Minnesota', 'Illinois', 'Northwestern', 'Purdue', 'Indiana', 'Rutgers', 'Maryland',
    # ACC
    'Clemson', 'Florida St.', 'Miami (FL)', 'Virginia Tech', 'North Carolina', 'Duke', 
    'Georgia Tech', 'Louisville', 'NC State', 'Pittsburgh', 'Boston College', 'Syracuse', 
    'Virginia', 'Wake Forest', 'Miami',
    # Big 12
    'Oklahoma', 'Texas', 'West Virginia', 'TCU', 'Baylor', 'Oklahoma St.', 'Kansas St.', 
    'Iowa St.', 'Texas Tech', 'Kansas', 'BYU', 'Houston', 'UCF', 'Cincinnati',
    # Pac-12
    'USC', 'UCLA', 'Stanford', 'Oregon', 'Washington', 'Utah', 'Arizona St.', 'California', 
    'Washington St.', 'Oregon St.', 'Colorado', 'Arizona',
    # Independent
    'Notre Dame'
}

def get_archetype(row):
    """
    Categorizes players into physical archetypes deterministically based on height and weight.
    Height is in meters, Weight is in kg.
    """
    h, w = row['Height'], row['Weight']
    if pd.isnull(h) or pd.isnull(w):
        return 'Lightweight_Skills'
    if w >= 120:
        return 'Heavyweight_Lineman'
    elif w >= 95:
        if h >= 1.90:
            return 'Big_Hybrid_Edge_TE'
        else:
            return 'Dense_Power_LB_RB'
    else:
        if h >= 1.88:
            return 'Tall_Speed_WR_DB'
        else:
            return 'Lightweight_Skills'

def engineer_features(df):
    """
    Passthrough feature engineering. All advanced leak-free feature engineering is now 
    performed strictly inside PreprocessingPipeline during train/val split scaling.
    """
    df = df.copy()
    # Adding basic leak-free features
    df['Power_Factor'] = df['Weight'] * df['Vertical_Jump']
    df['Speed_to_Size_Ratio'] = df['Sprint_40yd'] / df['Weight']
    df['Total_Jump'] = df['Vertical_Jump'] + df['Broad_Jump']
    df['Agility_Score'] = df['Agility_3cone'] + df['Shuttle']
    df['BMI'] = (df['Weight'] / (df['Height'] ** 2)) * 703
    df['School_Conference'] = df['School'].apply(lambda x: 'Power_5' if x in POWER_5_SCHOOLS else 'Other_Conference')
    df['Physical_Archetype'] = df.apply(get_archetype, axis=1)
    return df