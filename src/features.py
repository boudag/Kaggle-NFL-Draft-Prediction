import pandas as pd
import numpy as np

def engineer_features(df):
    df = df.copy()
    
    # Required Features
    # Vertical_Jump is often missing, so we must handle that during imputation later, 
    # but for feature calculation, we use the values we have.
    df['Power_Factor'] = df['Weight'] * df['Vertical_Jump']
    df['Speed_to_Size_Ratio'] = df['Sprint_40yd'] / df['Weight']
    
    # Additional Features
    df['Total_Jump'] = df['Vertical_Jump'] + df['Broad_Jump']
    df['Agility_Score'] = df['Agility_3cone'] + df['Shuttle']
    
    # BMI (Height is in inches in NFL combine usually, but let's check)
    # If Height is ~70-80, it's inches. 
    # BMI = (Weight_lbs / Height_inches^2) * 703
    df['BMI'] = (df['Weight'] / (df['Height'] ** 2)) * 703
    
    return df
