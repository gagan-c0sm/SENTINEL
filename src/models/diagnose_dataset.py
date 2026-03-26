
import pandas as pd
import numpy as np
from src.database.connection import get_engine
from src.models.dataset import load_features_df, prepare_dataframe

def diagnose():
    df = load_features_df()
    df = prepare_dataframe(df)
    
    print("\n--- Diagnostic Results ---")
    print(f"Total rows: {len(df)}")
    
    nan_counts = df.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    
    if len(nan_cols) == 0:
        print("✅ No NaNs found in any column!")
    else:
        print("❌ Columns with NaNs:")
        print(nan_cols)
        
    # Check specific features used in TFT
    from src.models.config import (
        TARGET, STATIC_CATEGORICALS,
        TIME_VARYING_KNOWN_REALS, TIME_VARYING_KNOWN_CATEGORICALS,
        TIME_VARYING_OBSERVED_REALS_MODEL_B
    )
    
    all_features = [TARGET] + STATIC_CATEGORICALS + TIME_VARYING_KNOWN_REALS + \
                   TIME_VARYING_KNOWN_CATEGORICALS + TIME_VARYING_OBSERVED_REALS_MODEL_B
                   
    print("\n--- Feature Coverage ---")
    for col in all_features:
        if col not in df.columns:
            print(f"⚠️  Missing from DF: {col}")
        else:
            missing = df[col].isna().sum()
            print(f"{col}: {missing} missing ({missing/len(df)*100:.2f}%)")

if __name__ == "__main__":
    diagnose()
