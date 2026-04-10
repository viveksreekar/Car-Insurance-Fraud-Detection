"""
train_xgboost.py — Trains the tabular risk model for AutoShield.
Uses synthetic data generated from real-world insurance fraud patterns.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import os
import joblib

# Constants
MODEL_SAVE_PATH = "saved_model/xgboost_risk_model.joblib"
os.makedirs("saved_model", exist_ok=True)

def generate_synthetic_data(n=2000):
    """
    Generate synthetic insurance data based on the business rules:
    - FIR Mismatch: Major Damage + No FIR = Highly Suspicious
    - Claim Delay: > 14 days = Suspicious
    - Frequent Flyer: > 3 past claims = Suspicious
    """
    np.random.seed(42)
    
    data = {
        'claim_delay_days': np.random.randint(0, 31, n),
        'past_claims_count': np.random.randint(0, 6, n),
        'fir_filed': np.random.choice([0, 1], n),
        'damage_severity': np.random.choice([0, 1], n), # 0: Minor, 1: Major
        'policy_age_days': np.random.randint(0, 365*5, n),
    }
    
    df = pd.DataFrame(data)
    
    # Logic for Fraud Label
    # 1. FIR Mismatch (Major + No FIR)
    condition1 = (df['damage_severity'] == 1) & (df['fir_filed'] == 0)
    # 2. Extreme Delay
    condition2 = (df['claim_delay_days'] > 14)
    # 3. High claim frequency
    condition3 = (df['past_claims_count'] >= 3)
    
    df['fraud_label'] = ((condition1.astype(int) * 0.4) + 
                         (condition2.astype(int) * 0.3) + 
                         (condition3.astype(int) * 0.3) + 
                         (np.random.rand(n) * 0.2))
    
    df['fraud_label'] = (df['fraud_label'] > 0.5).astype(int)
    
    return df

def train():
    df = generate_synthetic_data()
    X = df.drop('fraud_label', axis=1)
    y = df['fraud_label']
    
    print(f"Training XGBoost on {len(df)} samples...")
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        objective='binary:logistic',
        random_state=42
    )
    model.fit(X, y)
    
    joblib.dump(model, MODEL_SAVE_PATH)
    print(f"✅ XGBoost model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
