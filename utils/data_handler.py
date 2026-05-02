# ============================================================
# data_handler.py  —  Load & Preprocess Myntra-style Dataset
# ============================================================
# Dataset: 691 rows, 13 columns (Myntra-style)
# TARGET  : style_tag  (10 classes)
# FEATURES: body_type, skin_tone, occasion, outfit_type, season
# EXTRA   : brand, price_inr, rating, num_reviews (for EDA only)
# ============================================================

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

DATA_PATH    = os.path.join(os.path.dirname(__file__), '..', 'data', 'outfits.csv')
FEATURE_COLS = ['body_type', 'skin_tone', 'occasion', 'outfit_type', 'season']
TARGET_COL   = 'style_tag'


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    print(f"[DATA] Loaded {len(df)} rows x {len(df.columns)} columns.")
    return df


def explore_data(df: pd.DataFrame) -> None:
    print("\n====== Dataset Overview ======")
    print(df[FEATURE_COLS + [TARGET_COL]].head(5).to_string(index=False))
    print(f"\nShape  : {df.shape}")
    print(f"\nStyle tag distribution (TARGET — what the model predicts):")
    print(df[TARGET_COL].value_counts().to_string())
    print(f"\nOutfit type distribution:")
    print(df['outfit_type'].value_counts().to_string())
    if 'price_inr' in df.columns:
        print(f"\nPrice range (INR): ₹{df['price_inr'].min():,} – ₹{df['price_inr'].max():,}")
        print(f"Average rating   : {df['rating'].mean():.2f} / 5.0")
    print("=" * 30)


def encode_and_split(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    df_enc   = df.copy()
    encoders = {}

    for col in FEATURE_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    target_le = LabelEncoder()
    df_enc[TARGET_COL] = target_le.fit_transform(df_enc[TARGET_COL].astype(str))

    X = df_enc[FEATURE_COLS].values
    y = df_enc[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[DATA] Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"[DATA] Classes: {list(target_le.classes_)}")
    return X_train, X_test, y_train, y_test, encoders, target_le


def encode_user_input(user_dict: dict, encoders: dict) -> np.ndarray:
    row = []
    for col in FEATURE_COLS:
        val = user_dict.get(col, '').lower().strip()
        le  = encoders[col]
        if val not in le.classes_:
            print(f"[WARN] Unknown '{val}' for '{col}'. Defaulting to 0.")
            row.append(0)
        else:
            row.append(int(le.transform([val])[0]))
    return np.array([row])
