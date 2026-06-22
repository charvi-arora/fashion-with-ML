# ============================================================
# learning_engine.py  —  Two Learning Approaches
# ============================================================
#
# APPROACH 1: Retrain on Accumulated Data
# ─────────────────────────────────────────
#   - Every user prediction is saved to SQLite
#   - When enough new data accumulates, we RETRAIN the Random Forest
#     on: original CSV data + all saved session data combined
#   - The model file (best_model.pkl) is updated on disk
#   - Next run uses the improved model automatically
#
#   Think of it like: the model "reads its diary" of past sessions
#   and relearns everything from scratch, now with more examples.
#
#   PROS : Simple, reliable, works with any sklearn model
#   CONS : Must retrain fully each time (slow for huge datasets)
#
#
# APPROACH 2: True Online Learning (SGDClassifier)
# ─────────────────────────────────────────────────
#   - Uses sklearn's SGDClassifier with partial_fit()
#   - partial_fit() updates model weights with JUST the new sample
#     without seeing old data again
#   - The online model is saved separately (online_model.pkl)
#
#   Think of it like: the model tweaks itself slightly after each
#   new person uses it — no full retraining needed.
#
#   PROS : Instant update, scales to millions of users
#   CONS : Can "forget" old patterns (called catastrophic forgetting)
#
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble        import RandomForestClassifier
from sklearn.linear_model    import SGDClassifier
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

from utils.data_handler import FEATURE_COLS, TARGET_COL, encode_and_split

MODEL_DIR          = '/tmp/fashion_ai_outputs'
BATCH_MODEL_PATH   = os.path.join(MODEL_DIR, 'best_model.pkl')
ONLINE_MODEL_PATH  = os.path.join(MODEL_DIR, 'online_model.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)

# How many new sessions must accumulate before we retrain (Approach 1)
RETRAIN_THRESHOLD = 5


# ══════════════════════════════════════════════════════════════
# APPROACH 1: RETRAIN ON ACCUMULATED DATA
# ══════════════════════════════════════════════════════════════

def retrain_with_accumulated_data(base_df: pd.DataFrame,
                                   session_df: pd.DataFrame) -> dict:
    """
    Combine the base dataset with all saved session data,
    then fully retrain the Random Forest on the combined data.

    Args:
        base_df    : Original outfits.csv DataFrame
        session_df : Sessions from SQLite (new I/O data)

    Returns:
        dict with new model, encoders, target_le, accuracy
    """
    print("\n" + "─"*55)
    print("  APPROACH 1: Retraining on Accumulated Session Data")
    print("─"*55)

    if session_df.empty:
        print("  [Retrain] No session data yet. Using base dataset only.")
        combined = base_df.copy()
    else:
        # Keep only columns that match our features + target
        needed_cols = FEATURE_COLS + [TARGET_COL]
        session_subset = session_df[
            [c for c in needed_cols if c in session_df.columns]
        ].dropna()

        if session_subset.empty:
            combined = base_df.copy()
        else:
            combined = pd.concat([base_df, session_subset], ignore_index=True)
            print(f"  [Retrain] Base rows: {len(base_df)} | "
                  f"Session rows: {len(session_subset)} | "
                  f"Combined: {len(combined)}")

    # Encode features
    encoders  = {}
    df_enc    = combined.copy()

    for col in FEATURE_COLS:
        le = LabelEncoder()
        df_enc[col] = le.fit_transform(df_enc[col].astype(str))
        encoders[col] = le

    target_le = LabelEncoder()
    df_enc[TARGET_COL] = target_le.fit_transform(df_enc[TARGET_COL].astype(str))

    X = df_enc[FEATURE_COLS].values
    y = df_enc[TARGET_COL].values

    # Need at least 2 samples per class for stratified split
    class_counts = pd.Series(y).value_counts()
    can_stratify = (class_counts >= 2).all()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if can_stratify else None
    )

    # Retrain Random Forest
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"  [Retrain] New test accuracy: {acc:.1%}  "
          f"(trained on {len(X_train)} samples)")

    # Save updated model
    payload = {'model': model, 'encoders': encoders, 'target_le': target_le}
    joblib.dump(payload, BATCH_MODEL_PATH)
    print(f"  [Retrain] Model updated and saved → {BATCH_MODEL_PATH}")
    print("─"*55)

    return {'model': model, 'encoders': encoders,
            'target_le': target_le, 'accuracy': acc}


def should_retrain(n_sessions: int) -> bool:
    """
    Returns True if enough new sessions have accumulated
    to justify a retrain.
    """
    return n_sessions > 0 and n_sessions % RETRAIN_THRESHOLD == 0


# ══════════════════════════════════════════════════════════════
# APPROACH 2: TRUE ONLINE LEARNING (SGDClassifier)
# ══════════════════════════════════════════════════════════════

def initialise_online_model(X_train: np.ndarray, y_train: np.ndarray,
                             class_list: list, target_le: LabelEncoder,
                             encoders: dict):
    """
    Create and warm-start the SGDClassifier on the base training data.
    Saves the online model + encoders to online_model.pkl.

    SGDClassifier supports partial_fit() — the key to online learning.
    loss='log_loss' makes it a probabilistic classifier (like logistic regression).
    """
    print("\n" + "─"*55)
    print("  APPROACH 2: Initialising Online Learning Model")
    print("─"*55)

    model = SGDClassifier(
        loss='log_loss',      # enables predict_proba()
        random_state=42,
        max_iter=1000,
        tol=1e-3
    )

    # Initial fit on all base training data
    model.fit(X_train, y_train)

    payload = {
        'model':     model,
        'encoders':  encoders,
        'target_le': target_le,
    }
    joblib.dump(payload, ONLINE_MODEL_PATH)
    print(f"  [Online] Initialised on {len(X_train)} training samples.")
    print(f"  [Online] Saved → {ONLINE_MODEL_PATH}")
    print("─"*55)
    return model


def online_update(X_new: np.ndarray, y_new_label: str, target_le: LabelEncoder):
    """
    Update the online model with ONE new data point using partial_fit().

    This is the core of online learning:
      - No full retraining needed
      - Model adjusts its weights for the new sample only
      - All previous knowledge is retained (mostly)

    Args:
        X_new       : encoded feature array, shape (1, 5)
        y_new_label : the TRUE style tag for this sample (string)
        target_le   : LabelEncoder to convert label → integer
    """
    if not os.path.exists(ONLINE_MODEL_PATH):
        print("[Online] Online model not found. Run initialise first.")
        return

    # Load current online model
    payload   = joblib.load(ONLINE_MODEL_PATH)
    model     = payload['model']
    all_classes = list(range(len(target_le.classes_)))

    # Encode the true label
    if y_new_label not in target_le.classes_:
        print(f"[Online] Unknown label '{y_new_label}'. Skipping update.")
        return

    y_new = target_le.transform([y_new_label])

    # THE KEY LINE: partial_fit updates weights with just this one sample
    model.partial_fit(X_new, y_new, classes=all_classes)

    # Save updated model back to disk
    payload['model'] = model
    joblib.dump(payload, ONLINE_MODEL_PATH)
    print(f"  [Online] Model updated with new sample (label: '{y_new_label}').")


def online_predict(X_input: np.ndarray) -> tuple:
    """
    Make a prediction using the online model.
    Returns (predicted_label, probabilities_dict).
    """
    if not os.path.exists(ONLINE_MODEL_PATH):
        return None, {}

    payload    = joblib.load(ONLINE_MODEL_PATH)
    model      = payload['model']
    target_le  = payload['target_le']

    pred_index = model.predict(X_input)[0]
    label      = target_le.inverse_transform([pred_index])[0]

    try:
        proba = model.predict_proba(X_input)[0]
        probs = {cls: float(f"{p:.2f}")
                 for cls, p in zip(target_le.classes_, proba)}
    except Exception:
        probs = {label: 1.0}

    return label, probs


# ══════════════════════════════════════════════════════════════
# COMPARE BOTH MODELS ON THE SAME INPUT
# ══════════════════════════════════════════════════════════════

def compare_predictions(X_input: np.ndarray,
                         batch_model, batch_target_le,
                         online_label: str, online_probs: dict):
    """
    Print a side-by-side comparison of both models' predictions
    for the same user input — great for showing the difference visually.
    """
    # Batch model prediction
    batch_pred  = batch_model.predict(X_input)[0]
    batch_label = batch_target_le.inverse_transform([batch_pred])[0]
    batch_proba = batch_model.predict_proba(X_input)[0]
    batch_top   = sorted(zip(batch_target_le.classes_, batch_proba),
                         key=lambda x: x[1], reverse=True)[:3]

    print("\n" + "═"*55)
    print("  PREDICTION COMPARISON: Batch vs Online Model")
    print("═"*55)
    print(f"  {'Model':<22} {'Prediction':<14} Top Confidence")
    print("  " + "─"*51)

    # Batch row
    top3_batch = "  |  ".join([f"{c}: {p*100:.0f}%" for c, p in batch_top])
    print(f"  {'Random Forest (Batch)':<22} {batch_label:<14} {top3_batch}")

    # Online row
    if online_label:
        top3_online = sorted(online_probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top3_str    = "  |  ".join([f"{c}: {v*100:.0f}%" for c, v in top3_online])
        print(f"  {'SGD (Online)':<22} {online_label:<14} {top3_str}")
    else:
        print(f"  {'SGD (Online)':<22} {'(not ready)':<14}")

    print("═"*55)
    match = "✅ AGREE" if online_label == batch_label else "⚠️  DIFFER"
    print(f"  Both models: {match}")
    print("═"*55)
