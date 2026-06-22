# ============================================================
# ml_engine.py  —  Train, Evaluate & Compare ML Models
# ============================================================
# This is the CORE machine learning file.
#
# Models trained & compared:
#   1. Random Forest Classifier  (main model)
#   2. Decision Tree Classifier  (simple baseline)
#   3. K-Nearest Neighbors       (distance-based baseline)
#
# Metrics computed:
#   - Accuracy
#   - Classification Report (Precision, Recall, F1 per class)
#   - Confusion Matrix
#   - Feature Importance (Random Forest only)
#
# The best model is saved to disk using joblib so it can be
# loaded later without retraining.
# ============================================================

import os
import numpy as np
import joblib

from sklearn.ensemble         import RandomForestClassifier
from sklearn.tree             import DecisionTreeClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.metrics          import (accuracy_score,
                                      classification_report,
                                      confusion_matrix)

MODEL_DIR  = '/tmp/fashion_ai_outputs'
MODEL_PATH = os.path.join(MODEL_DIR, 'best_model.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# 1. BUILD ALL THREE MODELS
# ─────────────────────────────────────────────

def build_models() -> dict:
    """
    Returns a dict of model_name -> untrained model object.

    Random Forest: many Decision Trees voting together → more accurate
    Decision Tree: single tree, very interpretable
    KNN          : predicts based on K nearest training examples
    """
    return {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,   # 100 trees in the forest
            max_depth=10,       # prevent overfitting
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        ),
        'KNN (k=5)': KNeighborsClassifier(
            n_neighbors=5      # compare each sample to its 5 nearest neighbours
        ),
    }


# ─────────────────────────────────────────────
# 2. TRAIN + EVALUATE ALL MODELS
# ─────────────────────────────────────────────

def train_and_evaluate(X_train, X_test, y_train, y_test,
                       class_names: list) -> tuple:
    """
    Train every model, evaluate on the test set, print a comparison table.

    Returns:
      best_model   : the model with highest test accuracy
      best_name    : its name (string)
      results      : dict of {model_name: {accuracy, report, cm}}
    """
    models  = build_models()
    results = {}

    print("\n====== Model Training & Evaluation ======")
    print(f"{'Model':<20} {'Train Acc':>10} {'Test Acc':>10}")
    print("-" * 42)

    best_acc   = 0.0
    best_model = None
    best_name  = ""

    for name, model in models.items():
        # --- TRAIN ---
        model.fit(X_train, y_train)

        # --- PREDICT ---
        y_pred_train = model.predict(X_train)
        y_pred_test  = model.predict(X_test)

        # --- METRICS ---
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc  = accuracy_score(y_test,  y_pred_test)

        report = classification_report(
            y_test, y_pred_test,
            target_names=class_names,
            zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred_test)

        results[name] = {
            'model':      model,
            'train_acc':  train_acc,
            'test_acc':   test_acc,
            'report':     report,
            'cm':         cm,
            'y_pred':     y_pred_test,
        }

        print(f"{name:<20} {train_acc:>9.1%} {test_acc:>10.1%}")

        if test_acc > best_acc:
            best_acc   = test_acc
            best_model = model
            best_name  = name

    print("-" * 42)
    print(f"\nBest model: {best_name}  ({best_acc:.1%} test accuracy)")

    # Print full classification report for the best model
    print(f"\n--- Classification Report: {best_name} ---")
    print(results[best_name]['report'])

    return best_model, best_name, results


# ─────────────────────────────────────────────
# 3. FEATURE IMPORTANCE (Random Forest only)
# ─────────────────────────────────────────────

def get_feature_importance(model, feature_names: list) -> dict:
    """
    Return feature importances from a Random Forest.
    Higher value = that feature matters more for predictions.
    """
    if not hasattr(model, 'feature_importances_'):
        return {}
    importances = model.feature_importances_
    return dict(zip(feature_names, importances))


# ─────────────────────────────────────────────
# 4. SAVE / LOAD MODEL
# ─────────────────────────────────────────────

def save_model(model, encoders: dict, target_le) -> None:
    """Save the trained model + encoders to disk with joblib."""
    payload = {
        'model':      model,
        'encoders':   encoders,
        'target_le':  target_le,
    }
    joblib.dump(payload, MODEL_PATH)
    print(f"[MODEL] Saved to {MODEL_PATH}")


def load_model() -> tuple:
    """Load a previously saved model from disk."""
    payload   = joblib.load(MODEL_PATH)
    print(f"[MODEL] Loaded from {MODEL_PATH}")
    return payload['model'], payload['encoders'], payload['target_le']


# ─────────────────────────────────────────────
# 5. PREDICT FOR A SINGLE USER
# ─────────────────────────────────────────────

def predict_style(model, X_input: np.ndarray, target_le) -> tuple:
    """
    Predict style_tag for a single encoded user input.

    Returns:
      predicted_label : string like 'feminine'
      probabilities   : dict of {class_name: probability}
    """
    pred_index = model.predict(X_input)[0]
    predicted_label = target_le.inverse_transform([pred_index])[0]

    # Confidence scores for each class
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X_input)[0]
        probabilities = {
            cls: float(f"{p:.2f}")
            for cls, p in zip(target_le.classes_, proba)
        }
    else:
        probabilities = {predicted_label: 1.0}

    return predicted_label, probabilities
