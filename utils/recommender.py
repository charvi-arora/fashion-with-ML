# ============================================================
# recommender.py — Machine Learning Recommender Engine
# ============================================================
# This module:
#   1. Trains a simple Decision Tree classifier on the outfit data
#   2. Uses it to predict & score outfit matches for a given user
#   3. Falls back to rule-based filtering if needed
#
# Why Decision Tree?
#   → Beginner-friendly, interpretable, works great on small datasets
#   → You can visually explain "if body_type=pear AND occasion=party
#     THEN recommend Floral Wrap Dress"
# ============================================================

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from utils.data_handler import load_data, encode_features, filter_outfits
from models.outfit_model import User, Outfit


class FashionRecommender:
    """
    Core ML recommendation engine.

    Steps:
      1. __init__  → Load data and train the model automatically
      2. recommend → Accept a User object and return top outfit suggestions
    """

    def __init__(self):
        print("[INFO] Initialising FashionRecommender...")
        self.df       = load_data()          # Raw DataFrame
        self.model    = None                 # Will hold trained classifier
        self.encoders = {}                   # Label encoders for features
        self.accuracy = 0.0
        self._train()                        # Train right away

    # ----------------------------------------------------------
    # PRIVATE: Train the Decision Tree classifier
    # ----------------------------------------------------------
    def _train(self):
        """Train a Decision Tree on the outfit dataset."""
        X, y, self.encoders = encode_features(self.df)

        # Split data: 80% train, 20% test (reproducible with random_state)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train the Decision Tree
        # max_depth=5 keeps the tree shallow → less overfitting on small data
        self.model = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        self.accuracy = accuracy_score(y_test, y_pred)

        # NOTE: With only 30 outfit records and each outfit name being a unique
        # label, exact-name accuracy will be low. In a real project you would:
        #   a) Use thousands of rows, OR
        #   b) Predict a category (e.g. outfit_type or style_tag) instead of name.
        # The model still learns the feature patterns correctly — we layer rule-
        # based filtering on top for reliable recommendations.
        print(f"[INFO] Model trained. Test Accuracy (exact name): {self.accuracy:.0%}")
        print("[INFO] Hybrid mode ON: ML prediction + rule-based filtering combined.")

    # ----------------------------------------------------------
    # PUBLIC: Recommend outfits for a given user
    # ----------------------------------------------------------
    def recommend(self, user: User, top_n: int = 3) -> list:
        """
        Returns a list of top_n Outfit objects recommended for the user.

        Strategy:
          1. Use ML model to identify the predicted outfit name
          2. Filter dataset for outfits matching the user's profile
          3. Rank and return top N results
        """
        # Step 1: Encode user's input the same way we encoded training data
        body_enc     = self.encoders['body_type'].transform([user.body_type])[0]
        skin_enc     = self.encoders['skin_tone'].transform([user.skin_tone])[0]
        occasion_enc = self.encoders['occasion'].transform([user.occasion])[0]

        input_features = np.array([[body_enc, skin_enc, occasion_enc]])

        # Step 2: Model predicts the most fitting outfit name
        predicted_outfit_name = self.model.predict(input_features)[0]
        print(f"\n[ML MODEL] Top predicted outfit: '{predicted_outfit_name}'")

        # Step 3: Filter dataset for all matching outfits (rule-based layer)
        matched_df = filter_outfits(
            self.df, user.body_type, user.skin_tone, user.occasion
        )

        # If the ML predicted outfit isn't in the filtered results, add it
        if predicted_outfit_name not in matched_df['outfit_name'].values:
            ml_row = self.df[self.df['outfit_name'] == predicted_outfit_name]
            matched_df = pd.concat([ml_row, matched_df])

        # Step 4: De-duplicate and limit to top_n results
        matched_df = matched_df.drop_duplicates(subset='outfit_name').head(top_n)

        # Step 5: Convert DataFrame rows → Outfit objects
        recommendations = []
        for _, row in matched_df.iterrows():
            outfit = Outfit(
                outfit_id   = int(row['outfit_id']),
                outfit_name = row['outfit_name'],
                outfit_type = row['outfit_type'],
                color       = row['color'],
                season      = row['season'],
                occasion    = row['occasion'],
                body_type   = row['body_type'],
                skin_tone   = row['skin_tone'],
                style_tag   = row['style_tag'],
            )
            recommendations.append(outfit)

        return recommendations

    def get_accuracy(self) -> float:
        """Return model accuracy as a float (e.g., 0.83 = 83%)."""
        return self.accuracy
