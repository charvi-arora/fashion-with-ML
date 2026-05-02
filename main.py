#!/usr/bin/env python3
# ============================================================
# main.py  —  Fashion AI: Final Complete Version
# ============================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from models.outfit_model   import User, Outfit
from utils.data_handler    import (load_data, explore_data, encode_and_split,
                                    encode_user_input, FEATURE_COLS, TARGET_COL)
from utils.ml_engine       import (train_and_evaluate, get_feature_importance,
                                    save_model, predict_style)
from utils.learning_engine import (retrain_with_accumulated_data, should_retrain,
                                    initialise_online_model, online_update,
                                    online_predict, compare_predictions,
                                    RETRAIN_THRESHOLD)
from utils.visualizer      import generate_all_charts, plot_budget_analysis
from utils.db_handler      import (setup_database, save_session, save_feedback,
                                    get_sessions_as_dataframe,
                                    count_sessions, print_session_history)
from utils.price_linker    import (get_price_input, get_price_filtered_recommendations,
                                    display_price_results)


def banner():
    print("\n" + "="*66)
    print("   Fashion AI — Myntra-Style Personal Styling System")
    print("   691 products | 10 style classes | Budget filter + Shop links")
    print("="*66)


def show_ml_recommendations(user, style_predicted, df):
    """Show top ML-recommended outfits (no price filter)."""
    matches = df[
        (df['style_tag'] == style_predicted) &
        (df['occasion']  == user.occasion)
    ].sort_values('rating', ascending=False)

    if matches.empty:
        matches = df[df['style_tag'] == style_predicted].sort_values('rating', ascending=False)

    outfits = []
    if not matches.empty:
        print(f"\n  Top ML Picks  [{style_predicted.upper().replace('_',' ')} | {user.occasion}]:\n")
        for _, row in matches.head(5).iterrows():
            o = Outfit(
                outfit_id    = int(row['outfit_id']),
                product_name = row['product_name'],
                brand        = row['brand'],
                outfit_type  = row['outfit_type'],
                color        = row['color'],
                season       = row['season'],
                occasion     = row['occasion'],
                body_type    = row['body_type'],
                skin_tone    = row['skin_tone'],
                style_tag    = row['style_tag'],
                price_inr    = int(row.get('price_inr', 0)),
                rating       = float(row.get('rating', 0)),
                num_reviews  = int(row.get('num_reviews', 0)),
            )
            print(f"    • {o.product_name:<46} [{o.brand:<18}]  ₹{o.price_inr:>6,}  ★{o.rating}")
            outfits.append(o)
    return outfits


def get_user_input() -> User:
    print("\nEnter your profile:\n")
    name        = input("  Name           : ").strip() or "Guest"
    body_type   = input("  Body type      [pear/apple/rectangle/hourglass]     : ").strip().lower()
    skin_tone   = input("  Skin tone      [warm/cool/neutral]                  : ").strip().lower()
    occasion    = input("  Occasion       [casual/party/formal]                : ").strip().lower()
    print("  Outfit types   : kurta | saree | lehenga | dress | formal_set")
    print("                   top_jeans | co_ord_set | jumpsuit | anarkali")
    print("                   palazzo_set | shirt_trouser | maxi_dress")
    outfit_type = input("  Outfit type    : ").strip().lower() or "dress"
    season      = input("  Season         [summer/winter/spring/autumn/all]    : ").strip().lower() or "all"
    try:
        return User(name, body_type, skin_tone, occasion, outfit_type, season)
    except ValueError as e:
        print(f"[ERROR] {e}"); sys.exit(1)


def handle_feedback(session_id, batch_style, target_le, X_user, is_demo):
    if is_demo:
        styles     = list(target_le.classes_)
        correction = [s for s in styles if s != batch_style][0]
        print(f"\n  [DEMO Feedback] Simulating correction → '{correction}'")
        save_feedback(session_id, 4, correction)
        online_update(X_user, correction, target_le)
        print(f"  [Online SGD] Instantly updated with correction '{correction}'.")
        return

    print("\n  --- Feedback (press Enter to skip) ---")
    r = input("  Rate recommendation [1-5]: ").strip()
    if not r: return
    try:
        rating = max(1, min(5, int(r)))
    except ValueError:
        return
    correct = None
    if rating <= 3:
        print(f"  Style classes: {list(target_le.classes_)}")
        correct = input("  Correct style should be: ").strip().lower()
        if correct not in target_le.classes_: correct = None
    save_feedback(session_id, rating, correct)
    if correct:
        online_update(X_user, correct, target_le)
        print(f"  [Online SGD] Updated with '{correct}'.")


def main():
    banner()
    is_demo = '--demo' in sys.argv

    # ── DB ────────────────────────────────────────────────────
    setup_database()
    n_sessions = count_sessions()
    print(f"[DB] Sessions accumulated: {n_sessions}")

    # ── Load & explore ────────────────────────────────────────
    df = load_data()
    explore_data(df)

    # ── Encode + split ────────────────────────────────────────
    X_train, X_test, y_train, y_test, encoders, target_le = encode_and_split(df)
    class_names = list(target_le.classes_)

    # ── Train & evaluate ──────────────────────────────────────
    best_model, best_name, results = train_and_evaluate(
        X_train, X_test, y_train, y_test, class_names
    )
    feat_imp = get_feature_importance(best_model, FEATURE_COLS)

    if feat_imp:
        print("\n--- Feature Importance ---")
        for f, v in sorted(feat_imp.items(), key=lambda x: x[1], reverse=True):
            print(f"  {f:<15} {'█'*int(v*40)} {v*100:.1f}%")

    save_model(best_model, encoders, target_le)

    # ── Online model ──────────────────────────────────────────
    print("\n[SETUP] Initialising Online SGD model...")
    initialise_online_model(X_train, y_train,
                             list(range(len(class_names))), target_le, encoders)

    # ── Batch retrain check ───────────────────────────────────
    if should_retrain(n_sessions):
        print(f"\n[RETRAIN] {n_sessions} sessions → retraining now...")
        sdf = get_sessions_as_dataframe()
        if 'predicted_style' in sdf.columns:
            sdf = sdf.rename(columns={'predicted_style': TARGET_COL})
        r = retrain_with_accumulated_data(df, sdf)
        best_model  = r['model'];  encoders    = r['encoders']
        target_le   = r['target_le']; class_names = list(target_le.classes_)
    else:
        left = RETRAIN_THRESHOLD - (n_sessions % RETRAIN_THRESHOLD)
        print(f"\n[RETRAIN] Batch retrain in {left} more session(s).")

    # ── User profile ──────────────────────────────────────────
    print("\n" + "─"*66)
    if is_demo:
        print("[DEMO] Profile: Sneha | hourglass | warm | party | lehenga | winter")
        user = User("Sneha", "hourglass", "warm", "party", "lehenga", "winter")
    else:
        user = get_user_input()

    # ── ML Predict ────────────────────────────────────────────
    X_user = encode_user_input(user.as_feature_dict(), encoders)
    batch_style,  batch_probs  = predict_style(best_model, X_user, target_le)
    online_style, online_probs = online_predict(X_user)

    compare_predictions(X_user, best_model, target_le, online_style, online_probs)

    print(f"\n{'─'*66}")
    print(f"  Hello {user.name}!  Predicted style: {batch_style.upper().replace('_',' ')}")
    recommended = show_ml_recommendations(user, batch_style, df)

    # ── Save session ──────────────────────────────────────────
    session_id = save_session(
        user.name, user.body_type, user.skin_tone, user.occasion,
        user.outfit_type, user.season, batch_style, recommended
    )

    # ── PRICE FILTER + LINKS ──────────────────────────────────
    print(f"\n{'─'*66}")
    print("  BUDGET-BASED SHOPPING")
    print(f"{'─'*66}")
    min_price, max_price = get_price_input(is_demo)

    price_result = get_price_filtered_recommendations(
        df          = df,
        style_tag   = batch_style,
        outfit_type = user.outfit_type,
        occasion    = user.occasion,
        min_price   = min_price,
        max_price   = max_price,
        top_n       = 5,
    )
    display_price_results(price_result, batch_style)

    # ── Feedback ──────────────────────────────────────────────
    handle_feedback(session_id, batch_style, target_le, X_user, is_demo)
    print_session_history()

    # ── All charts including budget analysis ──────────────────
    y_pred = results[best_name]['y_pred']
    chart_paths = generate_all_charts(
        results=results, best_name=best_name, df=df,
        feature_importance=feat_imp, class_names=class_names,
        y_test=y_test, y_pred=y_pred,
        probabilities=batch_probs,
        user_name=user.name, predicted_style=batch_style,
    )

    # Budget chart (chart #9)
    budget_chart = plot_budget_analysis(df, min_price, max_price, batch_style, price_result)
    if budget_chart:
        chart_paths.append(budget_chart)

    # ── Final summary ─────────────────────────────────────────
    best_acc = results[best_name]['test_acc']
    print("\n" + "="*66)
    print("  FINAL SUMMARY")
    print("="*66)
    print(f"  Dataset          : 691 Myntra-style products | 10 style classes")
    print(f"  Best Model       : {best_name}  ({best_acc:.1%} test accuracy)")
    print(f"  Predicted Style  : {batch_style.replace('_',' ').title()}")
    print(f"  Budget           : ₹{min_price:,} – ₹{max_price:,}")
    budget_found = len(price_result['in_range']) if not price_result['in_range'].empty else 0
    print(f"  In-budget outfits: {budget_found}")
    print(f"  Sessions in DB   : {count_sessions()}")
    print(f"  Charts generated : {len(chart_paths)}")
    print(f"  Shop links       : Myntra | AJIO | Amazon India")
    print("="*66 + "\n")


if __name__ == '__main__':
    main()
