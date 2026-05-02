# ============================================================
# explainer.py  —  "Why this recommendation?" Logic
# ============================================================
# Pure rule-based logic. No ML, no AI.
# Simply checks user inputs against known fashion rules
# and returns plain-English reasons.
#
# Interview answer: "I used simple if-else rules based on
# standard fashion guidelines — body type flattery, occasion
# appropriateness, and budget fit. No black-box AI needed."
# ============================================================

# Body type → outfit types that flatter it
BODY_TYPE_RULES = {
    'hourglass':  ['lehenga', 'dress', 'saree', 'anarkali', 'co_ord_set'],
    'pear':       ['anarkali', 'palazzo_set', 'kurta', 'maxi_dress', 'lehenga'],
    'apple':      ['kurta', 'palazzo_set', 'shirt_trouser', 'maxi_dress', 'jumpsuit'],
    'rectangle':  ['co_ord_set', 'formal_set', 'dress', 'shirt_trouser', 'jumpsuit'],
}

BODY_TYPE_TIPS = {
    'hourglass':  'balances your proportions beautifully',
    'pear':       'draws attention upward and adds volume to your top half',
    'apple':      'creates a defined silhouette and flows over the midsection',
    'rectangle':  'adds curves and creates the illusion of a defined waist',
}

# Occasion → suitable style tags
OCCASION_STYLE_MAP = {
    'party':  ['festive', 'glam', 'chic', 'western_casual'],
    'formal': ['corporate', 'ethnic_formal', 'traditional'],
    'casual': ['ethnic_casual', 'western_casual', 'boho', 'minimal'],
}

# Skin tone → color families that complement it
SKIN_TONE_COLORS = {
    'warm':    ['coral', 'rust', 'mustard', 'terracotta', 'gold', 'olive', 'saffron'],
    'cool':    ['cobalt', 'lavender', 'teal', 'royal blue', 'emerald', 'lilac'],
    'neutral': ['white', 'black', 'ivory', 'cream', 'grey', 'nude', 'champagne'],
}


def get_recommendation_reasons(user_body_type: str,
                                user_skin_tone: str,
                                user_occasion: str,
                                user_outfit_type: str,
                                predicted_style: str,
                                outfit_color: str,
                                outfit_price: int,
                                min_price: int,
                                max_price: int) -> list:
    """
    Returns a list of plain-English reasons why this outfit was recommended.
    Each reason is a short string shown in the UI.
    """
    reasons = []

    # ── Reason 1: Body type match ────────────────────────────
    suitable_types = BODY_TYPE_RULES.get(user_body_type, [])
    tip = BODY_TYPE_TIPS.get(user_body_type, 'suits your body type')
    if user_outfit_type in suitable_types:
        reasons.append(
            f"✅ Body type match — {user_outfit_type.replace('_',' ').title()} "
            f"{tip} for a {user_body_type} figure"
        )
    else:
        reasons.append(
            f"👗 Recommended for your {user_body_type} body type "
            f"based on occasion and style preferences"
        )

    # ── Reason 2: Occasion suitability ──────────────────────
    suitable_styles = OCCASION_STYLE_MAP.get(user_occasion, [])
    if predicted_style in suitable_styles:
        reasons.append(
            f"✅ Occasion match — {predicted_style.replace('_',' ').title()} "
            f"style is ideal for {user_occasion} events"
        )
    else:
        reasons.append(
            f"📅 Chosen for your {user_occasion} occasion "
            f"based on overall style compatibility"
        )

    # ── Reason 3: Skin tone → color compatibility ────────────
    good_colors = SKIN_TONE_COLORS.get(user_skin_tone, [])
    color_lower = outfit_color.lower().replace('_', ' ')
    color_match = any(gc in color_lower for gc in good_colors)
    if color_match:
        reasons.append(
            f"✅ Color match — {outfit_color.replace('_',' ').title()} "
            f"complements {user_skin_tone} skin tones beautifully"
        )
    else:
        reasons.append(
            f"🎨 Color selected based on style and seasonal trends "
            f"for {user_skin_tone} undertones"
        )

    # ── Reason 4: Budget fit ─────────────────────────────────
    if min_price <= outfit_price <= max_price:
        reasons.append(
            f"✅ Within budget — ₹{outfit_price:,} fits your "
            f"₹{min_price:,}–₹{max_price:,} range"
        )
    else:
        reasons.append(
            f"💡 Nearest available option — ₹{outfit_price:,} "
            f"(closest to your ₹{min_price:,}–₹{max_price:,} budget)"
        )

    return reasons


def get_confidence_explanation(confidence: float) -> tuple:
    """
    Returns (label, explanation, color) based on confidence score.

    Interview answer: "I used predict_proba() from scikit-learn to get
    the probability score for the predicted class, then mapped it to
    human-readable confidence levels."
    """
    if confidence >= 0.80:
        return ("Very High", "The model is highly confident this style suits your profile.", "#2E7D32")
    elif confidence >= 0.60:
        return ("High", "Strong match between your inputs and this style category.", "#558B2F")
    elif confidence >= 0.40:
        return ("Moderate", "Good match — a few factors are borderline.", "#F57F17")
    else:
        return ("Low", "General recommendation based on your inputs — explore other styles too.", "#C62828")


def get_fallback_message(confidence: float) -> str | None:
    """
    Returns a fallback message if confidence is below threshold.
    Returns None if confidence is fine.
    """
    if confidence < 0.40:
        return (
            "This is a general recommendation based on your inputs. "
            "Try adjusting your occasion or outfit type for a more confident match."
        )
    return None
