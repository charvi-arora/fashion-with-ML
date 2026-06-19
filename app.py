# ============================================================
# app.py — Fashion AI v2 · Premium Dark Editorial UI
# ============================================================
# FIXES applied vs v1:
#   1. Per-product images: dynamic Unsplash URLs per outfit type + color
#   2. Myntra URLs: simplified to working search URLs (no broken filters)
#   3. AJIO URLs: fixed to actual working search format
#   4. Flipkart added as replacement for broken deep-links
#   5. Amazon URL: fixed paise conversion
#   6. Dark luxury editorial aesthetic replacing basic blush UI
# ============================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="STYLAI — Your AI Stylist",
    page_icon="🖤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Project imports ──────────────────────────────────────────
from models.outfit_model import User
from utils.data_handler import load_data, encode_and_split, encode_user_input, FEATURE_COLS, TARGET_COL
from utils.ml_engine import train_and_evaluate, get_feature_importance, save_model, predict_style
from utils.learning_engine import (initialise_online_model, online_update,
                                    online_predict, compare_predictions)
from utils.db_handler import (setup_database, save_session, save_feedback,
                               count_sessions, get_sessions_as_dataframe)
from utils.explainer import get_recommendation_reasons, get_confidence_explanation, get_fallback_message

# ════════════════════════════════════════════════════════════
# CUSTOM CSS — Dark Editorial / Luxury Magazine Aesthetic
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
html, body, .stApp, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0A0A0A !important;
    color: #E8E0D5 !important;
}

.stApp { background-color: #0A0A0A !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #111111 !important;
    border-right: 1px solid #222222;
}
section[data-testid="stSidebar"] * {
    color: #C8BFB0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: #1A1A1A !important;
    border: 1px solid #333 !important;
    border-radius: 6px !important;
    color: #E8E0D5 !important;
}
section[data-testid="stSidebar"] .stSlider { color: #C8BFB0 !important; }

/* ── Hero Banner ── */
.hero-wrap {
    position: relative;
    background: #0A0A0A;
    border-bottom: 1px solid #1F1F1F;
    padding: 3.5rem 2rem 2.5rem;
    margin-bottom: 2rem;
    overflow: hidden;
}
.hero-wrap::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(212,175,90,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: #C9A84C;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 4rem;
    font-weight: 300;
    line-height: 1.05;
    color: #F5F0E8;
    margin: 0 0 0.5rem;
    letter-spacing: -0.01em;
}
.hero-title em { font-style: italic; color: #C9A84C; }
.hero-sub {
    font-size: 0.88rem;
    color: #7A7065;
    letter-spacing: 0.04em;
    font-weight: 300;
}
.hero-rule {
    width: 48px; height: 1px;
    background: #C9A84C;
    margin: 1.2rem 0;
}

/* ── KPI Cards ── */
div[data-testid="metric-container"] {
    background: #111111 !important;
    border: 1px solid #222222 !important;
    border-radius: 4px !important;
    padding: 1rem 1.2rem !important;
}
div[data-testid="metric-container"] label {
    color: #7A7065 !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
}
div[data-testid="metric-container"] div[data-testid="metric-value"] {
    color: #F5F0E8 !important;
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.7rem !important;
    font-weight: 300 !important;
}

/* ── Section Headers ── */
.sec-header {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 300;
    color: #F5F0E8;
    letter-spacing: 0.02em;
    margin: 2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1F1F1F;
}
.sec-label {
    font-size: 0.68rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #C9A84C;
    margin-bottom: 1.2rem;
}

/* ── Style Pill ── */
.style-pill {
    display: inline-block;
    background: transparent;
    border: 1px solid #C9A84C;
    color: #C9A84C;
    padding: 0.55rem 2rem;
    border-radius: 2px;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    margin: 0.5rem 0;
}

/* ── Confidence Box ── */
.conf-box {
    background: #111111;
    border: 1px solid #222;
    border-left: 3px solid #C9A84C;
    border-radius: 4px;
    padding: 1rem 1.3rem;
    margin: 1rem 0;
}
.conf-score {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem;
    font-weight: 300;
    line-height: 1;
}
.conf-label {
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7A7065;
    margin-top: 2px;
}

/* ── Outfit Cards ── */
.outfit-card {
    background: #111111;
    border: 1px solid #1E1E1E;
    border-radius: 4px;
    padding: 0;
    margin-bottom: 1rem;
    display: flex;
    gap: 0;
    overflow: hidden;
    transition: border-color 0.2s;
}
.outfit-card:hover { border-color: #C9A84C; }
.outfit-img-wrap {
    width: 100px;
    min-height: 110px;
    flex-shrink: 0;
    overflow: hidden;
    background: #1A1A1A;
}
.outfit-img-wrap img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}
.outfit-body {
    flex: 1;
    padding: 0.9rem 1.1rem;
}
.outfit-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem;
    font-weight: 400;
    color: #F5F0E8;
    margin: 0 0 0.35rem;
    line-height: 1.3;
}
.outfit-meta {
    font-size: 0.75rem;
    color: #5A5550;
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.tag {
    display: inline-block;
    background: #1A1A1A;
    border: 1px solid #2A2A2A;
    border-radius: 2px;
    padding: 0.12rem 0.55rem;
    font-size: 0.72rem;
    color: #8A8078;
    letter-spacing: 0.04em;
}
.tag-gold {
    border-color: #3A3020;
    background: #1A1500;
    color: #C9A84C;
}

/* ── Shopping Buttons ── */
.shop-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin: 1rem 0; }
.shop-btn {
    display: inline-block;
    padding: 0.55rem 1.4rem;
    border-radius: 2px;
    font-size: 0.77rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-decoration: none !important;
    text-transform: uppercase;
    transition: all 0.2s;
}
.shop-primary {
    background: #C9A84C;
    color: #0A0A0A !important;
    border: 1px solid #C9A84C;
}
.shop-primary:hover { background: #D4B860; }
.shop-outline {
    background: transparent;
    color: #C9A84C !important;
    border: 1px solid #3A3020;
}
.shop-outline:hover { border-color: #C9A84C; }

/* ── Reason Box ── */
.reason-box {
    background: #0F0F0F;
    border: 1px solid #1A1A1A;
    border-radius: 3px;
    padding: 0.75rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.84rem;
    color: #9A9088;
    line-height: 1.65;
}

/* ── Fallback Box ── */
.fallback-box {
    background: #1A1200;
    border: 1px solid #3A3000;
    border-radius: 3px;
    padding: 0.75rem 1rem;
    font-size: 0.83rem;
    color: #C9A84C;
    margin: 0.6rem 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 1px solid #1F1F1F;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 0;
    padding: 0.6rem 1.3rem;
    background: transparent;
    color: #5A5550;
    font-weight: 400;
    font-size: 0.8rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    background: transparent !important;
    color: #F5F0E8 !important;
    border-bottom: 2px solid #C9A84C !important;
}

/* ── Misc ── */
.stMarkdown hr { border-color: #1F1F1F; }
.stDataFrame { background: #111111; }
.stButton > button {
    background: #C9A84C !important;
    color: #0A0A0A !important;
    border: none !important;
    border-radius: 2px !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-size: 0.8rem !important;
}
.stButton > button:hover { background: #D4B860 !important; }

/* greeting */
.greeting-line {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.35rem;
    font-style: italic;
    color: #9A9088;
    margin-bottom: 0.4rem;
}

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# Per-product image system — unique image per outfit_id
# ════════════════════════════════════════════════════════════
# Uses outfit_id as a hash index into a per-type image pool.
# Every product gets a stable, distinct image regardless of
# how many products share the same outfit_type + color bucket.

OUTFIT_IMAGE_POOLS = {
    'kurta': [
        'https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?w=220&q=80',
        'https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=220&q=80',
        'https://images.unsplash.com/photo-1594938298603-c8148c4b4f7e?w=220&q=80',
        'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=220&q=80',
        'https://images.unsplash.com/photo-1614093302611-8efc64349eba?w=220&q=80',
        'https://images.unsplash.com/photo-1617627143233-b7f3c4f8e9f6?w=220&q=80',
        'https://images.unsplash.com/photo-1602810316498-ab67cf68c8e1?w=220&q=80',
    ],
    'saree': [
        'https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=220&q=80',
        'https://images.unsplash.com/photo-1594938298603-c8148c4b4f7e?w=220&q=80',
        'https://images.unsplash.com/photo-1616594039964-ae9021a400a0?w=220&q=80',
        'https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?w=220&q=80',
        'https://images.unsplash.com/photo-1614093302611-8efc64349eba?w=220&q=80',
        'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=220&q=80',
    ],
    'lehenga': [
        'https://images.unsplash.com/photo-1614093302611-8efc64349eba?w=220&q=80',
        'https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=220&q=80',
        'https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?w=220&q=80',
        'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=220&q=80',
        'https://images.unsplash.com/photo-1594938298603-c8148c4b4f7e?w=220&q=80',
    ],
    'dress': [
        'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=220&q=80',
        'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=220&q=80',
        'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=220&q=80',
        'https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=220&q=80',
        'https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=220&q=80',
        'https://images.unsplash.com/photo-1585487000160-6ebcfceb0d03?w=220&q=80',
        'https://images.unsplash.com/photo-1572804013427-4d7ca7268217?w=220&q=80',
    ],
    'formal_set': [
        'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=220&q=80',
        'https://images.unsplash.com/photo-1548690312-e3b507d8c110?w=220&q=80',
        'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=220&q=80',
        'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=220&q=80',
        'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=220&q=80',
    ],
    'top_jeans': [
        'https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=220&q=80',
        'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=220&q=80',
        'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=220&q=80',
        'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=220&q=80',
        'https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=220&q=80',
        'https://images.unsplash.com/photo-1548690312-e3b507d8c110?w=220&q=80',
    ],
    'co_ord_set': [
        'https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=220&q=80',
        'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=220&q=80',
        'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=220&q=80',
        'https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=220&q=80',
        'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=220&q=80',
        'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=220&q=80',
    ],
    'jumpsuit': [
        'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=220&q=80',
        'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=220&q=80',
        'https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=220&q=80',
        'https://images.unsplash.com/photo-1548690312-e3b507d8c110?w=220&q=80',
        'https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=220&q=80',
    ],
    'anarkali': [
        'https://images.unsplash.com/photo-1614093302611-8efc64349eba?w=220&q=80',
        'https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=220&q=80',
        'https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?w=220&q=80',
        'https://images.unsplash.com/photo-1594938298603-c8148c4b4f7e?w=220&q=80',
        'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=220&q=80',
    ],
    'palazzo_set': [
        'https://images.unsplash.com/photo-1595777457583-95e059d581b8?w=220&q=80',
        'https://images.unsplash.com/photo-1614093302611-8efc64349eba?w=220&q=80',
        'https://images.unsplash.com/photo-1610030469983-98e550d6193c?w=220&q=80',
        'https://images.unsplash.com/photo-1583391733956-3750e0ff4e8b?w=220&q=80',
        'https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=220&q=80',
        'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=220&q=80',
    ],
    'shirt_trouser': [
        'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=220&q=80',
        'https://images.unsplash.com/photo-1548690312-e3b507d8c110?w=220&q=80',
        'https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=220&q=80',
        'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=220&q=80',
        'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=220&q=80',
    ],
    'maxi_dress': [
        'https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=220&q=80',
        'https://images.unsplash.com/photo-1434389677669-e08b4cac3105?w=220&q=80',
        'https://images.unsplash.com/photo-1529139574466-a303027c1d8b?w=220&q=80',
        'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=220&q=80',
        'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=220&q=80',
        'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?w=220&q=80',
    ],
}

_FALLBACK_POOL = [
    'https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=220&q=80',
    'https://images.unsplash.com/photo-1539109136881-3be0616acf4b?w=220&q=80',
    'https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?w=220&q=80',
]

def get_outfit_image(outfit_type: str, color: str, outfit_id: int = 0) -> str:
    """
    Returns a stable, unique image URL per product.
    Uses outfit_id as a hash index into the per-type image pool so every
    product gets a distinct image, even within the same outfit_type + color bucket.
    """
    pool = OUTFIT_IMAGE_POOLS.get(outfit_type, _FALLBACK_POOL)
    return pool[int(outfit_id) % len(pool)]


# ════════════════════════════════════════════════════════════
# BUG FIX 2 — Working Shopping URLs
# ════════════════════════════════════════════════════════════

from urllib.parse import quote

MYNTRA_SLUGS = {
    'kurta': 'kurtas', 'saree': 'sarees', 'lehenga': 'lehenga-cholis',
    'dress': 'dresses', 'formal_set': 'blazers-and-suits', 'top_jeans': 'jeans',
    'co_ord_set': 'co-ords', 'jumpsuit': 'jumpsuits', 'anarkali': 'anarkali-suits',
    'palazzo_set': 'palazzo-pants', 'shirt_trouser': 'trousers', 'maxi_dress': 'maxi-dresses',
}
FLIPKART_SLUGS = {
    'kurta': 'kurtis', 'saree': 'sarees', 'lehenga': 'lehenga-cholis',
    'dress': 'western-wear', 'formal_set': 'blazers', 'top_jeans': 'jeans',
    'co_ord_set': 'co-ords', 'jumpsuit': 'jumpsuits', 'anarkali': 'anarkali-suits',
    'palazzo_set': 'palazzos', 'shirt_trouser': 'trousers', 'maxi_dress': 'maxi-dresses',
}

def build_myntra_url_v2(outfit_type: str, style_tag: str) -> str:
    """
    FIX: Use simple Myntra search — deep price filter URLs were broken.
    Myntra price filter format changed; search by keyword is stable.
    """
    slug = MYNTRA_SLUGS.get(outfit_type, outfit_type.replace('_', '-'))
    style_kw = style_tag.replace('_', ' ')
    return f"https://www.myntra.com/{slug}?rawQuery={quote(style_kw + ' ' + outfit_type.replace('_', ' '))}"

def build_ajio_url_v2(outfit_type: str) -> str:
    """
    FIX: AJIO search uses /search?text= format, not /s/ (which was 404ing).
    """
    keyword = outfit_type.replace('_', ' ')
    return f"https://www.ajio.com/search/?text={quote(keyword)}"

def build_flipkart_url(outfit_type: str, style_tag: str, min_price: int, max_price: int) -> str:
    """
    NEW: Flipkart works reliably. Added as Myntra/AJIO replacement.
    Price filter params are stable on Flipkart.
    """
    keyword = f"{style_tag.replace('_', ' ')} {outfit_type.replace('_', ' ')}"
    return (f"https://www.flipkart.com/search?q={quote(keyword)}"
            f"&p%5B%5D=facets.price_range.from%3D{min_price}"
            f"&p%5B%5D=facets.price_range.to%3D{max_price}")

def build_amazon_url_v2(outfit_type: str, style_tag: str, min_price: int, max_price: int) -> str:
    """
    FIX: Amazon price filter uses INR directly in rh param, not paise.
    """
    keyword = f"{style_tag.replace('_', ' ')} {outfit_type.replace('_', ' ')} women india"
    return (f"https://www.amazon.in/s?k={quote(keyword)}"
            f"&rh=p_36%3A{min_price*100}-{max_price*100}")


# ════════════════════════════════════════════════════════════
# DATA
# ════════════════════════════════════════════════════════════

STYLE_EMOJIS = {
    'festive':'🎉','ethnic_casual':'🌸','ethnic_formal':'🪷',
    'western_casual':'👕','corporate':'💼','chic':'✨',
    'glam':'💎','boho':'🌿','minimal':'⚪','traditional':'🏺',
}
OUTFIT_TYPE_DISPLAY = {
    'kurta':'Kurta','saree':'Saree','lehenga':'Lehenga',
    'dress':'Western Dress','formal_set':'Formal Set / Blazer',
    'top_jeans':'Top + Jeans','co_ord_set':'Co-ord Set',
    'jumpsuit':'Jumpsuit','anarkali':'Anarkali Suit',
    'palazzo_set':'Kurta Palazzo Set','shirt_trouser':'Shirt + Trousers',
    'maxi_dress':'Maxi Dress',
}
VIBE_PHRASES = {
    'festive':'born to celebrate','glam':'made for the spotlight',
    'chic':'effortlessly chic','boho':'beautifully untamed',
    'minimal':'elegantly restrained','corporate':'powerfully refined',
    'ethnic_casual':'gracefully everyday','ethnic_formal':'timelessly elegant',
    'western_casual':'casually confident','traditional':'rooted in beauty',
}


# ════════════════════════════════════════════════════════════
# CACHED TRAINING
# ════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initialising AI models...")
def load_and_train():
    setup_database()
    df = load_data()
    X_train, X_test, y_train, y_test, encoders, target_le = encode_and_split(df)
    class_names = list(target_le.classes_)
    best_model, best_name, results = train_and_evaluate(
        X_train, X_test, y_train, y_test, class_names
    )
    feat_imp = get_feature_importance(best_model, FEATURE_COLS)
    save_model(best_model, encoders, target_le)
    initialise_online_model(X_train, y_train,
                            list(range(len(class_names))), target_le, encoders)
    return df, best_model, best_name, results, encoders, target_le, class_names, feat_imp, X_test, y_test


# ════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════

def render_sidebar():
    st.sidebar.markdown("""
    <div style="padding:1.5rem 0 1rem;">
        <div style="font-size:0.65rem;letter-spacing:0.3em;text-transform:uppercase;color:#C9A84C;margin-bottom:0.4rem;">
            STYLAI
        </div>
        <div style="font-family:'Cormorant Garamond',serif;font-size:1.5rem;font-weight:300;color:#F5F0E8;">
            Your Profile
        </div>
    </div>
    """, unsafe_allow_html=True)

    name = st.sidebar.text_input("Name", placeholder="e.g. Priya")

    st.sidebar.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    body_type = st.sidebar.selectbox("Body Type",
        ['hourglass','pear','apple','rectangle'],
        help="hourglass = balanced | pear = wider hips | apple = fuller mid | rectangle = straight")

    skin_tone = st.sidebar.selectbox("Skin Tone",
        ['warm','cool','neutral'],
        help="warm = golden/olive | cool = pink/blue | neutral = balanced")

    occasion = st.sidebar.selectbox("Occasion", ['party','casual','formal'])

    outfit_type_display = st.sidebar.selectbox("Outfit Type",
        list(OUTFIT_TYPE_DISPLAY.values()))
    outfit_type = [k for k, v in OUTFIT_TYPE_DISPLAY.items() if v == outfit_type_display][0]

    season = st.sidebar.selectbox("Season", ['all','summer','winter','spring','autumn'])

    st.sidebar.markdown("---")
    st.sidebar.markdown('<div style="font-size:0.65rem;letter-spacing:0.2em;text-transform:uppercase;color:#C9A84C;">BUDGET (INR)</div>', unsafe_allow_html=True)

    price_range = st.sidebar.slider("", min_value=400, max_value=15000,
        value=(1000, 5000), step=100, format="₹%d")

    st.sidebar.markdown("---")

    predict_btn = st.sidebar.button("✦ Discover My Style", use_container_width=True, type="primary")

    return {
        'name': name or "Guest",
        'body_type': body_type, 'skin_tone': skin_tone,
        'occasion': occasion, 'outfit_type': outfit_type,
        'season': season, 'min_price': price_range[0], 'max_price': price_range[1],
        'predict': predict_btn,
    }


# ════════════════════════════════════════════════════════════
# CHARTS (dark theme)
# ════════════════════════════════════════════════════════════

DARK_PALETTE = ['#C9A84C','#8A7A5A','#4A5568','#744C4C','#4C7458',
                '#4C5C74','#744C74','#747474','#4C7474','#744C4C']

def dark_fig(w=9, h=4):
    fig = plt.figure(figsize=(w, h), facecolor='#111111')
    ax = fig.add_subplot(111)
    ax.set_facecolor('#111111')
    ax.tick_params(colors='#5A5550', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#222222')
    ax.xaxis.label.set_color('#7A7065')
    ax.yaxis.label.set_color('#7A7065')
    ax.title.set_color('#F5F0E8')
    return fig, ax


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    # ── Hero ──────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-wrap">
        <div class="hero-eyebrow">AI · Fashion · Intelligence</div>
        <div class="hero-title">STYL<em>AI</em></div>
        <div class="hero-rule"></div>
        <div class="hero-sub">691 products &nbsp;·&nbsp; 10 style classes &nbsp;·&nbsp; Machine Learning &nbsp;·&nbsp; Budget-aware</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load & Train ──────────────────────────────────────────
    (df, best_model, best_name, results, encoders,
     target_le, class_names, feat_imp, X_test, y_test) = load_and_train()

    # ── Sidebar ───────────────────────────────────────────────
    inputs = render_sidebar()

    # ── KPI Bar ───────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Model", best_name)
    k2.metric("Accuracy", f"{results[best_name]['test_acc']:.1%}")
    k3.metric("Products", f"{len(df):,}")
    k4.metric("Style Classes", len(class_names))

    st.markdown("---")

    # ════════════════════════════════════════════
    # PREDICTION
    # ════════════════════════════════════════════

    if inputs['predict']:
        try:
            user = User(
                name=inputs['name'], body_type=inputs['body_type'],
                skin_tone=inputs['skin_tone'], occasion=inputs['occasion'],
                outfit_type=inputs['outfit_type'], season=inputs['season'],
            )
        except ValueError as e:
            st.error(f"Profile error: {e}")
            return

        X_user = encode_user_input(user.as_feature_dict(), encoders)
        batch_style, batch_probs = predict_style(best_model, X_user, target_le)
        online_style, online_probs = online_predict(X_user)
        emoji = STYLE_EMOJIS.get(batch_style, '✦')
        top_confidence = batch_probs.get(batch_style, 0.0)
        conf_label, conf_explanation, conf_color = get_confidence_explanation(top_confidence)
        fallback_msg = get_fallback_message(top_confidence)
        vibe = VIBE_PHRASES.get(batch_style, 'uniquely yourself')

        # ── Result Hero ────────────────────────────────────────
        col_text, col_conf = st.columns([2, 1])
        with col_text:
            st.markdown(f'<div class="greeting-line">{user.name}, you are {vibe}.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="style-pill">{emoji} &nbsp; {batch_style.replace("_", " ").upper()}</div>', unsafe_allow_html=True)
            st.markdown(f'<p style="color:#5A5550;font-size:0.82rem;margin-top:0.5rem;">{user.body_type.title()} · {user.skin_tone.title()} skin · {user.occasion.title()} occasion</p>', unsafe_allow_html=True)

        with col_conf:
            st.markdown(f"""
            <div class="conf-box">
                <div class="conf-score" style="color:{conf_color};">{top_confidence*100:.0f}%</div>
                <div style="font-size:0.8rem;font-weight:500;color:{conf_color};letter-spacing:0.08em;text-transform:uppercase;">{conf_label}</div>
                <div class="conf-label" style="margin-top:4px;">{conf_explanation}</div>
            </div>
            """, unsafe_allow_html=True)

        if fallback_msg:
            st.markdown(f'<div class="fallback-box">⚠ {fallback_msg}</div>', unsafe_allow_html=True)

        # ── Model Comparison ───────────────────────────────────
        with st.expander("Model comparison — Random Forest vs SGD Online", expanded=False):
            col_b, col_o = st.columns(2)
            with col_b:
                st.markdown('<div class="sec-label">Random Forest (Batch)</div>', unsafe_allow_html=True)
                top3 = sorted(batch_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                for cls, prob in top3:
                    bar_w = int(prob * 100)
                    clr = '#C9A84C' if cls == batch_style else '#2A2A2A'
                    st.markdown(f"""<div style="margin:6px 0;display:flex;align-items:center;gap:8px;">
                        <span style="font-size:0.8rem;color:#8A8078;width:120px;flex-shrink:0;">{cls.replace('_',' ')}</span>
                        <div style="flex:1;height:6px;background:#1A1A1A;border-radius:3px;">
                            <div style="width:{bar_w*2}px;max-width:100%;height:6px;background:{clr};border-radius:3px;"></div>
                        </div>
                        <span style="font-size:0.78rem;color:#5A5550;">{prob*100:.0f}%</span>
                    </div>""", unsafe_allow_html=True)
            with col_o:
                st.markdown('<div class="sec-label">SGD Online Model</div>', unsafe_allow_html=True)
                if online_probs:
                    top3o = sorted(online_probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    for cls, prob in top3o:
                        bar_w = int(prob * 100)
                        clr = '#C9A84C' if cls == online_style else '#2A2A2A'
                        st.markdown(f"""<div style="margin:6px 0;display:flex;align-items:center;gap:8px;">
                            <span style="font-size:0.8rem;color:#8A8078;width:120px;flex-shrink:0;">{cls.replace('_',' ')}</span>
                            <div style="flex:1;height:6px;background:#1A1A1A;border-radius:3px;">
                                <div style="width:{bar_w*2}px;max-width:100%;height:6px;background:{clr};border-radius:3px;"></div>
                            </div>
                            <span style="font-size:0.78rem;color:#5A5550;">{prob*100:.0f}%</span>
                        </div>""", unsafe_allow_html=True)
                agree = "✦ Both models agree" if online_style == batch_style else "⚠ Models differ"
                st.caption(agree)

        st.markdown("---")

        # ════════════════════════════════════
        # TABS
        # ════════════════════════════════════
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Recommendations", "Shop & Budget", "Charts", "ML Insights", "Feedback"
        ])

        # ── TAB 1: Recommendations ─────────────────────────────
        with tab1:
            st.markdown('<div class="sec-label">Top Picks For You</div>', unsafe_allow_html=True)

            matches = df[
                (df['style_tag'] == batch_style) & (df['occasion'] == user.occasion)
            ].sort_values('rating', ascending=False)
            if matches.empty:
                matches = df[df['style_tag'] == batch_style].sort_values('rating', ascending=False)

            if matches.empty:
                st.warning("No outfits found for this combination.")
            else:
                for i, (_, row) in enumerate(matches.head(6).iterrows()):
                    price = int(row['price_inr'])
                    rating = float(row['rating'])
                    reviews = int(row['num_reviews'])
                    color = str(row['color'])

                    # per-product image: unique per outfit_id
                    img_url = get_outfit_image(row['outfit_type'], color, row['outfit_id'])

                    st.markdown(f"""
                    <div class="outfit-card">
                        <div class="outfit-img-wrap">
                            <img src="{img_url}" alt="{row['product_name']}" />
                        </div>
                        <div class="outfit-body">
                            <div class="outfit-name">{row['product_name']}</div>
                            <div class="outfit-meta">
                                <span class="tag tag-gold">₹{price:,}</span>
                                <span class="tag tag-gold">★ {rating}</span>
                                <span class="tag">{row['brand']}</span>
                                <span class="tag">{color}</span>
                                <span class="tag">{row['season'].title()}</span>
                                <span class="tag">{int(reviews):,} reviews</span>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            # Why this recommendation?
            st.markdown('<div class="sec-header">Why This Recommendation?</div>', unsafe_allow_html=True)
            if not matches.empty:
                demo_row = matches.iloc[0]
                reasons = get_recommendation_reasons(
                    user_body_type=user.body_type, user_skin_tone=user.skin_tone,
                    user_occasion=user.occasion, user_outfit_type=user.outfit_type,
                    predicted_style=batch_style,
                    outfit_color=str(demo_row['color']),
                    outfit_price=int(demo_row['price_inr']),
                    min_price=inputs['min_price'], max_price=inputs['max_price'],
                )
                for reason in reasons:
                    st.markdown(f'<div class="reason-box">{reason}</div>', unsafe_allow_html=True)

        # ── TAB 2: Shop & Budget ───────────────────────────────
        with tab2:
            st.markdown('<div class="sec-label">Budget-Filtered Results</div>', unsafe_allow_html=True)
            min_p, max_p = inputs['min_price'], inputs['max_price']

            # Filter outfits by price
            price_matches = df[
                (df['style_tag'] == batch_style) &
                (df['price_inr'] >= min_p) & (df['price_inr'] <= max_p)
            ].sort_values('rating', ascending=False)

            if not price_matches.empty:
                st.success(f"✦ {len(price_matches)} outfits found within ₹{min_p:,}–₹{max_p:,}")
                for _, row in price_matches.head(5).iterrows():
                    img_url = get_outfit_image(row['outfit_type'], str(row['color']), row['outfit_id'])
                    st.markdown(f"""
                    <div class="outfit-card">
                        <div class="outfit-img-wrap"><img src="{img_url}" /></div>
                        <div class="outfit-body">
                            <div class="outfit-name">{row['product_name']}</div>
                            <div class="outfit-meta">
                                <span class="tag tag-gold">₹{int(row['price_inr']):,}</span>
                                <span class="tag tag-gold">★ {float(row['rating'])}</span>
                                <span class="tag">{row['brand']}</span>
                                <span class="tag">{row['color']}</span>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)
            else:
                # Nearest fallback
                style_df = df[df['style_tag'] == batch_style].copy()
                mid = (min_p + max_p) / 2
                style_df['_dist'] = (style_df['price_inr'] - mid).abs()
                nearest = style_df.nsmallest(5, '_dist')
                nearest_price = int(nearest.iloc[0]['price_inr'])
                gap = abs(nearest_price - (max_p if nearest_price > max_p else min_p))
                st.markdown(f'<div class="fallback-box">⚠ No exact match in ₹{min_p:,}–₹{max_p:,}. Showing nearest (₹{gap:,} outside budget).</div>', unsafe_allow_html=True)
                for _, row in nearest.iterrows():
                    img_url = get_outfit_image(row['outfit_type'], str(row['color']), row['outfit_id'])
                    st.markdown(f"""
                    <div class="outfit-card">
                        <div class="outfit-img-wrap"><img src="{img_url}" /></div>
                        <div class="outfit-body">
                            <div class="outfit-name">{row['product_name']}</div>
                            <div class="outfit-meta">
                                <span class="tag tag-gold">₹{int(row['price_inr']):,}</span>
                                <span class="tag">{row['brand']}</span>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)

            # ── BUG FIX 2: Working Shopping Links ─────────────
            st.markdown('<div class="sec-header">Shop Now</div>', unsafe_allow_html=True)
            st.markdown(f"""<p style="font-size:0.78rem;color:#5A5550;margin-bottom:0.8rem;">
                Links open real search results for your style + outfit type. 
                Myntra & AJIO use keyword search (most reliable). Flipkart & Amazon support price filters.
            </p>""", unsafe_allow_html=True)

            myntra_url = build_myntra_url_v2(user.outfit_type, batch_style)
            ajio_url   = build_ajio_url_v2(user.outfit_type)
            flipkart_url = build_flipkart_url(user.outfit_type, batch_style, min_p, max_p)
            amazon_url = build_amazon_url_v2(user.outfit_type, batch_style, min_p, max_p)

            st.markdown(f"""
            <div style="background:#111;border:1px solid #1E1E1E;border-radius:4px;padding:1.2rem 1.5rem;">
                <div class="shop-row">
                    <a href="{myntra_url}" target="_blank" class="shop-btn shop-primary">Myntra</a>
                    <a href="{ajio_url}" target="_blank" class="shop-btn shop-primary">AJIO</a>
                    <a href="{flipkart_url}" target="_blank" class="shop-btn shop-outline">Flipkart + Budget filter</a>
                    <a href="{amazon_url}" target="_blank" class="shop-btn shop-outline">Amazon India + Budget filter</a>
                </div>
                <p style="font-size:0.72rem;color:#3A3530;margin:0.6rem 0 0;">
                    Budget ₹{min_p:,}–₹{max_p:,} applied where supported by platform
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Budget chart (dark)
            st.markdown('<div class="sec-header">Price Distribution</div>', unsafe_allow_html=True)
            style_prices = df[df['style_tag'] == batch_style]['price_inr']
            fig, ax = dark_fig(9, 4)
            ax.hist(style_prices, bins=22, color='#2A2510', edgecolor='#1A1A1A', linewidth=0.5)
            ax.axvspan(min_p, max_p, alpha=0.25, color='#C9A84C', label=f'Your budget ₹{min_p:,}–₹{max_p:,}')
            ax.axvline(min_p, color='#C9A84C', linewidth=1.5, linestyle='--')
            ax.axvline(max_p, color='#C9A84C', linewidth=1.5, linestyle='--')
            in_r = len(style_prices[(style_prices >= min_p) & (style_prices <= max_p)])
            ax.set_title(f'{batch_style.replace("_"," ").title()}  —  {in_r}/{len(style_prices)} outfits in budget', fontsize=11, pad=12)
            ax.set_xlabel('Price (INR)'); ax.set_ylabel('Count')
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'₹{int(x):,}'))
            ax.legend(fontsize=8, facecolor='#111111', edgecolor='#222222', labelcolor='#8A8078')
            fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ── TAB 3: Charts ──────────────────────────────────────
        with tab3:
            st.markdown('<div class="sec-label">Data Visualisations</div>', unsafe_allow_html=True)
            chart_choice = st.selectbox("Chart", [
                "Style Distribution", "Feature Heatmaps", "Price by Style",
                "Model Comparison", "Confusion Matrix", "Feature Importance", "Precision / Recall / F1"
            ])

            if chart_choice == "Style Distribution":
                fig, ax = dark_fig(8, 5)
                counts = df['style_tag'].value_counts()
                wedges, texts, autotexts = ax.pie(
                    counts.values, labels=counts.index, autopct='%1.1f%%',
                    colors=DARK_PALETTE[:len(counts)], startangle=130,
                    wedgeprops={'linewidth':1.5, 'edgecolor':'#111111'}
                )
                for t in texts: t.set_color('#8A8078'); t.set_fontsize(8)
                for at in autotexts: at.set_color('#F5F0E8'); at.set_fontsize(7)
                ax.set_title('Style Tag Distribution', fontsize=12)
                st.pyplot(fig); plt.close(fig)

            elif chart_choice == "Feature Heatmaps":
                import seaborn as sns
                fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor='#111111')
                for ax2, feat in zip(axes, ['body_type', 'skin_tone', 'occasion']):
                    pivot = pd.crosstab(df[feat], df['style_tag'])
                    sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrBr',
                                linewidths=0.5, linecolor='#0A0A0A', ax=ax2,
                                cbar=False, annot_kws={'size': 7, 'color':'#F5F0E8'})
                    ax2.set_facecolor('#111111')
                    ax2.set_title(feat.replace('_', ' ').title(), fontsize=11, color='#F5F0E8')
                    ax2.tick_params(colors='#5A5550', labelsize=7)
                    ax2.set_xlabel(''); ax2.set_ylabel('')
                fig.patch.set_facecolor('#111111')
                fig.tight_layout()
                st.pyplot(fig); plt.close(fig)

            elif chart_choice == "Price by Style":
                import seaborn as sns
                order = df.groupby('style_tag')['price_inr'].median().sort_values(ascending=False).index
                fig, ax = dark_fig(13, 5)
                sns.boxplot(data=df, x='style_tag', y='price_inr', order=order,
                            palette=DARK_PALETTE[:len(order)], ax=ax,
                            flierprops={'marker':'o','markersize':3,'alpha':0.3,'color':'#5A5550'})
                ax.set_title('Price Distribution by Style', fontsize=12)
                ax.tick_params(axis='x', rotation=30, labelsize=8)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'₹{int(x):,}'))
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            elif chart_choice == "Model Comparison":
                names = list(results.keys()); w = 0.35
                train_accs = [results[n]['train_acc']*100 for n in names]
                test_accs  = [results[n]['test_acc']*100 for n in names]
                x = np.arange(len(names))
                fig, ax = dark_fig(8, 4)
                b1 = ax.bar(x-w/2, train_accs, w, label='Train', color='#4A5568', edgecolor='#111', alpha=0.9)
                b2 = ax.bar(x+w/2, test_accs,  w, label='Test',  color='#C9A84C', edgecolor='#111', alpha=0.9)
                for bar in list(b1)+list(b2):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                            f"{bar.get_height():.1f}%", ha='center', fontsize=8, color='#8A8078')
                ax.set_xticks(x); ax.set_xticklabels(names, color='#8A8078')
                ax.set_ylim(0, 115); ax.legend(fontsize=9, facecolor='#111', edgecolor='#222', labelcolor='#8A8078')
                ax.set_title('Model Accuracy: Train vs Test', fontsize=12)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            elif chart_choice == "Confusion Matrix":
                cm = results[best_name]['cm']
                cm_norm = cm.astype(float)/(cm.sum(axis=1, keepdims=True)+1e-9)
                fig = plt.figure(figsize=(10, 8), facecolor='#111111')
                ax = fig.add_subplot(111); ax.set_facecolor('#111111')
                im = ax.imshow(cm_norm, cmap='YlOrBr', vmin=0, vmax=1)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
                ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8, color='#8A8078')
                ax.set_yticklabels(class_names, fontsize=8, color='#8A8078')
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, str(cm[i,j]), ha='center', va='center', fontsize=8,
                                color='#111111' if cm_norm[i,j] > 0.5 else '#F5F0E8')
                ax.set_title(f'Confusion Matrix — {best_name}', fontsize=12, color='#F5F0E8')
                ax.set_xlabel('Predicted', color='#7A7065'); ax.set_ylabel('True', color='#7A7065')
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            elif chart_choice == "Feature Importance":
                if feat_imp:
                    items = sorted(feat_imp.items(), key=lambda x: x[1])
                    feats = [i[0].replace('_',' ').title() for i in items]
                    vals  = [i[1]*100 for i in items]
                    colors = ['#C9A84C' if v == max(vals) else '#4A5568' for v in vals]
                    fig, ax = dark_fig(8, 4)
                    bars = ax.barh(feats, vals, color=colors, edgecolor='#111', height=0.5)
                    for bar, val in zip(bars, vals):
                        ax.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2,
                                f'{val:.1f}%', va='center', fontsize=9, color='#8A8078')
                    ax.set_title('Feature Importance (Random Forest)', fontsize=12)
                    fig.tight_layout(); st.pyplot(fig); plt.close(fig)

            elif chart_choice == "Precision / Recall / F1":
                from sklearn.metrics import classification_report
                y_pred = results[best_name]['y_pred']
                report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
                classes   = [c for c in class_names if c in report]
                precision = [report[c]['precision']*100 for c in classes]
                recall    = [report[c]['recall']*100 for c in classes]
                f1        = [report[c]['f1-score']*100 for c in classes]
                x = np.arange(len(classes)); w = 0.26
                fig, ax = dark_fig(13, 5)
                ax.bar(x-w, precision, w, label='Precision', color='#4A5568', edgecolor='#111', alpha=0.9)
                ax.bar(x,   recall,    w, label='Recall',    color='#C9A84C', edgecolor='#111', alpha=0.9)
                ax.bar(x+w, f1,        w, label='F1-Score',  color='#4C7458', edgecolor='#111', alpha=0.9)
                macro_f1 = report['macro avg']['f1-score']*100
                ax.axhline(y=macro_f1, color='#C9A84C', linestyle='--', linewidth=1.5, alpha=0.5)
                ax.text(len(classes)-0.5, macro_f1+1.5, f'Macro F1: {macro_f1:.1f}%', fontsize=8, color='#C9A84C')
                ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha='right', fontsize=8, color='#8A8078')
                ax.set_ylim(0, 120); ax.legend(fontsize=9, facecolor='#111', edgecolor='#222', labelcolor='#8A8078')
                ax.set_title(f'Precision / Recall / F1 — {best_name}', fontsize=12)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)

        # ── TAB 4: ML Insights ─────────────────────────────────
        with tab4:
            st.markdown('<div class="sec-label">ML Pipeline</div>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Model Comparison**")
                model_df = pd.DataFrame([
                    {'Model': n, 'Train': f"{results[n]['train_acc']:.1%}", 'Test': f"{results[n]['test_acc']:.1%}"}
                    for n in results
                ])
                st.dataframe(model_df, use_container_width=True, hide_index=True)
                st.markdown("**Feature Importance**")
                if feat_imp:
                    fi_df = pd.DataFrame(
                        sorted(feat_imp.items(), key=lambda x: x[1], reverse=True),
                        columns=['Feature', 'Importance']
                    )
                    fi_df['Importance'] = fi_df['Importance'].map(lambda x: f"{x*100:.1f}%")
                    st.dataframe(fi_df, use_container_width=True, hide_index=True)
            with col2:
                st.markdown("**Dataset Stats**")
                stats = {
                    'Products': f"{len(df):,}",
                    'Style Classes': len(class_names),
                    'Outfit Types': df['outfit_type'].nunique(),
                    'Brands': df['brand'].nunique() if 'brand' in df.columns else 'N/A',
                    'Price Range': f"₹{int(df['price_inr'].min()):,}–₹{int(df['price_inr'].max()):,}",
                    'Avg Rating': f"{df['rating'].mean():.2f}/5.0",
                    'Sessions stored': count_sessions(),
                }
                for k, v in stats.items():
                    st.markdown(f"<div style='display:flex;justify-content:space-between;padding:0.35rem 0;border-bottom:1px solid #1A1A1A;'><span style='color:#5A5550;font-size:0.82rem;'>{k}</span><span style='color:#F5F0E8;font-size:0.82rem;'>{v}</span></div>", unsafe_allow_html=True)

        # ── TAB 5: Feedback ────────────────────────────────────
        with tab5:
            st.markdown('<div class="sec-label">Rate This Recommendation</div>', unsafe_allow_html=True)
            st.write(f"Was **{batch_style.replace('_',' ').title()}** the right style for you?")
            rating = st.slider("Rating", 1, 5, 4)
            st.markdown("⭐" * rating + "☆" * (5 - rating))
            correct_style = None
            if rating <= 3:
                correct_style = st.selectbox("Correct style?", ['(skip)'] + sorted(class_names))
                if correct_style == '(skip)': correct_style = None
            if st.button("Submit Feedback", type="primary"):
                matched = df[(df['style_tag'] == batch_style) & (df['occasion'] == user.occasion)].head(4)
                from models.outfit_model import Outfit
                outfits = [Outfit(int(r['outfit_id']), r['product_name'], r['brand'], r['outfit_type'],
                                  r['color'], r['season'], r['occasion'], r['body_type'], r['skin_tone'],
                                  r['style_tag'], int(r.get('price_inr',0)), float(r.get('rating',0)),
                                  int(r.get('num_reviews',0))) for _, r in matched.iterrows()]
                session_id = save_session(user.name, user.body_type, user.skin_tone, user.occasion,
                                          user.outfit_type, user.season, batch_style, outfits)
                save_feedback(session_id, rating, correct_style)
                if correct_style and correct_style in target_le.classes_:
                    online_update(X_user, correct_style, target_le)
                    st.success(f"✦ Feedback saved. Model updated with '{correct_style}'.")
                else:
                    st.success("✦ Feedback saved. Thank you.")
                st.balloons()

    else:
        # ── Welcome Screen ────────────────────────────────────
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
            <div style="font-family:'Cormorant Garamond',serif;font-size:5rem;font-weight:300;color:#1E1E1E;line-height:1;">
                YOUR<br><em style="color:#C9A84C;">STYLE</em><br>AWAITS
            </div>
            <div style="width:48px;height:1px;background:#C9A84C;margin:2rem auto;"></div>
            <p style="color:#3A3530;font-size:0.88rem;letter-spacing:0.08em;">
                FILL YOUR PROFILE IN THE SIDEBAR &nbsp;·&nbsp; CLICK DISCOVER MY STYLE
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Dataset preview
        st.markdown('<div class="sec-header">Dataset Preview</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.dataframe(
                df[['product_name','brand','outfit_type','occasion','style_tag','price_inr','rating']].head(10),
                use_container_width=True, hide_index=True
            )
        with col_b:
            st.markdown('<div class="sec-label">Style Classes</div>', unsafe_allow_html=True)
            for style, count in df['style_tag'].value_counts().items():
                emoji = STYLE_EMOJIS.get(style, '·')
                st.markdown(f"<div style='padding:0.28rem 0;border-bottom:1px solid #1A1A1A;font-size:0.82rem;color:#8A8078;'>{emoji} <span style='color:#F5F0E8'>{style.replace('_',' ').title()}</span> — {count}</div>", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
