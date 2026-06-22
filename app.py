# ============================================================
# app.py — STYLAI · Soft Step-by-Step Fashion Experience (v3)
# ============================================================
# UI-LAYER REWRITE ONLY — same ML pipeline, same rule-based
# explainer, same shopping-link logic as the original app.
#
# Changes in this version vs v2:
#  1. FIXED: form fields (radios/sliders) are now genuinely
#     inside the soft glass card — using st.container(key=...)
#     instead of raw HTML divs that Streamlit can't actually
#     nest native widgets inside of.
#  2. FIXED: outfit visuals are no longer hotlinked third-party
#     stock photos (which were both mismatched AND unreliable —
#     your CSV has no real product-image column). They are now
#     guaranteed-to-render soft gradient swatch cards using the
#     product's real color + a garment icon. Nothing can "break"
#     anymore because nothing is fetched from the internet.
#  3. FIXED: all four shop links (Myntra/AJIO/Flipkart/Amazon)
#     now explicitly scope the search to women's fashion, so
#     men's/kids' results shouldn't surface.
# ============================================================

import streamlit as st
from urllib.parse import quote

# ── Project imports (UNCHANGED — same ML + logic as before) ──
from outfit_model import User
from utils.data_handler import load_data, encode_and_split, encode_user_input, FEATURE_COLS
from utils.ml_engine import train_and_evaluate, get_feature_importance, save_model, predict_style
from utils.learning_engine import initialise_online_model
from utils.db_handler import setup_database
from utils.explainer import get_recommendation_reasons, get_confidence_explanation, get_fallback_message

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="STYLAI — Find Your Vibe",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════
# CUSTOM CSS — Soft Pastel / Pinterest / Glassmorphism
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;1,9..144,400;1,9..144,500&family=Quicksand:wght@400;500;600;700&display=swap');

:root {
  --rose: #FF7AAE; --rose-deep: #FF5C9A; --lavender: #B98CE8;
  --cream: #FFF8F0; --plum: #4A2E4A; --muted: #8C7A8C;
  --glass: rgba(255,255,255,0.55); --glass-strong: rgba(255,255,255,0.78);
  --glass-border: rgba(255,255,255,0.7);
}
html, body, .stApp { font-family: 'Quicksand', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #FFE3F0 0%, #F3E6FF 45%, #FFF6E8 100%) !important; background-attachment: fixed !important; }
section[data-testid="stSidebar"] { display: none !important; }
div[data-testid="collapsedControl"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stDecoration"] { display: none; }
.block-container { padding-top: 2.2rem; max-width: 800px; }

.progress-row { display:flex; justify-content:center; gap:0.5rem; margin-bottom: 1.6rem; }
.dot { width: 9px; height: 9px; border-radius: 50%; background: rgba(255,255,255,0.6); border:1px solid rgba(255,124,170,0.4); transition: all .25s; }
.dot.active { background: linear-gradient(135deg, var(--rose), var(--lavender)); width:22px; border-radius:5px; border:none; }
.dot.done { background: var(--rose); opacity: 0.55; }

/* ── REAL glass card: applied to an actual st.container(key=...), so it
      truly contains every widget placed inside the `with` block ── */
[class*="st-key-card"] {
  background: var(--glass) !important;
  backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
  border: 1px solid var(--glass-border) !important;
  border-radius: 28px !important;
  padding: 2.2rem 2rem !important;
  box-shadow: 0 12px 40px rgba(255, 124, 170, 0.18);
  margin-bottom: 1.4rem;
  animation: fadeInUp 0.35s ease;
}
@keyframes fadeInUp { from { opacity:0; transform: translateY(10px); } to { opacity:1; transform: translateY(0); } }

/* ── Ghost (back) button: also a real container, reliably targeted ── */
[class*="st-key-ghost"] button {
  background: transparent !important; color: var(--muted) !important;
  box-shadow: none !important; font-weight: 600 !important; padding: 0.4rem 1rem !important;
}

.eyebrow { font-size: 0.72rem; letter-spacing: 0.25em; text-transform: uppercase; color: var(--rose-deep); font-weight: 700; margin-bottom: 0.5rem; text-align:center; }
.headline { font-family: 'Fraunces', serif; font-weight: 500; font-size: 2.1rem; line-height: 1.18; color: var(--plum); margin: 0 0 0.5rem; text-align:center; }
.headline em { font-style: italic; color: var(--rose-deep); font-weight: 400; }
.subline { color: var(--muted); font-size: 0.92rem; margin-bottom: 1.2rem; text-align:center; }
.field-label { font-size: 0.78rem; font-weight: 700; color: var(--plum); letter-spacing: 0.05em; text-transform: uppercase; margin: 1.1rem 0 0.4rem; text-align:left; }

.stTextInput > div > div > input {
  background: var(--glass-strong) !important; border: 1px solid rgba(255,124,170,0.35) !important;
  border-radius: 16px !important; padding: 0.8rem 1.1rem !important; font-family: 'Quicksand', sans-serif !important;
  font-size: 1.05rem !important; color: var(--plum) !important; text-align: center; }

div[data-testid="stRadio"] > div { gap: 0.5rem; justify-content: center; flex-wrap: wrap; }
div[data-testid="stRadio"] label { background: var(--glass-strong) !important; border: 1px solid rgba(255,124,170,0.3) !important;
  border-radius: 999px !important; padding: 0.5rem 1.1rem !important; margin: 0.15rem !important;
  font-weight: 600 !important; color: var(--plum) !important; font-size: 0.92rem !important; transition: all .2s; }
div[data-testid="stRadio"] label:hover { border-color: var(--rose) !important; }
div[data-testid="stRadio"] label:has(input:checked) { background: linear-gradient(135deg, var(--rose), var(--lavender)) !important;
  color: white !important; border: none !important; box-shadow: 0 6px 18px rgba(255,124,170,0.35); }
div[data-testid="stRadio"] label > div:first-child { transform: scale(0.6); margin-right: 2px !important; opacity: 0.55; }

div[data-testid="stSlider"] { padding: 0 0.4rem; }

.stButton > button { background: linear-gradient(135deg, var(--rose), var(--lavender)) !important; color: white !important;
  border: none !important; border-radius: 999px !important; padding: 0.65rem 2.2rem !important; font-weight: 700 !important;
  font-family: 'Quicksand', sans-serif !important; box-shadow: 0 8px 22px rgba(255,124,170,0.32) !important; transition: transform .15s !important; }
.stButton > button:hover { transform: translateY(-2px); }

.vibe-pill { display:inline-block; background: linear-gradient(135deg, var(--rose), var(--lavender)); color: white;
  font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; font-size: 0.76rem; padding: 0.5rem 1.5rem;
  border-radius: 999px; margin-bottom: 0.7rem; }

.conf-card { background: var(--glass-strong); border-radius: 20px; padding: 1.1rem 1.3rem; text-align:center;
  border: 1px solid var(--glass-border); box-shadow: 0 6px 20px rgba(255,124,170,0.14); }
.conf-score { font-family: 'Fraunces', serif; font-size: 2.2rem; font-weight: 500; color: var(--rose-deep); line-height:1; }
.conf-tag { font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-top: 2px; }
.fallback-box { background: rgba(255,196,0,0.16); border: 1px solid rgba(255,196,0,0.35); border-radius: 14px;
  padding: 0.7rem 1rem; font-size: 0.85rem; color: #8a6a00; margin: 0.8rem 0; text-align:left; }

.pin-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1.1rem; margin: 0.6rem 0 1.2rem; }
.pin-card { background: var(--glass-strong); border: 1px solid var(--glass-border); border-radius: 22px; overflow: hidden;
  box-shadow: 0 8px 26px rgba(255,124,170,0.16); transition: transform .2s, box-shadow .2s; }
.pin-card:hover { transform: translateY(-4px); box-shadow: 0 14px 32px rgba(255,124,170,0.28); }
.pin-swatch { width: 100%; aspect-ratio: 4/3; display:flex; align-items:center; justify-content:center; font-size: 2.6rem; position:relative; }
.pin-swatch .color-dot { position:absolute; bottom:10px; right:12px; width:18px; height:18px; border-radius:50%; border: 2px solid rgba(255,255,255,0.85); box-shadow: 0 1px 4px rgba(0,0,0,0.2); }
.pin-body { padding: 0.9rem 1.05rem 1.05rem; text-align: left; }
.pin-title { font-family: 'Fraunces', serif; font-weight: 500; font-size: 0.98rem; color: var(--plum); margin: 0 0 0.3rem; line-height: 1.3; }
.pin-reason { font-size: 0.8rem; color: var(--muted); line-height: 1.45; margin-bottom: 0.5rem; }
.pin-tags { display:flex; gap:0.3rem; flex-wrap:wrap; }
.pin-tag { font-size: 0.68rem; font-weight: 700; color: var(--rose-deep); background: rgba(255,124,170,0.12); border-radius: 999px; padding: 0.16rem 0.55rem; }

.shop-row { display:flex; gap:0.6rem; flex-wrap:wrap; justify-content:center; margin-top: 0.6rem; }
.shop-btn { display:inline-block; padding: 0.6rem 1.4rem; border-radius: 999px; font-weight: 700; font-size: 0.85rem;
  text-decoration:none !important; transition: transform .15s; }
.shop-primary { background: linear-gradient(135deg, var(--rose), var(--lavender)); color: white !important;
  box-shadow: 0 6px 18px rgba(255,124,170,0.3); }
.shop-outline { background: var(--glass-strong); color: var(--plum) !important; border: 1px solid rgba(255,124,170,0.4); }
.shop-btn:hover { transform: translateY(-2px); }
.budget-note { font-size: 0.74rem; color: var(--muted); margin-top: 0.5rem; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# Guaranteed-reliable outfit visuals — gradient swatch + icon
# (No external image hosts. Nothing can ever fail to load.)
# ════════════════════════════════════════════════════════════
OUTFIT_ICON = {
    'kurta': '👗', 'saree': '🥻', 'lehenga': '👰', 'dress': '👗',
    'formal_set': '🤵', 'top_jeans': '👖', 'co_ord_set': '✨',
    'jumpsuit': '🎽', 'anarkali': '👗', 'palazzo_set': '👗',
    'shirt_trouser': '👔', 'maxi_dress': '👗',
}

COLOR_HEX = {
    'black': '#2B2B2B', 'brick red': '#A23B2E', 'burgundy': '#7B1E33', 'burnt orange': '#C1521B',
    'camel': '#C19A6B', 'champagne': '#F0DFC4', 'charcoal': '#3A3A3D', 'cobalt': '#1E4FA3',
    'coral': '#F26B6B', 'cream': '#FFF6E0', 'dark grey': '#4A4A4A', 'dusty rose': '#C98F94',
    'ecru': '#EDE3D0', 'emerald': '#2E8B6B', 'gold': '#C9A227', 'ink blue': '#1B2A57',
    'ivory': '#F6F1E6', 'khaki': '#9C8A5E', 'lavender': '#C8B6E2', 'light grey': '#B8B8B8',
    'lilac': '#CBA8D8', 'midnight navy': '#10193A', 'mustard': '#C9A227', 'nude': '#E3C2A5',
    'off-white': '#F4F1EA', 'olive': '#6B6B2A', 'peach': '#F4B79A', 'pearl': '#EDE7DD',
    'powder blue': '#A9C6E0', 'royal blue': '#1E3FA3', 'rust': '#A0522D', 'saffron': '#E8A33D',
    'sage green': '#9CAE8C', 'silver': '#C8C8C8', 'slate grey': '#6B7480', 'stone': '#B8AE9C',
    'teal': '#2E7C7C', 'terracotta': '#B5603F', 'tomato red': '#D1442E', 'turquoise': '#3FAFAE',
    'warm beige': '#D9C3A3', 'white': '#FAFAFA',
}
_DEFAULT_HEX = '#D9A6C2'

def _hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def get_outfit_visual(outfit_type: str, color: str):
    """Returns (gradient_css, icon, hex) for a guaranteed-to-render swatch card."""
    icon = OUTFIT_ICON.get(outfit_type, '🌸')
    hexcode = COLOR_HEX.get(color.strip().lower(), _DEFAULT_HEX)
    r, g, b = _hex_to_rgb(hexcode)
    # Lighten toward white for the gradient start so it always feels soft/pastel
    light = f"rgb({min(255, r+90)}, {min(255, g+90)}, {min(255, b+90)})"
    gradient = f"linear-gradient(160deg, {light} 0%, {hexcode} 100%)"
    return gradient, icon, hexcode

def render_swatch_html(outfit_type: str, color: str) -> str:
    gradient, icon, hexcode = get_outfit_visual(outfit_type, color)
    return f'<div class="pin-swatch" style="background:{gradient};"><span>{icon}</span><span class="color-dot" style="background:{hexcode};"></span></div>'

# ════════════════════════════════════════════════════════════
# Shopping link builders — women's fashion explicitly scoped
# ════════════════════════════════════════════════════════════
MYNTRA_SLUGS = {
    'kurta': 'women-kurtas', 'saree': 'women-sarees', 'lehenga': 'women-lehenga-cholis',
    'dress': 'women-dresses', 'formal_set': 'women-blazers', 'top_jeans': 'women-jeans',
    'co_ord_set': 'women-co-ords', 'jumpsuit': 'women-jumpsuits', 'anarkali': 'women-anarkali-suits',
    'palazzo_set': 'women-palazzos', 'shirt_trouser': 'women-trousers', 'maxi_dress': 'women-maxi-dresses',
}

def build_myntra_url_v2(outfit_type: str, style_tag: str) -> str:
    slug = MYNTRA_SLUGS.get(outfit_type, 'women-' + outfit_type.replace('_', '-'))
    style_kw = style_tag.replace('_', ' ')
    query = f"women {style_kw} {outfit_type.replace('_', ' ')}"
    return f"https://www.myntra.com/{slug}?rawQuery={quote(query)}"

def build_ajio_url_v2(outfit_type: str) -> str:
    keyword = f"women {outfit_type.replace('_', ' ')}"
    # AJIO category fnl filter restricts results to the Women department
    return f"https://www.ajio.com/search/?text={quote(keyword)}&fnl_gender=Women"

def build_flipkart_url(outfit_type: str, style_tag: str, min_price: int, max_price: int) -> str:
    keyword = f"women's {style_tag.replace('_', ' ')} {outfit_type.replace('_', ' ')}"
    return (f"https://www.flipkart.com/search?q={quote(keyword)}"
            f"&p%5B%5D=facets.price_range.from%3D{min_price}"
            f"&p%5B%5D=facets.price_range.to%3D{max_price}")

def build_amazon_url_v2(outfit_type: str, style_tag: str, min_price: int, max_price: int) -> str:
    keyword = f"{style_tag.replace('_', ' ')} {outfit_type.replace('_', ' ')} women india"
    # n=1571271031 is Amazon.in's "Women's Clothing" department node — scopes results to women's fashion
    return (f"https://www.amazon.in/s?k={quote(keyword)}"
            f"&n=1571271031"
            f"&rh=p_36%3A{min_price*100}-{max_price*100}")

# ════════════════════════════════════════════════════════════
# Constants — same vocabulary as the original sidebar
# ════════════════════════════════════════════════════════════
BODY_TYPES = ['hourglass', 'pear', 'apple', 'rectangle']
SKIN_TONES = ['warm', 'cool', 'neutral']
OCCASIONS = ['party', 'casual', 'formal']
SEASONS = ['all', 'summer', 'winter', 'spring', 'autumn']
OUTFIT_TYPE_DISPLAY = {
    'kurta': 'Kurta', 'saree': 'Saree', 'lehenga': 'Lehenga',
    'dress': 'Western Dress', 'formal_set': 'Formal Set / Blazer',
    'top_jeans': 'Top + Jeans', 'co_ord_set': 'Co-ord Set',
    'jumpsuit': 'Jumpsuit', 'anarkali': 'Anarkali Suit',
    'palazzo_set': 'Kurta Palazzo Set', 'shirt_trouser': 'Shirt + Trousers',
    'maxi_dress': 'Maxi Dress',
}
VIBE_PHRASES = {
    'festive': 'born to celebrate', 'glam': 'made for the spotlight',
    'chic': 'effortlessly chic', 'boho': 'beautifully untamed',
    'minimal': 'elegantly restrained', 'corporate': 'powerfully refined',
    'ethnic_casual': 'gracefully everyday', 'ethnic_formal': 'timelessly elegant',
    'western_casual': 'casually confident', 'traditional': 'rooted in beauty',
}

# ════════════════════════════════════════════════════════════
# CACHED TRAINING — UNCHANGED ML pipeline
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_and_train():
    setup_database()
    df = load_data()
    X_train, X_test, y_train, y_test, encoders, target_le = encode_and_split(df)
    class_names = list(target_le.classes_)
    best_model, best_name, results = train_and_evaluate(
        X_train, X_test, y_train, y_test, class_names
    )
    get_feature_importance(best_model, FEATURE_COLS)
    save_model(best_model, encoders, target_le)
    initialise_online_model(X_train, y_train, list(range(len(class_names))), target_le, encoders)
    return df, best_model, encoders, target_le

# ════════════════════════════════════════════════════════════
# Session state
# ════════════════════════════════════════════════════════════
defaults = {
    "page": 1, "name": "",
    "body_type": BODY_TYPES[0], "skin_tone": SKIN_TONES[0],
    "occasion": OCCASIONS[0], "outfit_type": list(OUTFIT_TYPE_DISPLAY.keys())[0],
    "season": SEASONS[0], "min_price": 1000, "max_price": 5000,
    "batch_style": None, "top_confidence": 0.0,
    "outfit_cards": [], "budget_cards": [], "budget_in_range": True,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go(page):
    st.session_state.page = page
    st.rerun()

def restart():
    for k, v in defaults.items():
        st.session_state[k] = v
    st.rerun()

def render_progress():
    dots = ""
    for i in range(1, 7):
        cls = "active" if i == st.session_state.page else ("done" if i < st.session_state.page else "")
        dots += f'<div class="dot {cls}"></div>'
    st.markdown(f'<div class="progress-row">{dots}</div>', unsafe_allow_html=True)

def nav_buttons(back_label, back_target, continue_label="Continue →", on_continue=None):
    """Renders Back / Continue side by side. on_continue(): return True to advance."""
    c1, c2 = st.columns([1, 1])
    with c1:
        with st.container(key=f"ghost_{back_target}"):
            if st.button(back_label):
                go(back_target)
    with c2:
        if st.button(continue_label):
            if on_continue is None or on_continue():
                pass  # on_continue itself calls go(...)

# ════════════════════════════════════════════════════════════
# PAGE 1 — Name
# ════════════════════════════════════════════════════════════
def page_name():
    with st.container(key="card_1"):
        st.markdown("""
        <div class="eyebrow">Welcome to STYLAI</div>
        <div class="headline">Let's find<br><em>your</em> vibe ✨</div>
        <div class="subline">A few soft questions, then your outfits await.</div>
        """, unsafe_allow_html=True)
        name = st.text_input("", placeholder="What's your name?", value=st.session_state.name, label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Continue →"):
            if name.strip():
                st.session_state.name = name.strip()
                go(2)
            else:
                st.warning("Tell me your name first 💌")

# ════════════════════════════════════════════════════════════
# PAGE 2 — Body type + Skin tone
# ════════════════════════════════════════════════════════════
def page_body_skin():
    with st.container(key="card_2"):
        st.markdown(f"""
        <div class="eyebrow">Step 2 of 5</div>
        <div class="headline">Hi {st.session_state.name} 👋<br>tell me about <em>you</em></div>
        <div class="subline">This helps pick what flatters you most.</div>
        <div class="field-label">Body Type</div>
        """, unsafe_allow_html=True)
        body_type = st.radio("", BODY_TYPES, index=BODY_TYPES.index(st.session_state.body_type),
                              horizontal=True, label_visibility="collapsed", key="r_body")
        st.markdown('<div class="field-label">Skin Tone</div>', unsafe_allow_html=True)
        skin_tone = st.radio("", SKIN_TONES, index=SKIN_TONES.index(st.session_state.skin_tone),
                              horizontal=True, label_visibility="collapsed", key="r_skin")
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            with st.container(key="ghost_2"):
                if st.button("← Back"):
                    go(1)
        with c2:
            if st.button("Continue →"):
                st.session_state.body_type = body_type
                st.session_state.skin_tone = skin_tone
                go(3)

# ════════════════════════════════════════════════════════════
# PAGE 3 — Occasion + Outfit type
# ════════════════════════════════════════════════════════════
def page_occasion_outfit():
    with st.container(key="card_3"):
        st.markdown("""
        <div class="eyebrow">Step 3 of 5</div>
        <div class="headline">What's the <em>occasion</em>?</div>
        <div class="subline">And what kind of outfit are you picturing?</div>
        <div class="field-label">Occasion</div>
        """, unsafe_allow_html=True)
        occasion = st.radio("", OCCASIONS, index=OCCASIONS.index(st.session_state.occasion),
                             horizontal=True, label_visibility="collapsed", key="r_occ")
        st.markdown('<div class="field-label">Outfit Type</div>', unsafe_allow_html=True)
        outfit_keys = list(OUTFIT_TYPE_DISPLAY.keys())
        outfit_labels = list(OUTFIT_TYPE_DISPLAY.values())
        current_idx = outfit_keys.index(st.session_state.outfit_type)
        outfit_label = st.radio("", outfit_labels, index=current_idx,
                                 horizontal=True, label_visibility="collapsed", key="r_outfit")
        outfit_type = outfit_keys[outfit_labels.index(outfit_label)]
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            with st.container(key="ghost_3"):
                if st.button("← Back"):
                    go(2)
        with c2:
            if st.button("Continue →"):
                st.session_state.occasion = occasion
                st.session_state.outfit_type = outfit_type
                go(4)

# ════════════════════════════════════════════════════════════
# PAGE 4 — Season + Budget
# ════════════════════════════════════════════════════════════
def page_season_budget():
    with st.container(key="card_4"):
        st.markdown("""
        <div class="eyebrow">Step 4 of 5</div>
        <div class="headline">One last detail ✨</div>
        <div class="subline">Season, and your budget in INR.</div>
        <div class="field-label">Season</div>
        """, unsafe_allow_html=True)
        season = st.radio("", SEASONS, index=SEASONS.index(st.session_state.season),
                           horizontal=True, label_visibility="collapsed", key="r_season")
        st.markdown('<div class="field-label">Budget (₹)</div>', unsafe_allow_html=True)
        price_range = st.slider("", min_value=400, max_value=15000,
                                 value=(st.session_state.min_price, st.session_state.max_price),
                                 step=100, format="₹%d", label_visibility="collapsed")
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            with st.container(key="ghost_4"):
                if st.button("← Back"):
                    go(3)
        with c2:
            if st.button("Continue →"):
                st.session_state.season = season
                st.session_state.min_price, st.session_state.max_price = price_range
                go(5)

# ════════════════════════════════════════════════════════════
# PAGE 5 — Generate (runs the ML model + explainer behind the scenes)
# ════════════════════════════════════════════════════════════
def run_recommendation_engine():
    df, best_model, encoders, target_le = load_and_train()
    s = st.session_state

    user = User(
        name=s.name, body_type=s.body_type, skin_tone=s.skin_tone,
        occasion=s.occasion, outfit_type=s.outfit_type, season=s.season,
    )
    X_user = encode_user_input(user.as_feature_dict(), encoders)
    batch_style, batch_probs = predict_style(best_model, X_user, target_le)
    top_confidence = batch_probs.get(batch_style, 0.0)

    # ── Top picks ──
    matches = df[(df['style_tag'] == batch_style) & (df['occasion'] == user.occasion)] \
        .sort_values('rating', ascending=False)
    if matches.empty:
        matches = df[df['style_tag'] == batch_style].sort_values('rating', ascending=False)

    cards = []
    for _, row in matches.head(6).iterrows():
        reasons = get_recommendation_reasons(
            user_body_type=user.body_type, user_skin_tone=user.skin_tone,
            user_occasion=user.occasion, user_outfit_type=user.outfit_type,
            predicted_style=batch_style, outfit_color=str(row['color']),
            outfit_price=int(row['price_inr']), min_price=s.min_price, max_price=s.max_price,
        )
        cards.append({
            "outfit_type": row['outfit_type'], "color": str(row['color']),
            "title": row['product_name'], "price": int(row['price_inr']),
            "rating": float(row['rating']), "reviews": int(row['num_reviews']),
            "brand": str(row['brand']),
            "reason": reasons[0] if reasons else "",
        })

    # ── Budget-filtered picks ──
    price_matches = df[(df['style_tag'] == batch_style) &
                        (df['price_inr'] >= s.min_price) & (df['price_inr'] <= s.max_price)] \
        .sort_values('rating', ascending=False)
    budget_in_range = not price_matches.empty
    if not budget_in_range:
        style_df = df[df['style_tag'] == batch_style].copy()
        mid = (s.min_price + s.max_price) / 2
        style_df['_dist'] = (style_df['price_inr'] - mid).abs()
        price_matches = style_df.nsmallest(5, '_dist')

    budget_cards = []
    for _, row in price_matches.head(5).iterrows():
        budget_cards.append({
            "outfit_type": row['outfit_type'], "color": str(row['color']),
            "title": row['product_name'], "price": int(row['price_inr']),
            "rating": float(row['rating']), "brand": str(row['brand']),
        })

    s.batch_style = batch_style
    s.top_confidence = top_confidence
    s.outfit_cards = cards
    s.budget_cards = budget_cards
    s.budget_in_range = budget_in_range

def page_generate():
    s = st.session_state
    with st.container(key="card_5"):
        st.markdown(f"""
        <div class="eyebrow">Step 5 of 5</div>
        <div class="headline">Ready, {s.name}? ✨</div>
        <div class="subline">
            {s.body_type.title()} · {s.skin_tone.title()} skin · {s.occasion.title()} ·
            {OUTFIT_TYPE_DISPLAY[s.outfit_type]} · ₹{s.min_price:,}–₹{s.max_price:,}
        </div>
        """, unsafe_allow_html=True)
        c1, c2 = st.columns([1, 1])
        with c1:
            with st.container(key="ghost_4b"):
                if st.button("← Back"):
                    go(4)
        with c2:
            if st.button("✨ Reveal My Outfits"):
                with st.spinner("Curating your looks..."):
                    run_recommendation_engine()
                go(6)

# ════════════════════════════════════════════════════════════
# PAGE 6 — Results: vibe + confidence + Pinterest cards + shop links
# ════════════════════════════════════════════════════════════
def page_results():
    s = st.session_state
    style = s.batch_style or ""
    vibe = VIBE_PHRASES.get(style, "uniquely yourself")
    conf_label, conf_explanation, _conf_color = get_confidence_explanation(s.top_confidence)
    fallback_msg = get_fallback_message(s.top_confidence)

    with st.container(key="card_6a"):
        col_text, col_conf = st.columns([2, 1])
        with col_text:
            st.markdown(f"""
            <div class="eyebrow">{s.name}, you are</div>
            <div class="headline">{vibe} ✨</div>
            <div style="text-align:center;"><span class="vibe-pill">{style.replace('_',' ').upper()}</span></div>
            """, unsafe_allow_html=True)
        with col_conf:
            st.markdown(f"""
            <div class="conf-card">
                <div class="conf-score">{s.top_confidence*100:.0f}%</div>
                <div class="conf-tag">{conf_label}</div>
            </div>
            """, unsafe_allow_html=True)
        if fallback_msg:
            st.markdown(f'<div class="fallback-box">⚠ {fallback_msg}</div>', unsafe_allow_html=True)
        st.markdown(f'<p class="subline" style="text-align:left;">{conf_explanation}</p>', unsafe_allow_html=True)

    # ── Top picks grid ──
    st.markdown('<div class="field-label">Top Picks For You</div>', unsafe_allow_html=True)
    if not s.outfit_cards:
        st.info("No outfits found for this combination — try adjusting your answers.")
    else:
        cards_html = '<div class="pin-grid">'
        for c in s.outfit_cards:
            cards_html += f"""
            <div class="pin-card">
                {render_swatch_html(c['outfit_type'], c['color'])}
                <div class="pin-body">
                    <div class="pin-title">{c['title']}</div>
                    <div class="pin-reason">{c['reason']}</div>
                    <div class="pin-tags">
                        <span class="pin-tag">₹{c['price']:,}</span>
                        <span class="pin-tag">★ {c['rating']}</span>
                        <span class="pin-tag">{c['brand']}</span>
                        <span class="pin-tag">{c['color']}</span>
                    </div>
                </div>
            </div>"""
        cards_html += "</div>"
        st.markdown(cards_html, unsafe_allow_html=True)

    # ── Budget-filtered picks ──
    st.markdown('<div class="field-label">Within Your Budget</div>', unsafe_allow_html=True)
    if s.budget_in_range:
        st.success(f"✦ {len(s.budget_cards)} outfits found within ₹{s.min_price:,}–₹{s.max_price:,}")
    else:
        st.markdown(f'<div class="fallback-box">⚠ No exact match in ₹{s.min_price:,}–₹{s.max_price:,}. Showing the closest options instead.</div>', unsafe_allow_html=True)
    if s.budget_cards:
        bcards_html = '<div class="pin-grid">'
        for c in s.budget_cards:
            bcards_html += f"""
            <div class="pin-card">
                {render_swatch_html(c['outfit_type'], c['color'])}
                <div class="pin-body">
                    <div class="pin-title">{c['title']}</div>
                    <div class="pin-tags">
                        <span class="pin-tag">₹{c['price']:,}</span>
                        <span class="pin-tag">★ {c['rating']}</span>
                        <span class="pin-tag">{c['brand']}</span>
                    </div>
                </div>
            </div>"""
        bcards_html += "</div>"
        st.markdown(bcards_html, unsafe_allow_html=True)

    # ── Shop now links ──
    st.markdown('<div class="field-label">Shop Now</div>', unsafe_allow_html=True)
    myntra_url = build_myntra_url_v2(s.outfit_type, style)
    ajio_url = build_ajio_url_v2(s.outfit_type)
    flipkart_url = build_flipkart_url(s.outfit_type, style, s.min_price, s.max_price)
    amazon_url = build_amazon_url_v2(s.outfit_type, style, s.min_price, s.max_price)
    with st.container(key="card_6b"):
        st.markdown(f"""
        <div class="shop-row">
            <a href="{myntra_url}" target="_blank" class="shop-btn shop-primary">Myntra (Women)</a>
            <a href="{ajio_url}" target="_blank" class="shop-btn shop-primary">AJIO (Women)</a>
            <a href="{flipkart_url}" target="_blank" class="shop-btn shop-outline">Flipkart + Budget filter</a>
            <a href="{amazon_url}" target="_blank" class="shop-btn shop-outline">Amazon India + Budget filter</a>
        </div>
        <div class="budget-note">Budget ₹{s.min_price:,}–₹{s.max_price:,} applied · results scoped to women's fashion where the platform supports it</div>
        """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        with st.container(key="ghost_6"):
            if st.button("← Adjust answers"):
                go(2)
    with c2:
        if st.button("🔁 Style Me Again"):
            restart()

# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    render_progress()
    page = st.session_state.page
    if page == 1:
        page_name()
    elif page == 2:
        page_body_skin()
    elif page == 3:
        page_occasion_outfit()
    elif page == 4:
        page_season_budget()
    elif page == 5:
        page_generate()
    elif page == 6:
        page_results()
    else:
        st.session_state.page = 1
        st.rerun()

if __name__ == '__main__':
    main()
