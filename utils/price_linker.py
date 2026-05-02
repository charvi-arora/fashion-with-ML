# ============================================================
# price_linker.py  —  Price Filter + Myntra Search Links
# ============================================================
# How it works:
#
#   1. User gives a budget (min_price, max_price in INR)
#   2. We filter the dataset for outfits in that range
#   3. If none found → find the NEAREST outfit by price distance
#   4. For every result, build a real Myntra search URL so the
#      user can click and shop directly
#
# Myntra URL format:
#   https://www.myntra.com/{category}?f=Price%3A{min}%20TO%20{max}
#
# Example:
#   https://www.myntra.com/kurta?f=Price%3A500%20TO%201500
# ============================================================

import pandas as pd
import numpy as np
from urllib.parse import quote


# ── Myntra category slugs (maps outfit_type → Myntra URL keyword) ──
MYNTRA_SLUGS = {
    'kurta':          'kurtas',
    'saree':          'sarees',
    'lehenga':        'lehenga-cholis',
    'dress':          'dresses',
    'formal_set':     'blazers-and-suits',
    'top_jeans':      'jeans',
    'co_ord_set':     'co-ords',
    'jumpsuit':       'jumpsuits',
    'anarkali':       'anarkali-suits',
    'palazzo_set':    'palazzo-pants',
    'shirt_trouser':  'trousers',
    'maxi_dress':     'maxi-dresses',
}

# Style tag → Myntra search keyword (used in text search fallback)
STYLE_KEYWORDS = {
    'festive':        'festive',
    'ethnic_casual':  'ethnic+casual',
    'ethnic_formal':  'ethnic+formal',
    'western_casual': 'casual',
    'corporate':      'formal+workwear',
    'chic':           'chic+party',
    'glam':           'glam+party',
    'boho':           'boho',
    'minimal':        'minimalist',
    'traditional':    'traditional',
}


def build_myntra_url(outfit_type: str, style_tag: str,
                     min_price: int, max_price: int) -> str:
    """
    Build a real Myntra search URL filtered by category and price range.

    Example output:
      https://www.myntra.com/kurtas?f=Price%3A500%20TO%201500
    """
    slug    = MYNTRA_SLUGS.get(outfit_type, outfit_type.replace('_', '-'))
    keyword = STYLE_KEYWORDS.get(style_tag, '')

    # Price filter param (Myntra format)
    price_filter = f"Price%3A{min_price}%20TO%20{max_price}"

    if keyword:
        # Search with style keyword + price filter
        base = f"https://www.myntra.com/{slug}/{quote(keyword.replace('+', ' '))}"
        url  = f"https://www.myntra.com/search?q={quote(keyword.replace('+', ' '))}&f={price_filter}"
    else:
        url = f"https://www.myntra.com/{slug}?f={price_filter}"

    return url


def build_ajio_url(outfit_type: str, min_price: int, max_price: int) -> str:
    """Build AJIO fallback URL (good for ethnic wear)."""
    slug = outfit_type.replace('_', '-')
    return f"https://www.ajio.com/s/{slug}?price={min_price}-{max_price}"


def build_amazon_url(outfit_type: str, style_tag: str,
                     min_price: int, max_price: int) -> str:
    """Build Amazon India fallback URL."""
    keyword = f"{style_tag.replace('_',' ')} {outfit_type.replace('_',' ')} women"
    return (f"https://www.amazon.in/s?k={quote(keyword)}"
            f"&rh=p_36%3A{min_price*100}-{max_price*100}")  # Amazon uses paise


# ── MAIN FUNCTION ────────────────────────────────────────────

def get_price_filtered_recommendations(df: pd.DataFrame,
                                        style_tag: str,
                                        outfit_type: str,
                                        occasion: str,
                                        min_price: int,
                                        max_price: int,
                                        top_n: int = 5) -> dict:
    """
    Filter dataset outfits by price range + style/occasion.
    If none found in range → find nearest by price distance.

    Returns a dict with:
      'in_range'     : list of outfits within budget
      'nearest'      : list of nearest outfits if budget empty
      'links'        : Myntra / AJIO / Amazon search links
      'budget_met'   : True if results found in range
      'price_gap'    : how far the nearest item is from budget (if fallback)
    """

    # Step 1: filter by style + occasion
    base = df[
        (df['style_tag'] == style_tag) &
        (df['occasion']  == occasion)
    ].copy()

    if base.empty:
        base = df[df['style_tag'] == style_tag].copy()

    # Step 2: filter by price range
    in_range = base[
        (base['price_inr'] >= min_price) &
        (base['price_inr'] <= max_price)
    ].sort_values('rating', ascending=False)

    budget_met = not in_range.empty

    # Step 3: if empty → find nearest by absolute price distance
    nearest = pd.DataFrame()
    price_gap = 0

    if not budget_met and not base.empty:
        # Find midpoint of user budget, compute distance to each outfit
        mid = (min_price + max_price) / 2
        base['_price_dist'] = (base['price_inr'] - mid).abs()
        nearest   = base.nsmallest(top_n, '_price_dist').sort_values('rating', ascending=False)
        # Price gap = distance from nearest item to budget boundary
        nearest_price = nearest.iloc[0]['price_inr']
        if nearest_price < min_price:
            price_gap = min_price - nearest_price
        else:
            price_gap = nearest_price - max_price
        base.drop(columns=['_price_dist'], inplace=True)

    # Step 4: build shopping links for both range AND a ±20% wider search
    wider_min = int(min_price * 0.8)
    wider_max = int(max_price * 1.2)

    links = {
        'myntra_exact':  build_myntra_url(outfit_type, style_tag, min_price, max_price),
        'myntra_wider':  build_myntra_url(outfit_type, style_tag, wider_min, wider_max),
        'ajio':          build_ajio_url(outfit_type, min_price, max_price),
        'amazon_india':  build_amazon_url(outfit_type, style_tag, min_price, max_price),
    }

    return {
        'in_range':   in_range.head(top_n),
        'nearest':    nearest.head(top_n) if not budget_met else pd.DataFrame(),
        'links':      links,
        'budget_met': budget_met,
        'price_gap':  price_gap,
        'min_price':  min_price,
        'max_price':  max_price,
    }


def display_price_results(result: dict, style_tag: str) -> None:
    """Pretty-print the price-filtered recommendation results."""
    min_p, max_p = result['min_price'], result['max_price']

    print(f"\n{'═'*66}")
    print(f"  PRICE-FILTERED RECOMMENDATIONS")
    print(f"  Budget: ₹{min_p:,} – ₹{max_p:,}  |  Style: {style_tag.upper().replace('_',' ')}")
    print(f"{'═'*66}")

    if result['budget_met']:
        df_show = result['in_range']
        print(f"\n  ✅ {len(df_show)} outfit(s) found within your budget:\n")
        for _, row in df_show.iterrows():
            print(f"  • {row['product_name']:<46} [{row['brand']:<18}]"
                  f"  ₹{int(row['price_inr']):>6,}  ★{row['rating']}")
    else:
        df_show = result['nearest']
        gap     = result['price_gap']
        print(f"\n  ⚠️  No exact match in ₹{min_p:,}–₹{max_p:,}.")
        print(f"  Nearest outfits (₹{int(gap):,} outside your range):\n")
        for _, row in df_show.iterrows():
            diff = int(row['price_inr']) - max_p if int(row['price_inr']) > max_p else min_p - int(row['price_inr'])
            direction = "above" if int(row['price_inr']) > max_p else "below"
            print(f"  • {row['product_name']:<46} [{row['brand']:<18}]"
                  f"  ₹{int(row['price_inr']):>6,}  ★{row['rating']}"
                  f"  (₹{abs(diff):,} {direction} budget)")

    # Shopping links
    links = result['links']
    print(f"\n  {'─'*62}")
    print(f"  🛍️  Shop Now (click to open):\n")
    print(f"  [Myntra — Exact Budget]  {links['myntra_exact']}")
    print(f"  [Myntra — Wider Search]  {links['myntra_wider']}")
    print(f"  [AJIO]                   {links['ajio']}")
    print(f"  [Amazon India]           {links['amazon_india']}")
    print(f"{'═'*66}")


def get_price_input(is_demo: bool = False) -> tuple:
    """
    Ask the user for their price range.
    Returns (min_price, max_price) as integers.
    """
    if is_demo:
        print("\n  [DEMO] Budget: ₹1,000 – ₹5,000")
        return 1000, 5000

    print("\n  --- Budget Filter ---")
    print("  Enter your price range in INR (e.g. 500 to 3000)")
    try:
        min_p = int(input("  Minimum price (₹): ").strip() or "500")
        max_p = int(input("  Maximum price (₹): ").strip() or "3000")
        if min_p > max_p:
            min_p, max_p = max_p, min_p  # swap if reversed
        return min_p, max_p
    except ValueError:
        print("  [WARN] Invalid input. Using default ₹500–₹3,000.")
        return 500, 3000
