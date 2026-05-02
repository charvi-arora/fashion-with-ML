# STYLAI — Deployment Guide
# Deploy on Render (free, stable URL) + Custom Domain

---

## WHAT'S FIXED IN v2

| Bug | Old Behaviour | Fix |
|-----|--------------|-----|
| Same image for all outfits | Every card showed the same style-tag photo | Per-card image based on outfit_type + color |
| Myntra links broken | Deep price-filter URLs were 404ing | Simplified to keyword search (stable) |
| AJIO links broken | `/s/` URL format was wrong | Fixed to `/search/?text=` format |
| Only Amazon opened | Myntra/AJIO blocked | Added Flipkart with working price filters |
| Amazon price wrong | Was multiplying by 100 twice | Fixed paise conversion |
| App asleep on Streamlit | Free tier kills idle apps | Move to Render (always-on) |

---

## STEP 1 — Push Full Project to GitHub

Make sure your GitHub repo has ALL these files:

```
fashion-with-ML/
├── app.py                   ← REPLACE with the new v2 file
├── requirements.txt
├── .streamlit/
│   └── config.toml          ← NEW (dark theme)
├── main.py
├── data/
│   └── outfits.csv
├── models/
│   ├── __init__.py
│   └── outfit_model.py
├── utils/
│   ├── __init__.py
│   ├── data_handler.py
│   ├── db_handler.py
│   ├── explainer.py
│   ├── learning_engine.py
│   ├── ml_engine.py
│   ├── price_linker.py
│   ├── recommender.py
│   └── visualizer.py
└── outputs/                 ← empty folder (gitkeep)
```

Push commands:
```bash
cd your-project-folder
git add .
git commit -m "v2: premium UI, fixed images, fixed shopping links"
git push origin main
```

---

## STEP 2 — Deploy on Render (Free Tier)

Render gives you a **stable, always-on URL** like:
`https://fashion-ai.onrender.com`

### 2a. Create Account
1. Go to https://render.com
2. Sign up with GitHub (click "Connect GitHub")

### 2b. Create a Web Service
1. Click **"New +"** → **"Web Service"**
2. Connect your GitHub repo: `charvi-arora/fashion-with-ML`
3. Fill in these settings:

| Field | Value |
|-------|-------|
| Name | `fashion-ai` (or any name) |
| Region | Singapore (closest to India) |
| Branch | `main` |
| Runtime | Python 3 |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` |
| Instance Type | **Free** |

4. Click **"Create Web Service"**

Render will build and deploy automatically. Takes ~3-5 minutes first time.

Your URL: `https://fashion-ai.onrender.com`

### 2c. Important: Free Tier Note
Render free tier **spins down after 15 min of inactivity** (same as Streamlit).
To keep it always-on (free workaround):
- Use **UptimeRobot** (free) to ping your URL every 5 minutes
- Go to https://uptimerobot.com → New Monitor → HTTP → paste your Render URL → every 5 mins

---

## STEP 3 — Custom Domain (Optional but Recommended)

### 3a. Buy a Domain
Good options for Indian projects:
- **GoDaddy India**: https://in.godaddy.com → search `fashionai.in` or `stylai.in`
- **Namecheap**: https://www.namecheap.com (usually cheaper)
- **BigRock**: https://www.bigrock.in (India-based)

Suggested domains:
- `stylai.in` (~₹700/year)
- `fashion-ai.in` (~₹700/year)
- `charvi-styles.in` (~₹700/year)

### 3b. Connect Domain to Render
1. In Render dashboard → your service → **"Custom Domains"** tab
2. Click **"Add Custom Domain"**
3. Enter your domain e.g. `stylai.in`
4. Render gives you a CNAME value like: `fashion-ai.onrender.com`

### 3c. Update DNS on GoDaddy/Namecheap
In your domain registrar's DNS settings, add:

| Type | Name | Value |
|------|------|-------|
| CNAME | `www` | `fashion-ai.onrender.com` |
| CNAME | `@` | `fashion-ai.onrender.com` |

DNS propagation takes 15 min – 48 hrs.

After that: `https://www.stylai.in` → your app ✓

---

## STEP 4 — Auto-Deploy on Push

Render auto-deploys every time you push to `main`.
No manual steps needed after setup.

Workflow going forward:
```bash
# Make changes to app.py locally
# Test with: streamlit run app.py
git add .
git commit -m "your change description"
git push origin main
# → Render auto-deploys in ~2 mins
```

---

## QUICK REFERENCE

| Platform | URL | Cost | Always-On |
|----------|-----|------|-----------|
| Render (current) | `fashion-ai.onrender.com` | Free | With UptimeRobot |
| Render (paid $7/mo) | same | $7/mo | Yes, native |
| Custom domain | `stylai.in` | ~₹700/year | Depends on above |

---

## TROUBLESHOOTING

**App crashes on startup:**
- Check that ALL files in `utils/` and `models/` are pushed to GitHub
- Render build logs will show exactly which import failed

**Images not loading:**
- Unsplash URLs need internet; Render has internet by default ✓

**Shopping links still not working:**
- Myntra/AJIO may change their URL format. The current fix uses simple search which is most stable.
- If they break again, switch to the Flipkart/Amazon links which are more reliable.

**SQLite DB resets on every deploy:**
- Render free tier has ephemeral disk. For persistent DB, upgrade to Render paid or use a free PostgreSQL on Render.
