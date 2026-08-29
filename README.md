# 🔎 Fake News Detector — ML Backend + Chrome Extension

An end-to-end fake news / misinformation detector: a fine-tuned DistilBERT model served
via a FastAPI backend, combined with a live news-search cross-check, and a Chrome
extension that lets you check any selected text or article directly in the browser.

## Project Structure

```
fake-news-detector/
├── backend/
│   ├── train_model.py       # baseline: TF-IDF + Logistic Regression
│   ├── train_bert.py         # fine-tunes DistilBERT (main model)
│   ├── app.py                 # FastAPI server: /predict, /predict_url, /verify
│   ├── requirements.txt
│   ├── requirements-bert.txt
│   └── data/                   # Fake.csv / True.csv (not included, see below)
├── extension/
│   ├── manifest.json           # Chrome extension config (Manifest V3)
│   ├── popup.html / popup.js   # extension UI + logic
│   ├── content.js              # runs on web pages (extend as needed)
│   └── background.js           # right-click "Analyze selection" context menu
└── README.md
```

## 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Dataset
Download the **Fake and Real News Dataset** from Kaggle
(clmentbisaillon/fake-and-real-news-dataset) and place `Fake.csv` / `True.csv`
inside `backend/data/`.

### Baseline model (fast, good starting point)
```bash
python train_model.py
```
Achieves ~98% accuracy using TF-IDF + Logistic Regression.

### Upgraded model: fine-tuned DistilBERT
```bash
pip install -r requirements-bert.txt
python train_bert.py
```
This trains a transformer model on a balanced sample of the dataset (adjust
`SAMPLE_SIZE` in `train_bert.py` — larger = better but slower on CPU). Takes
roughly 30–90 minutes on a CPU-only laptop. `app.py` automatically uses this
model instead of the baseline once `bert_model/` exists.

### Run the API
```bash
uvicorn app:app --reload --port 8000
```

Test the basic prediction:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists confirm the earth is flat, NASA admits cover-up."}'
```

Test the verification endpoint (prediction + live news cross-check):
```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"text": "Cricket World Cup 2027 host country announcement"}'
```

## 2. Load the Chrome Extension

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select the `extension/` folder
4. Open any article, select text, click the extension icon →
   **Use selected text on page** → **Analyze & Verify**

The popup shows both the model's prediction *and* live related articles so you
can check the claim against real sources — not just trust the model blindly.

## 3. Model Details

| Stage | Approach | Accuracy (on this dataset) |
|---|---|---|
| Baseline | TF-IDF (1-2 grams) + Logistic Regression | ~98.8% |
| Main model | Fine-tuned DistilBERT | ~99.9% on held-out test set |

The DistilBERT test accuracy is *very* high — likely because this dataset has
strong stylistic tells between its fake/real sources (different outlets,
punctuation conventions, etc.), not because the model has learned to detect
truth in general. See Limitations below.

## 4. Limitations (and how this project addresses them)

**The core problem:** the model was trained on a 2016–2017 US political news
dataset. It has no concept of "current events" — it only recognizes writing
style and topic patterns from that specific dataset. This creates two
concrete failure modes, both reproduced and measured during development:

- **Near-random confidence on out-of-distribution topics.** A genuinely real,
  neutral headline about Indian monetary policy ("The Reserve Bank of India
  kept its key interest rate unchanged...") got 52% confidence — essentially
  a coin flip, because the topic is far outside the training distribution.
- **Confident, wrong answers on recent events.** A real 2026 announcement —
  "Cricket World Cup 2027 host country announcement" — was flagged as 99.4%
  fake, purely because the model had never seen anything about it and
  defaulted to treating "unfamiliar" as "false."

**The fix implemented here:** the `/verify` endpoint does not rely on the
model alone. It combines the model's stylistic prediction with a **live news
search** (via the `ddgs` package) for the same claim, and returns both to the
user. In the Cricket World Cup example above, `/verify` surfaced five real,
current articles confirming the host announcement — directly contradicting
the model's wrong "fake" call, and giving the user what they need to make
their own informed judgment instead of trusting a single, dated model.

This is the same general approach real fact-checking systems use
(retrieval-augmented verification) rather than a text classifier alone. It
does not make the tool a perfect truth detector — no such thing exists — but
it directly compensates for the model's biggest, most measurable weakness.

## 5. Roadmap / Further Improvements

- [ ] Retrain periodically on more recent, more diverse (non-US, non-political) news
- [ ] Add LIME/SHAP explainability to show *why* text was flagged
- [ ] Auto-detect and analyze the main article body on news sites (extend `content.js`)
- [ ] Deploy backend to Render/Railway/Hugging Face Spaces and update `API_BASE_URL`
- [ ] Add a genuine "uncertain" zone (e.g. 40–60% confidence) instead of forcing
      a binary real/fake label, since confidence near 50% is not a meaningful signal

## License
MIT — built as a learning/portfolio project.
