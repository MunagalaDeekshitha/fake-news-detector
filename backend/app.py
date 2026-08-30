"""
app.py
------
FastAPI server that loads the trained model and serves predictions.
Supports TWO model backends:
  1. TF-IDF + Logistic Regression (fast baseline) - model.joblib / vectorizer.joblib
  2. Fine-tuned DistilBERT (higher quality) - bert_model/ folder

It automatically uses DistilBERT if bert_model/ exists (i.e. you ran
train_bert.py), otherwise it falls back to the TF-IDF baseline.

Also provides a /verify endpoint that combines the model's stylistic
prediction with a LIVE web search for related recent news articles, so
the user can cross-check claims the model has no real way of "knowing"
about (recent events, topics outside its training data, etc). This does
NOT make the tool a truth oracle -- it surfaces sources for the human to
judge, which is the honest way to handle this limitation.

Run:
    uvicorn app:app --reload --port 8000

Endpoints:
    POST /predict        -> body: {"text": "..."}          -> classify raw text only
    POST /predict_url     -> body: {"url": "https://..."}   -> scrape article then classify
    POST /verify           -> body: {"text": "..."}          -> classify + live news search
    GET  /health          -> basic health check (also reports which model is active)
"""

import os

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.joblib")
BERT_DIR = os.path.join(BASE_DIR, "bert_model")

app = FastAPI(title="Fake News Detector API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Populated at startup depending on which backend is available.
backend = None  # "bert" or "tfidf"
tfidf_model = None
tfidf_vectorizer = None
bert_tokenizer = None
bert_model = None
bert_device = "cpu"


@app.on_event("startup")
def load_artifacts():
    global backend, tfidf_model, tfidf_vectorizer, bert_tokenizer, bert_model, bert_device

    if os.path.isdir(BERT_DIR) and os.path.exists(os.path.join(BERT_DIR, "config.json")):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            bert_device = "cuda" if torch.cuda.is_available() else "cpu"
            bert_tokenizer = AutoTokenizer.from_pretrained(BERT_DIR)
            bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_DIR).to(bert_device)
            bert_model.eval()
            backend = "bert"
            print(f"Loaded DistilBERT model from {BERT_DIR} on {bert_device}.")
            return
        except Exception as e:
            print(f"Failed to load DistilBERT model ({e}); falling back to TF-IDF baseline.")

    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        tfidf_model = joblib.load(MODEL_PATH)
        tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
        backend = "tfidf"
        print("Loaded TF-IDF + Logistic Regression baseline.")
        return

    print(
        "WARNING: No trained model found. Run train_model.py (baseline) or "
        "train_bert.py (DistilBERT) first. /predict will fail until then."
    )


class TextRequest(BaseModel):
    text: str


class UrlRequest(BaseModel):
    url: str


class PredictionResponse(BaseModel):
    label: str
    confidence: float
    fake_probability: float
    real_probability: float
    model_used: str


class NewsResult(BaseModel):
    title: str
    url: str
    source: str = ""
    snippet: str = ""


class VerifyResponse(BaseModel):
    prediction: PredictionResponse
    related_articles: list[NewsResult]
    search_query_used: str
    note: str


def _predict_tfidf(text: str) -> PredictionResponse:
    vec = tfidf_vectorizer.transform([text])
    proba = tfidf_model.predict_proba(vec)[0]  # [P(fake), P(real)]
    fake_p, real_p = float(proba[0]), float(proba[1])
    label = "real" if real_p >= fake_p else "fake"
    return PredictionResponse(
        label=label,
        confidence=round(max(fake_p, real_p), 4),
        fake_probability=round(fake_p, 4),
        real_probability=round(real_p, 4),
        model_used="tfidf_logreg",
    )


def _predict_bert(text: str) -> PredictionResponse:
    import torch

    inputs = bert_tokenizer(
        text, truncation=True, padding="max_length", max_length=256, return_tensors="pt"
    ).to(bert_device)

    with torch.no_grad():
        logits = bert_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    fake_p, real_p = float(probs[0]), float(probs[1])
    label = "real" if real_p >= fake_p else "fake"
    return PredictionResponse(
        label=label,
        confidence=round(max(fake_p, real_p), 4),
        fake_probability=round(fake_p, 4),
        real_probability=round(real_p, 4),
        model_used="distilbert",
    )


def _predict_text(text: str) -> PredictionResponse:
    if backend is None:
        raise HTTPException(
            status_code=503,
            detail="No model loaded. Run train_model.py or train_bert.py first.",
        )
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text provided.")

    if backend == "bert":
        return _predict_bert(text)
    return _predict_tfidf(text)


def _search_related_news(text: str, max_results: int = 5) -> list[NewsResult]:
    """Search for recent news related to the claim, so the user can cross-check
    the model's stylistic prediction against real, current sources. This is the
    part that compensates for the model's fixed training-data cutoff."""
    # Use the first ~15 words as the search query - full articles are too long
    # and dilute the search relevance.
    query = " ".join(text.strip().split()[:15])
    if not query:
        return []

    try:
        from ddgs import DDGS

        results = []
        with DDGS() as ddgs:
            for r in ddgs.news(query, max_results=max_results):
                results.append(
                    NewsResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        source=r.get("source", ""),
                        snippet=r.get("body", "")[:200],
                    )
                )
        return results
    except Exception as e:
        print(f"News search failed: {e}")
        return []

@app.get("/")
def root():
    return {
        "message": "Fake News Detector API is running.",
        "github": "https://github.com/MunagalaDeekshitha/fake-news-detector",
        "endpoints": {
            "GET /health": "Check server + model status",
            "POST /predict": "Classify text as real/fake. Body: {\"text\": \"...\"}",
            "POST /predict_url": "Classify an article by URL. Body: {\"url\": \"...\"}",
            "POST /verify": "Classify text AND get live related news articles. Body: {\"text\": \"...\"}",
        },
        "note": "This is an API service, not a website -- use the endpoints above with a tool like curl or Postman, or use the Chrome extension in the GitHub repo.",
    }
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": backend is not None, "backend": backend}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: TextRequest):
    return _predict_text(req.text)


@app.post("/predict_url", response_model=PredictionResponse)
def predict_url(req: UrlRequest):
    try:
        from newspaper import Article
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="newspaper3k not installed. Run: pip install newspaper3k lxml_html_clean",
        )

    try:
        article = Article(req.url)
        article.download()
        article.parse()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch/parse article: {e}")

    full_text = f"{article.title} {article.text}"
    return _predict_text(full_text)


@app.post("/verify", response_model=VerifyResponse)
def verify(req: TextRequest):
    """Combines the model's stylistic prediction with a live news search.
    This is the honest way to handle claims the model has never seen: instead
    of pretending the model 'knows' if something recent/unfamiliar is true,
    we show the user real, current articles about the same topic so they can
    judge for themselves."""
    prediction = _predict_text(req.text)
    query = " ".join(req.text.strip().split()[:15])
    articles = _search_related_news(req.text)

    if articles:
        note = (
            f"Found {len(articles)} related article(s). Compare the claim against "
            "these sources -- the model's prediction alone should not be treated "
            "as a final verdict, especially for recent or unfamiliar topics."
        )
    else:
        note = (
            "No related articles found. This could mean the claim is very recent, "
            "very obscure, or the search failed. Treat the model's prediction with "
            "extra caution in this case."
        )

    return VerifyResponse(
        prediction=prediction,
        related_articles=articles,
        search_query_used=query,
        note=note,
    )
