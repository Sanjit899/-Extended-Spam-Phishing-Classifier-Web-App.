#!/usr/bin/env python3
"""
Extended Spam Filter Web App (app.py)

Enhancements added in this version:
- Robust CSV loading (auto-detect separators, handle extra columns) to avoid ParserError.
- Phishing-aware features:
    - has_link (binary)
    - digit_fraction (fraction of chars that are digits)
    - urgent_words_count (count of words like 'verify', 'urgent', 'account', etc.)
- Combined features: TF-IDF (1-2 grams) + numeric features fed to a single LogisticRegression model.
- top-features returns token features + numeric feature names.
- Preserves UI, admin upload & retrain, /predict API (with API key), /top-features, CLI.
- Single-file app (still).

Requirements:
pip install flask scikit-learn pandas joblib requests scipy
"""

import os
import sys
import re
import io
import zipfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import requests
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

from scipy.sparse import csr_matrix, hstack

# ------------------------- Config -------------------------
MODEL_PATH = "spam_model.joblib"
DATA_PATH = "spam.csv"
RANDOM_STATE = 42
API_KEY = os.environ.get("SPAM_API_KEY", "secret123")
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Urgent words used to help detect phishing/social-engineering style messages
URGENT_WORDS = [
    "verify", "verification", "urgent", "immediately", "account", "password",
    "reset", "locked", "suspended", "security", "alert", "click", "confirm",
    "update", "signin", "sign-in", "login", "limit",  "action required"
]


# ------------------------- Helpers -------------------------
def clean_text(s: str) -> str:
    """Basic cleaning: lowercase, remove urls/emails, non-alphanumeric -> space."""
    if not isinstance(s, str):
        return ""
    s = s.lower()
    # remove common urls and emails (keep as space so has_link still works on raw text if needed)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = re.sub(r"\S+@\S+", " ", s)
    # keep only alphanumerics (others to space)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def contains_link(text: str) -> int:
    """Return 1 if text contains a URL-looking substring."""
    if not isinstance(text, str):
        return 0
    return int(bool(re.search(r"https?://|www\.|\.com|\.net|\.org|bit\.ly|tinyurl|\.ru|\.click", text, re.I)))


def digit_fraction(text: str) -> float:
    """Fraction of characters that are digits (0..1)."""
    if not isinstance(text, str) or len(text) == 0:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    return digits / max(1, len(text))


def urgent_word_count(text: str) -> int:
    """Count occurrences of urgent words in text (case-insensitive)."""
    if not isinstance(text, str):
        return 0
    lc = text.lower()
    count = 0
    for w in URGENT_WORDS:
        # simple count; treat words boundaries lightly
        count += lc.count(w)
    return count


# ------------------------- Robust CSV loader -------------------------
def download_ucismsspam(dest: str = DATA_PATH) -> bool:
    """Attempt to download UCI SMS SpamCollection and store as CSV."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(r.content))
        if "SMSSpamCollection" in z.namelist():
            raw = z.read("SMSSpamCollection").decode("latin-1")
            rows = [line.split("\t", 1) for line in raw.splitlines() if "\t" in line]
            df = pd.DataFrame(rows, columns=["label", "text"])
            df.to_csv(dest, index=False, encoding="utf-8")
            return True
    except Exception:
        return False
    return False


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Robust CSV loader:
    - Tries to autodetect separator (engine='python', sep=None)
    - Falls back to common encodings
    - If >2 columns, takes first two columns as label/text
    - If only 1 column, tries tab-separated fallback
    - Returns DataFrame with columns ['label','text']
    """
    p = Path(path)
    if p.exists():
        # try a few options
        read_attempts = [
            {"encoding": "utf-8", "sep": None, "engine": "python"},   # autodetect
            {"encoding": "latin-1", "sep": None, "engine": "python"},
            {"encoding": "utf-8", "sep": ",", "engine": "python"},
            {"encoding": "latin-1", "sep": ",", "engine": "python"},
            {"encoding": "latin-1", "sep": "\t", "engine": "python"},
        ]
        last_exc = None
        for opts in read_attempts:
            try:
                df = pd.read_csv(p, **opts)
                # If columns are unnamed like 0,1 or one-column with tab inside, try to handle
                if df.shape[1] == 1:
                    # try splitting by tab
                    sample = df.iloc[:, 0].astype(str).head(10)
                    if sample.str.contains("\t").any():
                        # re-read as tab-separated
                        df = pd.read_csv(p, sep="\t", encoding=opts.get("encoding", "latin-1"), engine="python")
                # If there are more than 2 cols, take first 2
                if df.shape[1] >= 2:
                    df = df.iloc[:, :2]
                    df.columns = ["label", "text"]
                else:
                    # If only 1 col after above, give up this attempt
                    raise ValueError("Dataset has only 1 column")
                # Normalize common column names
                if set(["v1", "v2"]).issubset(df.columns):
                    df = df.rename(columns={"v1": "label", "v2": "text"})
                # Ensure label/text exist
                if "label" in df.columns and "text" in df.columns:
                    # dropna and ensure strings
                    df = df[["label", "text"]].dropna()
                    df["label"] = df["label"].astype(str).str.strip()
                    df["text"] = df["text"].astype(str)
                    return df
            except Exception as e:
                last_exc = e
                continue
        # if we reach here, loading failed
        raise RuntimeError(f"Failed to read dataset '{path}': {last_exc}")
    # if file not found, try to download the UCI dataset
    if download_ucismsspam(path):
        return load_data(path)
    # fallback tiny dataset for quick demo
    sample = [
        ("ham", "Hi, how are you?"),
        ("spam", "Congratulations! You've won a $1000 gift card"),
        ("ham", "I'll be there in 10 minutes"),
        ("spam", "Free entry to win tickets, text WIN to 8008"),
    ]
    return pd.DataFrame(sample, columns=["label", "text"])


# ------------------------- Feature combiner (TF-IDF + numeric) -------------------------
class TextAndStatsTransformer:
    """
    Custom transformer that:
     - fits a TfidfVectorizer on raw text
     - on transform returns a sparse matrix horizontally stacking [tfidf_matrix, numeric_features]
    This is NOT a full sklearn Estimator but behaves similarly for our Pipeline usage.
    The object is serializable with joblib because it stores sklearn vectorizer and nothing else exotic.
    """

    def __init__(self, ngram_range=(1, 2), max_df=0.9, stop_words="english", min_df=1):
        self.vect = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, stop_words=stop_words, min_df=min_df)
        # names of numeric features appended to TF-IDF
        self.stat_names = ["has_link", "digit_frac", "urgent_count"]

    def fit(self, X: List[str], y=None):
        # X expected to be iterable of raw text strings
        cleaned = [str(x) for x in X]
        self.vect.fit(cleaned)
        return self

    def transform(self, X: List[str]):
        cleaned = [str(x) for x in X]
        tf = self.vect.transform(cleaned)  # sparse
        # compute numeric features on original text (not cleaned) to preserve links etc.
        has_link_arr = [contains_link(x) for x in cleaned]
        digit_frac_arr = [digit_fraction(x) for x in cleaned]
        urgent_arr = [urgent_word_count(x) for x in cleaned]
        stats = np.vstack([has_link_arr, digit_frac_arr, urgent_arr]).T.astype(float)
        stats_sparse = csr_matrix(stats)
        # return hstack of tfidf and numeric features
        return hstack([tf, stats_sparse])

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self):
        # combine TF-IDF feature names and stat names
        try:
            tf_names = self.vect.get_feature_names_out()
        except Exception:
            tf_names = self.vect.get_feature_names()
        return list(tf_names) + self.stat_names


# ------------------------- Training & Model -------------------------
def build_pipeline() -> Pipeline:
    clf = LogisticRegression(solver="liblinear", max_iter=1000, class_weight="balanced")
    # Put our custom features transformer as a step
    return Pipeline([("features", TextAndStatsTransformer(ngram_range=(1, 2), max_df=0.9, stop_words="english")), ("clf", clf)])


def train_and_save(dataset: str = DATA_PATH, model_path: str = MODEL_PATH) -> Pipeline:
    df = load_data(dataset)
    # normalize labels to 'ham'/'spam'
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    # some datasets use 'spam'/'ham' and some use '0'/'1' - map common variants
    mapping = {}
    # detect numeric labels
    if set(df["label"].unique()) <= set(["0", "1"]):
        mapping = {"0": 0, "1": 1}
        df["label_num"] = df["label"].map(mapping).astype(int)
    else:
        # map strings to ham/spam numeric
        df["label"] = df["label"].replace({"ham": "ham", "spam": "spam"})
        df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
        # if label_num contains NaN (unknown labels), drop those rows
        df = df[~df["label_num"].isna()]
        df["label_num"] = df["label_num"].astype(int)

    # prepare X, y
    X = df["text"].astype(str).tolist()
    y = df["label_num"].values

    # simple stratified split when possible
    stratify = y if len(set(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=stratify)

    pipe = build_pipeline()
    print(f"Training on {len(X_train)} samples, validating on {len(X_test)} samples...")
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("\n=== Evaluation on test set ===")
    print(classification_report(y_test, y_pred, digits=4))
    # save pipeline
    joblib.dump(pipe, model_path)
    print(f"Saved model to {model_path}")
    return pipe


def load_model(model_path: str = MODEL_PATH) -> Pipeline:
    if Path(model_path).exists():
        pipe = joblib.load(model_path)
        return pipe
    return train_and_save(dataset=DATA_PATH, model_path=model_path)



# ------------------------- Explainability -------------------------
def top_features(pipeline: Pipeline, n: int = 20) -> dict:
    """
    Return top features for spam and ham.
    Note: our pipeline step 'features' exposes vect via .vect and stat names.
    """
    fe = pipeline.named_steps["features"]
    clf = pipeline.named_steps["clf"]
    # TF-IDF feature names + stat names
    try:
        tf_names = fe.vect.get_feature_names_out()
    except Exception:
        tf_names = fe.vect.get_feature_names()
    all_names = list(tf_names) + fe.stat_names
    coefs = clf.coef_[0]
    # ensure length matches
    if len(coefs) != len(all_names):
        # fallback: trim or pad
        L = min(len(coefs), len(all_names))
        coefs = coefs[:L]
        all_names = all_names[:L]
    inds_desc = np.argsort(coefs)[::-1]
    inds_asc = np.argsort(coefs)
    top_spam = [(all_names[i], float(coefs[i])) for i in inds_desc[:n]]
    top_ham = [(all_names[i], float(coefs[i])) for i in inds_asc[:n]]
    return {"spam": top_spam, "ham": top_ham}


# ------------------------- Flask App -------------------------
app = Flask(__name__)
model_pipeline: Pipeline = None

HOME_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Spam Classifier</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body { font-family: Arial, sans-serif; background: #f2f4f7; margin: 0; padding: 0; }
    .container { max-width: 800px; margin: 40px auto; background: #fff; padding: 28px; border-radius: 12px; box-shadow: 0 6px 20px rgba(0,0,0,0.08); }
    h1 { text-align: center; color: #222; margin-bottom: 10px; }
    textarea { width: 100%; padding: 12px; border-radius: 8px; border: 1px solid #d6d9de; font-size: 15px; resize: vertical; min-height:120px; }
    .actions { display:flex; gap:12px; align-items:center; margin-top:12px; }
    button { padding: 10px 16px; font-size: 15px; border: none; border-radius: 8px; background: #4a90e2; color: white; cursor: pointer; }
    button:hover { background: #357ab7; }
    .result { margin-top: 20px; padding: 16px; border-radius: 8px; font-size: 16px; }
    .spam { background: #fff0f0; color: #b71c1c; border:1px solid #f5c6c6; }
    .ham { background: #f0fff2; color: #1b5e20; border:1px solid #c6f5d1; }
    .meta { margin-top:8px; color:#555; font-size:13px; }
    footer { text-align: center; margin-top: 22px; font-size: 13px; color: #666; }
    .small { font-size: 12px; color:#666; }
  </style>
</head>
<body>
  <div class="container">
    <h1>📧 Spam & Phishing Classifier</h1>
    <form method="post" id="classifyForm">
      <textarea name="text" id="textInput" placeholder="Paste email/SMS text here...">{{ request.form.get('text','') }}</textarea>
      <div class="actions">
        <button type="submit">Classify</button>
        <div class="small">Tip: include subject + body for emails</div>
      </div>
    </form>

    {% if result %}
      <div class="result {{ result.label }}">
        <b>Result:</b> {{ result.label|upper }} &nbsp; &nbsp;
        <b>Spam Probability:</b> {{ \"%.2f\"|format(result.prob_spam * 100) }}%
        <div class="meta">Features: has_link={{ result.features.has_link }}, digit_frac={{ \"%.3f\"|format(result.features.digit_frac) }}, urgent_count={{ result.features.urgent_count }}</div>
      </div>
    {% endif %}

    <footer>
      <a href="/admin">⚙️ Admin</a> • <a href="/top-features">Top Features (API)</a>
    </footer>
  </div>
</body>
</html>
"""

ADMIN_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Admin - Spam Classifier</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <style>
    body { font-family: Arial, sans-serif; background: #f9f9f9; }
    .container { max-width: 680px; margin: 40px auto; background: white; padding: 22px; border-radius: 12px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); }
    h1 { text-align: center; margin-bottom: 10px; }
    input[type=file] { display:block; margin:12px 0; }
    button { padding: 10px 14px; background: #4caf50; color: white; border: none; border-radius: 8px; cursor: pointer; }
    .msg { margin-top: 16px; color: #333; }
    a { color: #4a90e2; text-decoration: none; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Admin - Upload New Dataset</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept=".csv" required>
      <button type="submit">Upload & Retrain</button>
    </form>
    <div class="msg">{{ message }}</div>
    <p><a href="/">⬅ Back to Home</a></p>
    <p class="small">CSV format: first two columns should be &quot;label&quot; and &quot;text&quot; (label: spam/ham or 1/0).</p>
  </div>
</body>
</html>
"""

# ------------------------- Routes -------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    global model_pipeline
    if model_pipeline is None:
        model_pipeline = load_model()
    result = None
    if request.method == "POST":
        text = request.form.get("text", "")
        # compute numeric features for display
        features = {"has_link": contains_link(text), "digit_frac": digit_fraction(text), "urgent_count": urgent_word_count(text)}
        clean = clean_text(text)
        # model expects raw text in our transformer, so we pass original text (transformer handles cleaning internally for TFIDF)
        label_num = int(model_pipeline.predict([text])[0])
        label = "spam" if label_num == 1 else "ham"
        prob = float(model_pipeline.predict_proba([text])[0][1])
        result = {"label": label, "prob_spam": prob, "features": features}
    return render_template_string(HOME_HTML, result=result)


@app.route("/predict", methods=["POST"])
def predict_api():
    """
    JSON API:
    POST /predict
    Headers: X-API-KEY: <your-key>
    Body: {"text": "..."} or {"message": "..."}
    """
    global model_pipeline
    # Require API key
    if request.headers.get("X-API-KEY") != API_KEY:
        return jsonify({"error": "Unauthorized"}), 403
    if model_pipeline is None:
        model_pipeline = load_model()
    try:
        data = request.get_json(force=True, silent=True) or {}
    except Exception:
        data = {}
    text = data.get("text") or data.get("message") or request.form.get("text") or ""
    features = {"has_link": contains_link(text), "digit_frac": digit_fraction(text), "urgent_count": urgent_word_count(text)}
    label_num = int(model_pipeline.predict([text])[0])
    label = "spam" if label_num == 1 else "ham"
    prob = float(model_pipeline.predict_proba([text])[0][1])
    return jsonify({"label": label, "prob_spam": prob, "features": features})


@app.route("/top-features")
def features_endpoint():
    # this endpoint is protected with API key too
    if request.headers.get("X-API-KEY") != API_KEY:
        return jsonify({"error": "Unauthorized"}), 403
    global model_pipeline
    if model_pipeline is None:
        model_pipeline = load_model()
    return jsonify(top_features(model_pipeline, n=50))


@app.route("/admin", methods=["GET", "POST"])
def admin():
    global model_pipeline
    message = ""
    if request.method == "POST":
        if "file" not in request.files:
            message = "No file uploaded"
        else:
            file = request.files["file"]
            if file.filename == "":
                message = "No file selected"
            else:
                filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
                file.save(filepath)
                try:
                    model_pipeline = train_and_save(dataset=filepath)
                    message = "✅ Model retrained with uploaded dataset."
                except Exception as e:
                    message = f"⚠️ Retrain failed: {e}"
    return render_template_string(ADMIN_HTML, message=message)


# ------------------------- CLI helpers -------------------------
def run_tests():
    pipe = load_model()
    examples = [
        "Congratulations! You've won a lottery! Click http://bit.ly/claim now",
        "Hey, are we still on for dinner tonight?",
        "Urgent: Your account will be locked unless you verify your login details",
        "We noticed a new sign-in to your Google Account. If this was you, you don't need to do anything.",
    ]
    for ex in examples:
        pred = pipe.predict([ex])[0]
        prob = pipe.predict_proba([ex])[0][1]
        feats = {"has_link": contains_link(ex), "digit_frac": digit_fraction(ex), "urgent_count": urgent_word_count(ex)}
        print(f"{ex[:80]:80} => {('spam' if pred==1 else 'ham'):4} prob={prob:.3f} features={feats}")


# ------------------------- Entrypoint -------------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
        if cmd == "train":
            train_and_save()
        elif cmd == "test":
            run_tests()
        elif cmd == "run":
            model_pipeline = load_model()
            app.run(host="0.0.0.0", port=5000, debug=True)
        else:
            print("Usage: python app.py [train|test|run]")
    else:
        # default: ensure model exists and run
        if not Path(MODEL_PATH).exists():
            train_and_save()
        model_pipeline = load_model()
        app.run(host="0.0.0.0", port=5000, debug=True)
