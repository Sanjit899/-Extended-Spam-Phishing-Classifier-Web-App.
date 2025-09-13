# 📧 Extended Spam & Phishing Classifier Web App

This is a **Python Flask web application** that classifies SMS or email messages as **spam** or **ham** (non-spam). It is enhanced with phishing-aware features and numeric indicators, combining **TF-IDF text features** with **numeric statistics** for improved accuracy.  

This project is ideal for demonstrating full-stack Python skills, including **ML model development, feature engineering, API creation, and Flask web app deployment**.  

---

## 🚀 Features

1. **Machine Learning**
   - Logistic Regression classifier with `TF-IDF (1-2 grams)` + numeric features.
   - Numeric/phishing-aware features:
     - `has_link` → Detects presence of URLs.
     - `digit_frac` → Fraction of characters that are digits.
     - `urgent_count` → Counts “urgent” keywords like `verify`, `urgent`, `account`, `free`, `win`.
   - Evaluates model using precision, recall, F1-score.  
   - Stratified train-test split to avoid imbalance issues.

2. **Web Application (Flask)**
   - User-friendly home page for **text input** and classification.
   - Admin page for **dataset upload & retraining**.
   - Displays **probability scores** and numeric feature values.

3. **API Endpoints**
   - `/predict` → Returns JSON prediction. Requires `X-API-KEY` header.
   - `/top-features` → Returns top spam/ham features used by the model.  
   - Easy integration into other apps or scripts.

4. **CLI Interface**
   - `python app.py train` → Train and save model.
   - `python app.py test` → Run sample predictions.
   - `python app.py run` → Start the Flask server.

5. **Robust Dataset Handling**
   - Auto-detects CSV separators and encodings.
   - Supports UCI SMS SpamCollection download if no dataset exists.
   - Drops invalid rows and normalizes labels (`ham`/`spam` or `0`/`1`).

---

## 📦 Installation

1. **Clone the repo**
```bash
git clone https://github.com/yourusername/spam-classifier.git
cd spam-classifier

Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

Install dependencies
pip install flask scikit-learn pandas joblib requests scipy

Set API key (for API access)
# Windows PowerShell
$env:SPAM_API_KEY = "your-secret-key"

# macOS/Linux
export SPAM_API_KEY="your-secret-key"

🏃 Running the Application

Train the model
python app.py train

Run test predictions
python app.py test

Access the web interface
Open your browser at http://127.0.0.1:5000

🖥️ Web Pages

Home: Enter text and classify it.

Admin: Upload new CSV dataset to retrain the model.

Top Features API: View top tokens and numeric features contributing to spam/ham.



📄 CSV Dataset Format

The CSV must have at least two columns:

label	text
ham	"Hey, are you coming tonight?"
spam	"Congratulations! You won a $1000 gift card. Click http://bit.ly/win
"

label → ham/spam or 0/1

text → Message content

Admin page allows upload and automatic retraining.


🔑 API Usage
1. Predict

Endpoint: /predict
Method: POST
Headers: X-API-KEY: your-secret-key
Body (JSON):

{
  "text": "Congratulations! You've won a prize."
}

Response:

{
  "label": "spam",
  "prob_spam": 0.994,
  "features": {
    "has_link": 1,
    "digit_frac": 0.034,
    "urgent_count": 3
  }
}

2. Top Features

Endpoint: /top-features
Method: GET
Headers: X-API-KEY: your-secret-key
Response: JSON list of top spam and ham features.


🔧 CLI Commands
Command	Description
python app.py train	Train the model with dataset.
python app.py test	Run sample predictions.
python app.py run	Start the Flask server locally.
📈 Example Predictions

Input: Urgent: Your account has been suspended. Verify now: http://fakebank.example.com
Output:Result: SPAM
Spam Probability: 100%
Features: has_link=1, digit_frac=0.034, urgent_count=5

For the  fruther  iprovements you can consider
⚡ Future Improvements

Expand dataset with modern spam/phishing messages.

Add HTML email/attachment parsing.

Deploy with HTTPS and production-ready server (Gunicorn + NGINX).

Track misclassifications for retraining.

Add more advanced NLP or deep learning features for better accuracy.

📝 License

MIT License – free to use, modify, and distribute
