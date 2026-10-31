# Phishing URL Detector

Machine-learning project: a Random Forest looks at the *text* of a URL and says whether it is phishing or safe. A Flask API serves the model and a friendly web page (Sam) demos it.

## Run it (4 commands)
```
pip install -r requirements.txt
python make_sample_data.py     # synthetic starter data (see note)
python train.py                # trains, prints metrics, saves charts in model/
python app.py                  # open http://127.0.0.1:5000
```

## Use a real dataset (do this for your report)
1. Download "Phishing Site URLs" from Kaggle (or any CSV with a URL column and a good/bad label).
2. Save it as `data/urls.csv` with the columns `url,label` (labels: good/bad or 0/1).
3. Run `python train.py` again. Use the new numbers in your report.

NOTE: `make_sample_data.py` creates synthetic URLs, so its accuracy is not a real-world result. Never quote it as your final accuracy.

## How it works
1. `features.py` converts a URL into 17 numbers (length, dots, hyphens, @, IP address, HTTPS, subdomains, bait words, risky ending, brand look-alike, and so on).
2. `train.py` splits the data 80/20, trains a Random Forest (200 trees), and reports accuracy, precision, recall, F1 and a confusion matrix.
3. `app.py` loads the model. `POST /predict {"url": "..."}` returns the phishing probability.
4. `static/index.html` shows the clues one by one and the model's risk score. If the API is offline, it falls back to simple rules.

## Interface tabs
- **Check a link** – clue-by-clue result with a "Why this verdict" breakdown.
- **Spot the phish** – a guessing game for live demos.
- **Batch check** – paste many links, sort by risk, copy as CSV.
- **Message check** – paste an SMS/email; links are found and scored, pressure words (urgent, blocked, claim now...) are underlined.
- **Model dashboard** – metrics, confusion matrix and feature importance (edit the numbers in `static/index.html` after training on your real dataset).

## Limits (say these in your interview)
- Reads only the URL text; never visits the page.
- Attackers can craft URLs that look clean, so this is a first filter, not full protection.
- Results depend on the training data's quality and balance.
- Possible next steps: page-content features, WHOIS domain age, TF-IDF character n-grams, XGBoost, browser extension.
