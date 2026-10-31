"""Flask API: POST /predict {"url": "..."} -> {"label", "probability"}"""
import joblib
from flask import Flask, jsonify, request, send_from_directory
from features import normalize, extract_features, is_trusted, looks_like_url

app = Flask(__name__, static_folder="static")
model = joblib.load("model/model.joblib")

@app.get("/")
def home():
    return send_from_directory("static", "index.html")

@app.post("/predict")
def predict():
    url = (request.get_json(silent=True) or {}).get("url", "").strip()
    if not url:
        return jsonify(error="Please send a url."), 400
    if not looks_like_url(url):
        return jsonify(error="That does not look like a link."), 400
    p = float(model.predict_proba([normalize(url)])[0][1])   # chance of phishing
    trusted = is_trusted(url) and "@" not in url      # well-known official domain (exact match)
    if trusted:
        p = min(p, 0.05)
    return jsonify(url=url, probability=round(p, 4), trusted_domain=trusted,
                   label="phishing" if p >= 0.5 else "safe",
                   features=extract_features(url))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
