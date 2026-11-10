"""Flask API: POST /predict {"url": "..."} -> {"label", "probability"}"""
import joblib
from flask import Flask, jsonify, request, send_from_directory
from features import normalize, extract_features, is_trusted, looks_like_url

app = Flask(__name__, static_folder="static")
model = joblib.load("model/model.joblib")
_vec = model.named_steps["tfidf"]
_clf = model.named_steps["clf"]
_NAMES = _vec.get_feature_names_out()

def evidence(text, up=6, down=4):
    """Which character pieces pushed the score up (phishing) or down (safe)."""
    row = _vec.transform([text]).tocoo()
    contrib = row.data * _clf.coef_[0][row.col]
    pairs = sorted(zip(_NAMES[row.col], contrib), key=lambda x: -abs(x[1]))
    def pick(positive, n):
        out = []
        for name, c in pairs:
            if (c > 0) == positive and not any(name in o[0] or o[0] in name for o in out):
                out.append([str(name), round(float(c), 3)])
            if len(out) == n:
                break
        return out
    return {"up": pick(True, up), "down": pick(False, down)}

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
    return jsonify(url=url, probability=round(p, 4), trusted_domain=trusted, evidence=evidence(normalize(url)),
                   label="phishing" if p >= 0.5 else "safe",
                   features=extract_features(url))

if __name__ == "__main__":
    app.run(debug=True, port=5000)
