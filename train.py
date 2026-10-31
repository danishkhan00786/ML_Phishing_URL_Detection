"""Trains the FINAL model: TF-IDF character n-grams + Logistic Regression (best in compare.py)."""
import joblib, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from features import normalize
from data import load_data

df, y = load_data()
text = df["url"].map(normalize)
print(f"{len(df)} unique URLs | phishing share: {y.mean():.0%}")

Xtr, Xte, ytr, yte = train_test_split(text, y, test_size=0.2, random_state=42, stratify=y)
model = Pipeline([
    ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=300000, sublinear_tf=True)),
    ("clf", LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear")),
])
model.fit(Xtr, ytr)
proba = model.predict_proba(Xte)[:, 1]
pred = (proba >= 0.5).astype(int)

print(f"Accuracy : {accuracy_score(yte, pred):.3f}")
print(f"Precision: {precision_score(yte, pred):.3f}")
print(f"Recall   : {recall_score(yte, pred):.3f}")
print(f"F1       : {f1_score(yte, pred):.3f}\n")
print(classification_report(yte, pred, target_names=["safe", "phishing"]))
print("Confusion matrix [[TN, FP], [FN, TP]]:", confusion_matrix(yte, pred).tolist())
print("\nDecision threshold effect:")
for t in (0.5, 0.4, 0.3):
    p = (proba >= t).astype(int)
    print(f"  threshold {t}: precision {precision_score(yte, p):.3f}  recall {recall_score(yte, p):.3f}  f1 {f1_score(yte, p):.3f}")

ConfusionMatrixDisplay(confusion_matrix(yte, pred), display_labels=["safe", "phishing"]).plot(cmap="Blues")
plt.title("Confusion matrix (test set)"); plt.tight_layout(); plt.savefig("model/confusion_matrix.png", dpi=150); plt.close()

# Which character patterns push a URL toward phishing / safe
names = model.named_steps["tfidf"].get_feature_names_out()
coef = model.named_steps["clf"].coef_[0]
order = np.argsort(coef)
top_bad, top_good = order[::-1][:12], order[:12]
print("\nPatterns that most suggest PHISHING:", [names[i] for i in top_bad])
print("Patterns that most suggest SAFE    :", [names[i] for i in top_good])
idx = list(top_good[::-1]) + list(top_bad[::-1])
plt.figure(figsize=(8, 7))
plt.barh([repr(names[i]) for i in idx], coef[idx], color=["#3E8E6B"] * 12 + ["#D4553F"] * 12)
plt.title("Character patterns the model relies on (green = safe, red = phishing)")
plt.tight_layout(); plt.savefig("model/top_patterns.png", dpi=150)
joblib.dump(model, "model/model.joblib")
print("Saved model/model.joblib, confusion_matrix.png, top_patterns.png")
