"""Compares several models on the same data. Use the table in your report (Model Comparison)."""
import time, pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from features import to_vector, FEATURE_NAMES
from data import load_data

df, y = load_data()
X = pd.DataFrame([to_vector(u) for u in df["url"]], columns=FEATURE_NAMES)
idx_tr, idx_te = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)
rows = []

def score(name, pred, secs):
    rows.append([name, accuracy_score(y[idx_te], pred), precision_score(y[idx_te], pred),
                 recall_score(y[idx_te], pred), f1_score(y[idx_te], pred), round(secs)])
    print(f"done: {name}")

t = time.time()
m = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_leaf=2, random_state=42, n_jobs=-1, class_weight="balanced")
m.fit(X.loc[idx_tr, FEATURE_NAMES[:17]], y[idx_tr]); score("Random Forest, 17 basic features", m.predict(X.loc[idx_te, FEATURE_NAMES[:17]]), time.time() - t)

t = time.time(); m.fit(X.loc[idx_tr], y[idx_tr]); score(f"Random Forest, {len(FEATURE_NAMES)} features", m.predict(X.loc[idx_te]), time.time() - t)

t = time.time()
g = HistGradientBoostingClassifier(random_state=42, class_weight="balanced")
g.fit(X.loc[idx_tr], y[idx_tr]); score(f"Gradient Boosting, {len(FEATURE_NAMES)} features", g.predict(X.loc[idx_te]), time.time() - t)

t = time.time()
tf = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=300000, sublinear_tf=True, lowercase=True)
A = tf.fit_transform(df.loc[idx_tr, "url"]); B = tf.transform(df.loc[idx_te, "url"])
lr = LogisticRegression(max_iter=300, class_weight="balanced", solver="liblinear").fit(A, y[idx_tr])
score("TF-IDF character n-grams + Logistic Regression", lr.predict(B), time.time() - t)

out = pd.DataFrame(rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "Seconds"]).round(3)
print("\n" + out.to_string(index=False)); out.to_csv("model/comparison.csv", index=False)
