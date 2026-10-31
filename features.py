"""Turns a URL into numbers a model can learn from. Used by training AND the API."""
import math
import re
from collections import Counter
from urllib.parse import urlparse

WORDS = ["login", "signin", "verify", "secure", "security", "account", "update",
         "bank", "confirm", "password", "wallet", "suspend"]
BRANDS = ["paypal", "google", "amazon", "microsoft", "apple", "netflix",
          "facebook", "instagram", "sbi", "hdfc", "icici", "paytm"]
OFFICIAL = {
    "paypal": ["paypal.com"], "google": ["google.com", "gmail.com", "youtube.com", "google.co.in", "googleapis.com"],
    "amazon": ["amazon.com", "amazon.in", "amazon.co.uk", "amazonaws.com"],
    "microsoft": ["microsoft.com", "microsoftonline.com", "live.com", "office.com", "outlook.com"],
    "apple": ["apple.com", "icloud.com"], "netflix": ["netflix.com"], "facebook": ["facebook.com", "fb.com"],
    "instagram": ["instagram.com"], "sbi": ["sbi.co.in", "onlinesbi.sbi", "onlinesbi.com", "sbi.bank.in"],
    "hdfc": ["hdfcbank.com", "hdfc.com", "hdfclife.com"], "icici": ["icicibank.com", "icicibank.co.in", "icici.com"],
    "paytm": ["paytm.com"],
}
TRUSTED = {d for v in OFFICIAL.values() for d in v} | {
    "github.com", "wikipedia.org", "stackoverflow.com", "linkedin.com", "flipkart.com", "nytimes.com",
    "bbc.co.uk", "bbc.com", "irctc.co.in", "twitter.com", "x.com", "whatsapp.com", "zoom.us", "dropbox.com",
    "reddit.com", "medium.com", "coursera.org", "python.org", "mozilla.org", "adobe.com"}
RISKY_TLDS = {"tk", "xyz", "top", "ml", "ga", "cf", "click", "zip", "gq", "work"}
SHORTENERS = {"bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly"}

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "num_dots", "num_hyphens_host",
    "num_digits", "digit_ratio", "num_slashes", "has_at", "has_ip", "is_https",
    "subdomain_count", "num_suspicious_words", "has_punycode", "is_shortener",
    "risky_tld", "brand_mismatch",
    # extra features (v2)
    "url_entropy", "num_special", "num_query_params", "query_length", "path_depth",
    "longest_token", "tld_length", "has_www", "host_digits", "has_port", "risky_file_ext",
]

def _entropy(s):
    if not s:
        return 0.0
    c, n = Counter(s), len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())

def _parse(url):
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    try:
        p = urlparse(url)
        p.hostname                       # forces validation
        return p
    except ValueError:                   # malformed link, e.g. a stray [ or ]
        try:
            return urlparse(url.replace("[", "").replace("]", ""))
        except ValueError:
            return urlparse("http://invalid.invalid")

def extract_features(url):
    p = _parse(url)
    host = (p.hostname or "").lower()
    labels = host.split(".")
    reg_len = 3 if (len(labels) >= 3 and len(labels[-1]) == 2 and
                    labels[-2] in {"co", "com", "org", "net", "gov", "ac"}) else 2
    reg_label = labels[-reg_len] if len(labels) >= reg_len else host
    reg_domain = ".".join(labels[-reg_len:])
    low = url.lower()
    digits = sum(c.isdigit() for c in url)
    f = {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(p.path),
        "num_dots": host.count("."),
        "num_hyphens_host": host.count("-"),
        "num_digits": digits,
        "digit_ratio": digits / max(len(url), 1),
        "num_slashes": url.count("/"),
        "has_at": int("@" in url),
        "has_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))),
        "is_https": int(p.scheme == "https"),
        "subdomain_count": max(len(labels) - reg_len, 0),
        "num_suspicious_words": sum(w in low for w in WORDS),
        "has_punycode": int("xn--" in host),
        "is_shortener": int(host in SHORTENERS),
        "risky_tld": int(labels[-1] in RISKY_TLDS),
        "brand_mismatch": int(any(b in host and reg_domain not in OFFICIAL.get(b, []) for b in BRANDS)),
        "url_entropy": round(_entropy(low), 4),
        "num_special": sum(url.count(ch) for ch in "?=&%_~#$"),
        "num_query_params": len(p.query.split("&")) if p.query else 0,
        "query_length": len(p.query),
        "path_depth": p.path.count("/"),
        "longest_token": max((len(t) for t in re.split(r"[^a-zA-Z0-9]+", url)), default=0),
        "tld_length": len(labels[-1]),
        "has_www": int(host.startswith("www.")),
        "host_digits": sum(c.isdigit() for c in host),
        "has_port": int(bool(re.search(r":\d+$", p.netloc.split("@")[-1]))),
        "risky_file_ext": int(bool(re.search(r"\.(php|exe|zip|js|apk|scr)$", p.path.lower()))),
    }
    return f

def is_trusted(url):
    """True when the registered domain is a well-known official site (exact match, so look-alikes fail)."""
    p = _parse(url)
    host = (p.hostname or "").lower()
    labels = host.split(".")
    n = 3 if (len(labels) >= 3 and len(labels[-1]) == 2 and labels[-2] in {"co", "com", "org", "net", "gov", "ac"}) else 2
    return ".".join(labels[-n:]) in TRUSTED or bool(re.search(r"\.(gov|edu|mil)$|\.(gov|nic|ac)\.in$", host))

def looks_like_url(url):
    """False for sentences and other text that is not a web address."""
    u = url.strip()
    if not u or re.search(r"\s", u):
        return False
    host = (_parse(u).hostname or "").lower()
    return bool(re.fullmatch(r"(?:[a-z0-9-]{1,63}\.)+(?:[a-z]{2,24}|xn--[a-z0-9-]+)", host)
                or re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))

def normalize(url):
    """Text the final model reads: the URL without its http(s):// start, so it matches the dataset."""
    return re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", url.strip())

def to_vector(url):
    f = extract_features(url)
    return [f[n] for n in FEATURE_NAMES]
