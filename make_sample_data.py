"""Creates data/urls.csv (url,label) so the project runs out of the box.
THIS IS SYNTHETIC. For your report, replace it with a real dataset (see README)."""
import csv, random
random.seed(7)
good_hosts = ["wikipedia.org","github.com","stackoverflow.com","bbc.co.uk","nytimes.com","amazon.in",
  "google.com","youtube.com","linkedin.com","microsoft.com","apple.com","paypal.com","hdfcbank.com",
  "sbi.co.in","netflix.com","medium.com","coursera.org","flipkart.com","irctc.co.in","python.org"]
good_paths = ["","/","/about","/wiki/Machine_learning","/products/item-482","/blog/2025/how-to-learn","/search?q=laptops","/login","/account/settings","/help/contact"]
words = ["login","verify","secure","account","update","signin","confirm","bank","wallet","password"]
brands = ["paypal","google","amazon","microsoft","apple","netflix","sbi","hdfc","paytm","facebook"]
tlds = ["tk","xyz","top","ml","ga","click","com","net","info","online"]
def rnd_word(): return random.choice(words)
def good():
    h = random.choice(good_hosts); sub = random.choice(["","www.","www.","accounts.","mail.","en."])
    return f"{random.choice(['https','https','https','http'])}://{sub}{h}{random.choice(good_paths)}"
def bad():
    k = random.random(); b = random.choice(brands); t = random.choice(tlds)
    if k < .25: return f"http://{random.randint(11,223)}.{random.randint(1,254)}.{random.randint(1,254)}.{random.randint(1,254)}/{rnd_word()}/{rnd_word()}.php"
    if k < .55: return f"{random.choice(['http','https'])}://{b}-{rnd_word()}-{rnd_word()}.{t}/{rnd_word()}"
    if k < .80: return f"https://{b}.com.{rnd_word()}-{rnd_word()}.{t}/{rnd_word()}?id={random.randint(1000,99999)}"
    if k < .92: return f"http://{rnd_word()}.{b}.{rnd_word()}.{random.choice(['co','in','biz'])}.{t}/user@{rnd_word()}"
    return f"http://xn--{b}-{random.randint(10,99)}k.{t}/{rnd_word()}"
rows = [(good(), "good") for _ in range(1500)] + [(bad(), "bad") for _ in range(1500)]
# label noise: real data is messy, so flip a few
rows = [(u, ("bad" if l=="good" else "good") if random.random()<.03 else l) for u,l in rows]
random.shuffle(rows)
with open("data/urls.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["url","label"]); w.writerows(rows)
print("wrote", len(rows), "rows to data/urls.csv")
