"""
Assignment 3 - Online bookstore: Apriori algorithm
min_support = 40 %, min_confidence = 60 %
Part A: Apriori written from scratch (shows every step)  Part B: same result with mlxtend
"""

import os
from itertools import combinations
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "out"
os.makedirs(OUT, exist_ok=True)
os.makedirs("data", exist_ok=True)
plt.rcParams.update(
    {"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9}
)
pd.set_option("display.width", 140)

# ---------------------------------------------------------------
# DATA: each line = books bought together in one order
# ---------------------------------------------------------------
orders = {
    "T1": ["Clean Code", "Pragmatic Programmer", "Design Patterns"],
    "T2": ["Clean Code", "Pragmatic Programmer", "Design Patterns"],
    "T3": ["Atomic Habits", "Deep Work", "Ikigai"],
    "T4": ["Clean Code", "Pragmatic Programmer", "Design Patterns", "Deep Work"],
    "T5": ["Atomic Habits", "Deep Work", "Clean Code"],
    "T6": ["Clean Code", "Design Patterns"],
    "T7": ["Atomic Habits", "Deep Work", "Clean Code"],
    "T8": ["Clean Code", "Pragmatic Programmer", "Design Patterns"],
    "T9": ["Atomic Habits", "Ikigai"],
    "T10": ["Atomic Habits", "Deep Work", "Pragmatic Programmer"],
}
pd.DataFrame({"order_id": list(orders), "books": [", ".join(v) for v in orders.values()]}).to_csv(
    "data/bookstore_orders.csv", index=False
)

df = pd.read_csv("data/bookstore_orders.csv")
transactions = [set(b.strip() for b in s.split(",")) for s in df["books"]]
N = len(transactions)
MIN_SUP, MIN_CONF = 0.40, 0.60
min_count = MIN_SUP * N
print(
    f"Transactions = {N}, min_support = {MIN_SUP:.0%} -> min count = {min_count:.0f}, min_conf = {MIN_CONF:.0%}"
)
print(df.to_string(index=False))


# ---------------------------------------------------------------
# PART A: APRIORI FROM SCRATCH
# ---------------------------------------------------------------
def support_count(itemset):
    return sum(1 for t in transactions if itemset <= t)


def name(s):
    return "{" + ", ".join(sorted(s)) + "}"


# Step 1: C1 and L1
items = sorted(set().union(*transactions))
C = [frozenset([i]) for i in items]
k, frequent = 1, {}
while C:
    counts = {c: support_count(c) for c in C}
    print(f"\nC{k} (candidates with support count):")
    for c in sorted(counts, key=lambda x: sorted(x)):
        print(f"   {name(c):60s} {counts[c]}  {'kept' if counts[c] >= min_count else 'removed'}")
    L = {c: v for c, v in counts.items() if v >= min_count}
    if not L:
        print(f"L{k} is empty -> stop")
        break
    print(f"L{k}: " + ", ".join(f"{name(c)}:{v}" for c, v in L.items()))
    frequent.update(L)
    # Step 2: JOIN Lk with itself to make C(k+1), then PRUNE using the Apriori property
    keys = list(L)
    cand = set()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            u = keys[i] | keys[j]
            if len(u) == k + 1:
                if all(frozenset(s) in L for s in combinations(u, k)):
                    cand.add(u)
                else:
                    bad = [name(frozenset(s)) for s in combinations(u, k) if frozenset(s) not in L]
                    print(f"   prune {name(u)} because subset {bad[0]} is not frequent")
    C = sorted(cand, key=lambda x: sorted(x))
    k += 1

# Step 3: generate rules  X -> Y  with confidence = sup(X u Y) / sup(X)
rules = []
for itemset, cnt in frequent.items():
    if len(itemset) < 2:
        continue
    for r in range(1, len(itemset)):
        for X in combinations(sorted(itemset), r):
            X = frozenset(X)
            Y = itemset - X
            conf = cnt / frequent[X]
            lift = conf / (frequent[Y] / N)
            rules.append((name(X), name(Y), cnt / N, conf, lift, conf >= MIN_CONF))
R = pd.DataFrame(rules, columns=["antecedent", "consequent", "support", "confidence", "lift", "strong"])
R = R.sort_values(["strong", "confidence", "lift"], ascending=False)
print("\nALL CANDIDATE RULES:")
print(R.round(3).to_string(index=False))
strong = R[R.strong]
print(f"\nSTRONG RULES (confidence >= {MIN_CONF:.0%}): {len(strong)} of {len(R)}")

# ---------------------------------------------------------------
# PART B: SAME THING WITH THE mlxtend LIBRARY
# ---------------------------------------------------------------
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

te = TransactionEncoder()
onehot = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
print("\nOne-hot (basket) matrix:")
print(onehot.astype(int).rename(columns=lambda c: c.split()[0]).to_string())
fi = apriori(onehot, min_support=MIN_SUP, use_colnames=True)
fi["length"] = fi["itemsets"].apply(len)
print("\nFrequent itemsets (mlxtend):")
print(
    fi.assign(itemsets=fi.itemsets.apply(name))
    .sort_values(["length", "support"], ascending=[True, False])
    .to_string(index=False)
)
ar = association_rules(fi, metric="confidence", min_threshold=MIN_CONF)
ar = ar.sort_values(["confidence", "lift"], ascending=False)
print("\nAssociation rules (mlxtend):")
print(
    ar.assign(antecedents=ar.antecedents.apply(name), consequents=ar.consequents.apply(name))[
        ["antecedents", "consequents", "support", "confidence", "lift"]
    ]
    .round(3)
    .to_string(index=False)
)


# ---------------------------------------------------------------
# RECOMMENDATION: books to suggest when a customer adds a book to cart
# ---------------------------------------------------------------
def recommend(book):
    hits = ar[ar.antecedents.apply(lambda a: a == frozenset([book]))]
    return [(name(r.consequents), round(r.confidence, 2)) for r in hits.itertuples()]


print("\nRECOMMENDATIONS")
for b in ["Clean Code", "Atomic Habits", "Design Patterns"]:
    print(f"  Customer buys {b!r:22s} -> recommend {recommend(b)}")

# ---------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------
fs = fi.assign(label=fi.itemsets.apply(lambda s: " + ".join(sorted(x.split()[0] for x in s))))
fs = fs.sort_values("support")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
cols = {1: "#1F3A5F", 2: "#2A9D8F", 3: "#E76F51"}
ax[0].barh(fs.label, fs.support, color=[cols[l] for l in fs.length])
ax[0].axvline(MIN_SUP, color="grey", ls="--", lw=1)
ax[0].text(MIN_SUP, -0.9, " min sup 40%", fontsize=7)
ax[0].set_title("Frequent itemsets (colour = size)")
ax[0].set_xlabel("support")
sc = ax[1].scatter(R.support, R.confidence, c=R.lift, s=70, cmap="viridis", edgecolor="k")
ax[1].axhline(MIN_CONF, color="#E76F51", ls="--", lw=1)
ax[1].text(0.405, MIN_CONF + 0.01, "min confidence 60%", fontsize=7, color="#E76F51")
plt.colorbar(sc, ax=ax[1], label="lift")
ax[1].set_xlabel("support")
ax[1].set_ylabel("confidence")
ax[1].set_title("All rules: strong ones above the line")
plt.tight_layout()
plt.savefig(f"{OUT}/a3_apriori.png")
plt.close()
