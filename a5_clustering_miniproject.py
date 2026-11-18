"""
Assignment 5 - Mini project: Customer segmentation for a shopping mall using clustering
Algorithms: K-Means (elbow + silhouette), Agglomerative hierarchical (dendrogram), DBSCAN
Evaluation: silhouette score, Davies-Bouldin index, cluster profiles
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, dendrogram

OUT = "out"
os.makedirs(OUT, exist_ok=True)
os.makedirs("data", exist_ok=True)
plt.rcParams.update(
    {"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9}
)
pd.set_option("display.width", 140)
PAL = ["#1F3A5F", "#2A9D8F", "#E9C46A", "#E76F51", "#8A4FBF", "#6C757D"]

# ---------------------------------------------------------------
# STEP 0: DATA  (200 mall customers; same columns as the popular "Mall_Customers.csv")
# ---------------------------------------------------------------
rng = np.random.default_rng(3)
groups = [  # (count, income mean, income sd, score mean, score sd, age mean)
    (35, 25, 6, 78, 9, 25),
    (35, 26, 6, 20, 9, 45),
    (66, 55, 7, 50, 7, 42),
    (34, 87, 9, 82, 8, 32),
    (30, 88, 10, 17, 8, 41),
]
rows = []
for cnt, im, isd, sm, ssd, am in groups:
    rows.append(
        pd.DataFrame(
            {
                "Annual_Income_k": rng.normal(im, isd, cnt).clip(15, 137).round(),
                "Spending_Score": rng.normal(sm, ssd, cnt).clip(1, 99).round(),
                "Age": rng.normal(am, 9, cnt).clip(18, 70).round(),
            }
        )
    )
df = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=1).reset_index(drop=True)
df.insert(0, "CustomerID", range(1, len(df) + 1))
df.insert(1, "Gender", rng.choice(["Male", "Female"], len(df), p=[0.44, 0.56]))
df.to_csv("data/mall_customers.csv", index=False)

df = pd.read_csv("data/mall_customers.csv")
print("Shape:", df.shape)
print(df.head().to_string(index=False))
print("\nSummary:")
print(df.describe().round(1).loc[["mean", "std", "min", "max"]].to_string())

# ---------------------------------------------------------------
# STEP 1: PREPROCESS - select features + standardise (distances need same scale)
# ---------------------------------------------------------------
features = ["Annual_Income_k", "Spending_Score"]
X = StandardScaler().fit_transform(df[features])

# ---------------------------------------------------------------
# STEP 2: CHOOSE k - elbow (WCSS / inertia) and silhouette
# ---------------------------------------------------------------
ks = range(1, 11)
wcss, sil = [], {}
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    wcss.append(km.inertia_)
    if k >= 2:
        sil[k] = silhouette_score(X, km.labels_)
print("\nk   WCSS     silhouette")
for k in ks:
    print(f"{k:<3d} {wcss[k-1]:8.2f}  {sil.get(k, float('nan')):.3f}")
best_k = max(sil, key=sil.get)
print("Best k by silhouette =", best_k)

# ---------------------------------------------------------------
# STEP 3: K-MEANS with the chosen k
# ---------------------------------------------------------------
km = KMeans(n_clusters=best_k, n_init=10, random_state=42).fit(X)
df["KMeans"] = km.labels_
prof = (
    df.groupby("KMeans")
    .agg(
        customers=("CustomerID", "count"),
        income=("Annual_Income_k", "mean"),
        score=("Spending_Score", "mean"),
        age=("Age", "mean"),
    )
    .round(1)
)


def label(r):
    hi_inc, hi_sc = r.income > 70, r.score > 60
    lo_inc, lo_sc = r.income < 40, r.score < 40
    if hi_inc and hi_sc:
        return "Premium (rich, spends a lot) - target with luxury offers"
    if hi_inc and lo_sc:
        return "Careful rich (earns a lot, spends little) - upsell"
    if lo_inc and hi_sc:
        return "Young spenders (low income, high spend) - discounts/EMI"
    if lo_inc and lo_sc:
        return "Budget (low income, low spend) - value packs"
    return "Average (medium income, medium spend) - loyalty programme"


prof["segment"] = prof.apply(label, axis=1)
print("\nK-MEANS CLUSTER PROFILE:")
print(prof.to_string())
centres = pd.DataFrame(
    StandardScaler().fit(df[features]).inverse_transform(km.cluster_centers_), columns=features
).round(1)
print("\nCentroids (original units):")
print(centres.to_string())

# ---------------------------------------------------------------
# STEP 4: HIERARCHICAL (agglomerative, Ward linkage)
# ---------------------------------------------------------------
agg = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit(X)
df["Hierarchical"] = agg.labels_
Z = linkage(X, method="ward")

# ---------------------------------------------------------------
# STEP 5: DBSCAN (density based - no k needed, finds noise)
# ---------------------------------------------------------------
db = DBSCAN(eps=0.35, min_samples=5).fit(X)
df["DBSCAN"] = db.labels_
n_db = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
print(f"\nDBSCAN: eps=0.35, min_samples=5 -> {n_db} clusters, {np.sum(db.labels_ == -1)} noise points")

# ---------------------------------------------------------------
# STEP 6: EVALUATION / COMPARISON
# ---------------------------------------------------------------
rows = []
for name in ["KMeans", "Hierarchical", "DBSCAN"]:
    lab = df[name].values
    mask = lab != -1
    rows.append(
        [
            name,
            len(set(lab[mask])),
            int((~mask).sum()),
            silhouette_score(X[mask], lab[mask]),
            davies_bouldin_score(X[mask], lab[mask]),
        ]
    )
cmp = pd.DataFrame(
    rows,
    columns=[
        "Algorithm",
        "clusters",
        "noise",
        "Silhouette (higher better)",
        "Davies-Bouldin (lower better)",
    ],
).round(3)
print("\nCOMPARISON:")
print(cmp.to_string(index=False))
print("\nAgreement K-Means vs Hierarchical (crosstab):")
print(pd.crosstab(df.KMeans, df.Hierarchical).to_string())
df.to_csv(f"{OUT}/mall_customers_segmented.csv", index=False)

# ---------------------------------------------------------------
# STEP 7: VISUALISATION
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
ax[0].plot(list(ks), wcss, "o-", color=PAL[0])
ax[0].axvline(best_k, color=PAL[3], ls="--", lw=1)
ax[0].set_title("Elbow method")
ax[0].set_xlabel("k")
ax[0].set_ylabel("WCSS (inertia)")
ax[1].plot(list(sil), list(sil.values()), "o-", color=PAL[1])
ax[1].axvline(best_k, color=PAL[3], ls="--", lw=1)
ax[1].set_title("Silhouette score")
ax[1].set_xlabel("k")
plt.tight_layout()
plt.savefig(f"{OUT}/a5_choose_k.png")
plt.close()

fig, axs = plt.subplots(1, 3, figsize=(10, 3.1), sharey=True)
for ax, name in zip(axs, ["KMeans", "Hierarchical", "DBSCAN"]):
    for c in sorted(df[name].unique()):
        s = df[df[name] == c]
        ax.scatter(
            s.Annual_Income_k,
            s.Spending_Score,
            s=12,
            color="lightgrey" if c == -1 else PAL[c % len(PAL)],
            label="noise" if c == -1 else f"C{c}",
        )
    ax.set_title(name)
    ax.set_xlabel("Annual income (k)")
axs[0].set_ylabel("Spending score")
axs[0].scatter(
    centres.Annual_Income_k, centres.Spending_Score, marker="X", s=120, color="black", label="centroid"
)
axs[0].legend(fontsize=6, loc="upper left", ncol=2)
axs[2].legend(fontsize=6, loc="upper left", ncol=2)
plt.tight_layout()
plt.savefig(f"{OUT}/a5_clusters.png")
plt.close()

fig, ax = plt.subplots(figsize=(9, 3.0))
dendrogram(Z, truncate_mode="lastp", p=30, ax=ax, color_threshold=Z[-best_k + 1, 2], no_labels=True)
ax.axhline(Z[-best_k, 2] + 0.5 * (Z[-best_k + 1, 2] - Z[-best_k, 2]), color="k", ls="--", lw=0.8)
ax.set_title(f"Dendrogram (Ward linkage) - cut gives {best_k} clusters")
ax.set_ylabel("distance")
plt.tight_layout()
plt.savefig(f"{OUT}/a5_dendrogram.png")
plt.close()
