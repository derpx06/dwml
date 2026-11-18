"""
Assignment 1 - Data Preprocessing in Python
Dataset : Titanic passenger data (open dataset, 891 rows)
Source  : https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

OUT = "out"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update(
    {"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9}
)
C1, C2, C3 = "#1F3A5F", "#2A9D8F", "#E76F51"
pd.set_option("display.max_colwidth", 22)  # shorten long names when printing
pd.set_option("display.width", 120)

# ---------------------------------------------------------------
# STEP 1: IMPORT / READ THE DATA
# ---------------------------------------------------------------
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
try:
    df = pd.read_csv("data/titanic.csv")  # local copy
except FileNotFoundError:
    df = pd.read_csv(URL)  # read directly from the internet
print("STEP 1: Shape (rows, columns):", df.shape)
print(df.head().to_string(max_colwidth=22))

# ---------------------------------------------------------------
# STEP 2: UNDERSTAND THE VARIABLES (data types)
# ---------------------------------------------------------------
print("\nSTEP 2: Data types")
print(df.dtypes.to_string())


def kind(s):
    if pd.api.types.is_bool_dtype(s):
        return "logical"
    if isinstance(s.dtype, pd.CategoricalDtype):
        return "factor (category)"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "numeric (float)"
    return "character (object/string)"


summary = pd.DataFrame(
    {
        "dtype": df.dtypes.astype(str),
        "type": [kind(df[c]) for c in df.columns],
        "unique": df.nunique(),
        "missing": df.isnull().sum(),
    }
)
print("\nVariable summary:")
print(summary.to_string())

print("\nStatistical summary (numeric):")
print(df.describe().round(2).to_string())

# ---------------------------------------------------------------
# STEP 3: CHECK MISSING VALUES
# ---------------------------------------------------------------
miss = df.isnull().sum()
miss_pct = (miss / len(df) * 100).round(2)
mtab = pd.DataFrame({"missing": miss, "percent": miss_pct})
mtab = mtab[mtab.missing > 0].sort_values("missing", ascending=False)
print("\nSTEP 3: Missing values")
print(mtab.to_string())

fig, ax = plt.subplots(figsize=(6, 2.6))
ax.barh(mtab.index[::-1], mtab.percent[::-1], color=[C3, C2, C1][::-1])
for i, v in enumerate(mtab.percent[::-1]):
    ax.text(v + 1, i, f"{v}%", va="center")
ax.set_xlabel("% of rows missing")
ax.set_xlim(0, 100)
ax.set_title("Missing values in Titanic dataset")
plt.tight_layout()
plt.savefig(f"{OUT}/a1_missing.png")
plt.close()

# ---------------------------------------------------------------
# STEP 4: HANDLE MISSING VALUES
# ---------------------------------------------------------------
print("\nSTEP 4: Handling missing values")
# Age (19.87%): numeric and skewed -> fill with MEDIAN of passengers with same Pclass & Sex
print("Median age by Pclass & Sex:")
print(df.groupby(["Pclass", "Sex"])["Age"].median().to_string())
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(lambda s: s.fillna(s.median()))
# Embarked (2 rows): categorical -> fill with MODE
mode_emb = df["Embarked"].mode()[0]
df["Embarked"] = df["Embarked"].fillna(mode_emb)
print("Embarked mode used:", mode_emb)
# Cabin (77% missing): too many missing -> keep only a flag HasCabin, drop the column
df["HasCabin"] = df["Cabin"].notnull().astype(int)
df = df.drop(columns=["Cabin"])
print("Missing values after cleaning:", int(df.isnull().sum().sum()))

# ---------------------------------------------------------------
# STEP 5: DUPLICATES
# ---------------------------------------------------------------
print("\nSTEP 5: Duplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------------
# STEP 6: DATA FORMATTING + TYPE CONVERSION
# ---------------------------------------------------------------
print("\nSTEP 6: Type conversion")
df["Name"] = df["Name"].str.strip()
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")[0]  # Mr, Mrs, Miss ...
df["Title"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
rare = df["Title"].value_counts()[lambda s: s < 10].index
df["Title"] = df["Title"].replace(rare, "Rare")

df["Survived"] = df["Survived"].astype(bool)  # logical
df["Pclass"] = pd.Categorical(df["Pclass"], categories=[3, 2, 1], ordered=True)  # ordered factor
for c in ["Sex", "Embarked", "Title"]:
    df[c] = df[c].astype("category")  # factor
df["PassengerId"] = df["PassengerId"].astype(str)  # ID is a label
print(
    pd.DataFrame(
        {"new dtype": df.dtypes.astype(str), "type": [kind(df[c]) for c in df.columns]}
    ).to_string()
)
print("\nTitle counts:")
print(df["Title"].value_counts().to_string())

# ---------------------------------------------------------------
# STEP 7: OUTLIERS (IQR rule on Fare)
# ---------------------------------------------------------------
q1, q3 = df["Fare"].quantile([0.25, 0.75])
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
n_out = ((df["Fare"] < lo) | (df["Fare"] > hi)).sum()
print(
    f"\nSTEP 7: Fare Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} -> bounds [{lo:.2f}, {hi:.2f}], outliers={n_out}"
)
df["Fare_capped"] = df["Fare"].clip(upper=hi)  # capping (winsorising) keeps the rows

fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].boxplot(df["Fare"], patch_artist=True, boxprops=dict(facecolor="#DCEBF7"))
ax[0].set_title("Fare (raw) - many outliers")
ax[0].set_xticks([])
ax[1].boxplot(df["Fare_capped"], patch_artist=True, boxprops=dict(facecolor="#DDF2EE"))
ax[1].set_title(f"Fare capped at {hi:.1f}")
ax[1].set_xticks([])
plt.tight_layout()
plt.savefig(f"{OUT}/a1_outliers.png")
plt.close()

# ---------------------------------------------------------------
# STEP 8: NORMALIZATION
# ---------------------------------------------------------------
df["Age_minmax"] = MinMaxScaler().fit_transform(df[["Age"]]).ravel()
df["Fare_zscore"] = StandardScaler().fit_transform(df[["Fare"]]).ravel()
j = int(np.ceil(np.log10(df["Fare"].abs().max() + 1e-9)))
df["Fare_decimal"] = df["Fare"] / 10**j
df["Fare_log"] = np.log1p(df["Fare"])
print(f"\nSTEP 8: Normalization (decimal scaling uses j={j})")
print(
    df[["Age", "Age_minmax", "Fare", "Fare_zscore", "Fare_decimal", "Fare_log"]]
    .describe()
    .loc[["mean", "std", "min", "max"]]
    .round(3)
    .to_string()
)

fig, ax = plt.subplots(1, 3, figsize=(8, 2.5))
ax[0].hist(df["Fare"], bins=30, color=C1)
ax[0].set_title("Fare (raw)")
ax[1].hist(df["Fare_zscore"], bins=30, color=C2)
ax[1].set_title("Fare z-score")
ax[2].hist(df["Fare_log"], bins=30, color=C3)
ax[2].set_title("log(1+Fare)")
plt.tight_layout()
plt.savefig(f"{OUT}/a1_normalize.png")
plt.close()

# ---------------------------------------------------------------
# STEP 9: CATEGORICAL -> QUANTITATIVE
# ---------------------------------------------------------------
df["Sex_code"] = LabelEncoder().fit_transform(df["Sex"])  # female=0, male=1
df["Pclass_num"] = df["Pclass"].map({1: 1, 2: 2, 3: 3}).astype(int)  # ordinal mapping
df = pd.get_dummies(df, columns=["Embarked", "Title"], prefix=["Emb", "T"], dtype=int)
df["Survived_num"] = df["Survived"].astype(int)
print("\nSTEP 9: Encoded columns")
enc_cols = ["Sex_code", "Pclass_num"] + [c for c in df.columns if c.startswith(("Emb_", "T_"))]
print(df[enc_cols].head().to_string())

# ---------------------------------------------------------------
# STEP 10: VISUAL SUMMARY
# ---------------------------------------------------------------
num = df[["Survived_num", "Pclass_num", "Sex_code", "Age", "SibSp", "Parch", "Fare", "HasCabin"]]
corr = num.corr().round(2)
print("\nSTEP 10: Correlation with Survived:")
print(corr["Survived_num"].sort_values(ascending=False).to_string())
fig, ax = plt.subplots(figsize=(5.2, 4.4))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)))
ax.set_yticklabels(corr.columns)
for i in range(len(corr)):
    for k in range(len(corr)):
        ax.text(
            k,
            i,
            corr.iloc[i, k],
            ha="center",
            va="center",
            fontsize=7,
            color="white" if abs(corr.iloc[i, k]) > 0.5 else "black",
        )
plt.colorbar(im, fraction=0.046)
ax.set_title("Correlation heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/a1_corr.png")
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
s = df.groupby("Sex", observed=True)["Survived_num"].mean()
ax[0].bar(s.index.astype(str), s.values, color=[C3, C1])
ax[0].set_title("Survival rate by Sex")
p = df.groupby("Pclass_num")["Survived_num"].mean()
ax[1].bar(p.index.astype(str), p.values, color=C2)
ax[1].set_title("Survival rate by Pclass")
for a in ax:
    a.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(f"{OUT}/a1_survival.png")
plt.close()

# ---------------------------------------------------------------
# STEP 11: SAVE CLEAN DATA
# ---------------------------------------------------------------
final = df.drop(columns=["Name", "Ticket", "PassengerId", "Sex", "Pclass", "Survived"])
final.to_csv(f"{OUT}/titanic_clean.csv", index=False)
print("\nSTEP 11: Final shape:", final.shape, "| saved out/titanic_clean.csv")
