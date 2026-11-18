"""
Assignment 4 - Bank loan approval prediction (Approved = 1, Rejected = 0)
Models: Decision Tree, Naive Bayes, KNN, SVM  -> compare accuracy, precision, recall, F1, ROC-AUC
"""

import os, warnings

warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

OUT = "out"
os.makedirs(OUT, exist_ok=True)
os.makedirs("data", exist_ok=True)
plt.rcParams.update(
    {"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False, "font.size": 9}
)
pd.set_option("display.width", 140)

# ---------------------------------------------------------------
# STEP 0: DATASET (synthetic but realistic; replace with any loan CSV having the same columns)
# ---------------------------------------------------------------
rng = np.random.default_rng(7)
n = 1000
emp = rng.choice(["Salaried", "Self-Employed", "Unemployed"], n, p=[0.6, 0.3, 0.1])
income = np.where(
    emp == "Unemployed",
    rng.normal(1.8e5, 6e4, n),
    np.where(emp == "Salaried", rng.lognormal(13.4, 0.45, n), rng.lognormal(13.5, 0.6, n)),
)
income = np.clip(income, 6e4, 5e6).round(-3)
credit = np.clip(rng.normal(690, 75, n), 300, 900).round()
loan = np.clip(income * rng.uniform(0.5, 4.0, n), 5e4, 1e7).round(-3)
term = rng.choice([12, 24, 36, 60, 120, 240], n, p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.1])
default = rng.choice(["No", "Yes"], n, p=[0.85, 0.15])
score = (
    0.018 * (credit - 680)
    + 1.3 * np.log(income * term / 12 / loan)
    + np.select([emp == "Salaried", emp == "Self-Employed"], [0.6, 0.2], -1.4)
    - 2.2 * (default == "Yes")
    + rng.normal(0, 0.8, n)
)
approved = (score > 0.3).astype(int)
data = pd.DataFrame(
    {
        "applicant_id": [f"A{1000+i}" for i in range(n)],
        "income": income,
        "credit_score": credit,
        "employment_status": emp,
        "loan_amount": loan,
        "loan_term": term,
        "past_default": default,
        "approved": approved,
    }
)
for col, k in [("income", 15), ("credit_score", 20), ("employment_status", 10)]:
    data.loc[rng.choice(n, k, replace=False), col] = np.nan
data.to_csv("data/loan_applications.csv", index=False)

# ---------------------------------------------------------------
# STEP 1: LOAD + EXPLORE
# ---------------------------------------------------------------
df = pd.read_csv("data/loan_applications.csv")
print("Shape:", df.shape)
print(df.head().to_string(index=False))
print("\nMissing values:")
print(df.isnull().sum()[lambda s: s > 0].to_string())
print("\nClass balance (approved):")
print(df["approved"].value_counts().rename({1: "Approved (1)", 0: "Rejected (0)"}).to_string())
print("\nApproval rate by employment:")
print(df.groupby("employment_status")["approved"].mean().round(2).to_string())
print("Approval rate by past default:")
print(df.groupby("past_default")["approved"].mean().round(2).to_string())

# ---------------------------------------------------------------
# STEP 2: PREPROCESS (impute, scale numeric, one-hot categorical) + FEATURE ENGINEERING
# ---------------------------------------------------------------
df["emi_to_income"] = (df["loan_amount"] / df["loan_term"]) / (
    df["income"] / 12
)  # monthly EMI / monthly income
X = df.drop(columns=["applicant_id", "approved"])
y = df["approved"]
num = ["income", "credit_score", "loan_amount", "loan_term", "emi_to_income"]
cat = ["employment_status", "past_default"]
prep = ColumnTransformer(
    [
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        (
            "cat",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            cat,
        ),
    ]
)

# trees and Naive Bayes do not need scaling -> keep real units so the tree rules are readable
prep_raw = ColumnTransformer(
    [
        ("num", SimpleImputer(strategy="median"), num),
        (
            "cat",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            cat,
        ),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
print(f"\nTrain = {len(X_train)} rows, Test = {len(X_test)} rows (75/25 stratified split)")

# ---------------------------------------------------------------
# STEP 3: TRAIN 4 CLASSIFIERS
# ---------------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(
        criterion="entropy", max_depth=4, min_samples_leaf=10, random_state=42
    ),
    "Naive Bayes": GaussianNB(),
    "KNN (k=7)": KNeighborsClassifier(n_neighbors=7),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
}
rows, fitted, cms = [], {}, {}
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for name, m in models.items():
    pre = prep_raw if name in ("Decision Tree", "Naive Bayes") else prep  # scaling only for KNN/SVM
    pipe = Pipeline([("prep", pre), ("model", m)]).fit(X_train, y_train)
    pred = pipe.predict(X_test)
    prob = pipe.predict_proba(X_test)[:, 1]
    cvs = cross_val_score(Pipeline([("prep", pre), ("model", m)]), X, y, cv=cv, scoring="accuracy")
    rows.append(
        [
            name,
            accuracy_score(y_test, pred),
            precision_score(y_test, pred),
            recall_score(y_test, pred),
            f1_score(y_test, pred),
            roc_auc_score(y_test, prob),
            cvs.mean(),
            cvs.std(),
        ]
    )
    fitted[name] = (pipe, prob)
    cms[name] = confusion_matrix(y_test, pred)
res = pd.DataFrame(
    rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "CV-5 mean", "CV-5 std"]
)
res = res.sort_values("F1", ascending=False).round(3)
print("\nMODEL COMPARISON (test set + 5-fold cross-validation):")
print(res.to_string(index=False))
for name, cm in cms.items():
    tn, fp, fn, tp = cm.ravel()
    print(f"  {name:14s} TN={tn:3d} FP={fp:3d} FN={fn:3d} TP={tp:3d}")
best = res.iloc[0]["Model"]
print(f"\nBest model by F1: {best}")
print(classification_report(y_test, fitted[best][0].predict(X_test), target_names=["Rejected", "Approved"]))

# Decision tree rules (readable)
dt = fitted["Decision Tree"][0]
feat = dt.named_steps["prep"].get_feature_names_out()
feat = [f.split("__")[1] for f in feat]
print("Decision tree (first levels):")
print("\n".join(export_text(dt.named_steps["model"], feature_names=feat, max_depth=2).splitlines()[:16]))
imp = pd.Series(dt.named_steps["model"].feature_importances_, index=feat).sort_values(ascending=False)
print("\nFeature importance (Decision Tree):")
print(imp[imp > 0].round(3).to_string())

# ---------------------------------------------------------------
# STEP 4: PREDICT NEW APPLICANTS
# ---------------------------------------------------------------
new = pd.DataFrame(
    {
        "income": [900000, 300000],
        "credit_score": [780, 560],
        "employment_status": ["Salaried", "Self-Employed"],
        "loan_amount": [1500000, 1200000],
        "loan_term": [60, 36],
        "past_default": ["No", "Yes"],
    }
)
new["emi_to_income"] = (new.loan_amount / new.loan_term) / (new.income / 12)
for name, (pipe, _) in fitted.items():
    p = pipe.predict(new)
    pr = pipe.predict_proba(new)[:, 1]
    print(
        f"  {name:14s} -> applicant 1: {'Approved' if p[0] else 'Rejected'} ({pr[0]:.2f}), "
        f"applicant 2: {'Approved' if p[1] else 'Rejected'} ({pr[1]:.2f})"
    )

# ---------------------------------------------------------------
# STEP 5: VISUALISATION
# ---------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(10, 2.6))
for ax, (name, cm) in zip(axs, cms.items()):
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() * 0.5 else "black",
            )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Rej", "App"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Rej", "App"])
    ax.set_title(name, fontsize=8)
    ax.set_xlabel("Predicted")
axs[0].set_ylabel("Actual")
plt.tight_layout()
plt.savefig(f"{OUT}/a4_confusion.png")
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
metr = res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]]
metr.T.plot(kind="bar", ax=ax[0], rot=0, color=["#1F3A5F", "#2A9D8F", "#E9C46A", "#E76F51"], width=0.8)
ax[0].set_ylim(0.5, 1.0)
ax[0].set_title("Model comparison")
ax[0].legend(fontsize=7, loc="lower right")
for (name, (_, prob)), c in zip(fitted.items(), ["#1F3A5F", "#2A9D8F", "#E9C46A", "#E76F51"]):
    fpr, tpr, _ = roc_curve(y_test, prob)
    ax[1].plot(fpr, tpr, color=c, label=f"{name} (AUC={roc_auc_score(y_test, prob):.2f})")
ax[1].plot([0, 1], [0, 1], "k--", lw=0.8)
ax[1].set_xlabel("False positive rate")
ax[1].set_ylabel("True positive rate")
ax[1].set_title("ROC curves")
ax[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUT}/a4_compare.png")
plt.close()

fig, ax = plt.subplots(figsize=(10, 4.2))
plot_tree(
    dt.named_steps["model"],
    feature_names=feat,
    class_names=["Rejected", "Approved"],
    filled=True,
    max_depth=2,
    fontsize=7,
    ax=ax,
    impurity=True,
    proportion=False,
    rounded=True,
)
plt.tight_layout()
plt.savefig(f"{OUT}/a4_tree.png")
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(9, 2.8))
for v, c in [(0, "#E76F51"), (1, "#2A9D8F")]:
    s = df[df.approved == v]
    ax[0].scatter(
        s.credit_score, s.emi_to_income, s=6, alpha=0.6, color=c, label=["Rejected", "Approved"][v]
    )
ax[0].set_ylim(0, 3)
ax[0].set_xlabel("credit score")
ax[0].set_ylabel("EMI / monthly income")
ax[0].legend(fontsize=7)
ax[0].set_title("Applicants: credit score vs EMI burden")
imp[imp > 0].sort_values().plot(kind="barh", ax=ax[1], color="#1F3A5F")
ax[1].set_title("Decision tree feature importance")
plt.tight_layout()
plt.savefig(f"{OUT}/a4_eda.png")
plt.close()
