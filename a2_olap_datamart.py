"""
Assignment 2 - Retail industry: data cube, OLAP operations, sales data mart, ETL
Steps: (1) create raw OLTP-style sales data  (2) ETL into a star-schema sales data mart (SQLite)
       (3) build a data cube  (4) OLAP: roll-up, drill-down, slice, dice, pivot  (5) queries + charts
"""

import os, sqlite3
import numpy as np
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
# STEP 0: SOURCE DATA (what the shop's billing system - the OLTP system - produces)
# ---------------------------------------------------------------
rng = np.random.default_rng(42)
stores = [
    ("S1", "Pune Central", "pune", "Maharashtra", "West"),
    ("S2", "Mumbai Andheri", "Mumbai ", "Maharashtra", "West"),
    ("S3", "Bengaluru MG Rd", "Bengaluru", "Karnataka", "South"),
    ("S4", "Chennai T Nagar", "chennai", "Tamil Nadu", "South"),
    ("S5", "Delhi CP", "Delhi", "Delhi", "North"),
]
products = [
    ("P1", "Smartphone", "Electronics", 15000, 0.12),
    ("P2", "Laptop", "Electronics", 55000, 0.10),
    ("P3", "Headphones", "Electronics", 2000, 0.30),
    ("P4", "Jeans", "Clothing", 1800, 0.40),
    ("P5", "T-Shirt", "Clothing", 600, 0.45),
    ("P6", "Rice 5kg", "Grocery", 450, 0.08),
    ("P7", "Cooking Oil 1L", "Grocery", 180, 0.07),
]
segments = ["Regular", "Premium", "New"]
n = 1200
dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 731, n), unit="D")
rows = []
for i in range(n):
    s = stores[rng.integers(0, 5)]
    p = products[rng.choice(7, p=[0.14, 0.06, 0.14, 0.16, 0.2, 0.16, 0.14])]
    qty = int(rng.integers(1, 4 if p[2] == "Electronics" else 8))
    price = round(p[3] * rng.uniform(0.95, 1.05), 2)
    rows.append(
        [
            1000 + i,
            dates[i].strftime("%d-%m-%Y"),
            s[0],
            s[1],
            s[2],
            s[3],
            s[4],
            p[0],
            p[1],
            p[2],
            rng.choice(segments, p=[0.6, 0.25, 0.15]),
            qty,
            price,
            round(p[4], 2),
        ]
    )
raw = pd.DataFrame(
    rows,
    columns=[
        "bill_no",
        "bill_date",
        "store_id",
        "store_name",
        "city",
        "state",
        "region",
        "product_id",
        "product_name",
        "category",
        "customer_segment",
        "qty",
        "unit_price",
        "margin",
    ],
)
raw.loc[rng.choice(n, 8, replace=False), "qty"] = np.nan  # dirty: missing quantities
raw = pd.concat([raw, raw.sample(10, random_state=1)])  # dirty: duplicate bills
raw.to_csv("data/retail_sales_raw.csv", index=False)
print("SOURCE: raw rows =", len(raw))
print(raw.head(3).to_string(index=False))

# ---------------------------------------------------------------
# STEP 1: ETL  -  EXTRACT
# ---------------------------------------------------------------
src = pd.read_csv("data/retail_sales_raw.csv")
log = {"extracted": len(src)}

# STEP 2: ETL  -  TRANSFORM
src = src.drop_duplicates()
log["after removing duplicates"] = len(src)
src = src.dropna(subset=["qty"])
log["after dropping missing qty"] = len(src)
src["qty"] = src["qty"].astype(int)
src["city"] = src["city"].str.strip().str.title()  # 'pune' -> 'Pune'
src["bill_date"] = pd.to_datetime(src["bill_date"], format="%d-%m-%Y")  # text -> date
src["sales_amount"] = (src["qty"] * src["unit_price"]).round(2)  # derived measures
src["profit"] = (src["sales_amount"] * src["margin"]).round(2)
print("\nETL LOG:")
[print(f"  {k:30s}: {v}") for k, v in log.items()]
print("  cities after cleaning          :", sorted(src.city.unique()))

# build dimension tables with surrogate keys
dim_time = pd.DataFrame({"full_date": sorted(src.bill_date.unique())})
dim_time["time_key"] = range(1, len(dim_time) + 1)
dim_time["day"] = dim_time.full_date.dt.day
dim_time["month"] = dim_time.full_date.dt.month
dim_time["month_name"] = dim_time.full_date.dt.strftime("%b")
dim_time["quarter"] = "Q" + dim_time.full_date.dt.quarter.astype(str)
dim_time["year"] = dim_time.full_date.dt.year
dim_time["full_date"] = dim_time.full_date.dt.strftime("%Y-%m-%d")
src["full_date"] = src.bill_date.dt.strftime("%Y-%m-%d")

dim_product = (
    src[["product_id", "product_name", "category"]]
    .drop_duplicates()
    .sort_values("product_id")
    .reset_index(drop=True)
)
dim_product.insert(0, "product_key", range(1, len(dim_product) + 1))
dim_store = (
    src[["store_id", "store_name", "city", "state", "region"]]
    .drop_duplicates()
    .sort_values("store_id")
    .reset_index(drop=True)
)
dim_store.insert(0, "store_key", range(1, len(dim_store) + 1))
dim_customer = pd.DataFrame({"customer_key": [1, 2, 3], "segment": segments})

fact = (
    src.merge(dim_time[["time_key", "full_date"]], on="full_date")
    .merge(dim_product[["product_key", "product_id"]], on="product_id")
    .merge(dim_store[["store_key", "store_id"]], on="store_id")
    .merge(dim_customer, left_on="customer_segment", right_on="segment")
)
fact_sales = fact[
    ["bill_no", "time_key", "product_key", "store_key", "customer_key", "qty", "sales_amount", "profit"]
].sort_values("bill_no")

# STEP 3: ETL  -  LOAD into the SALES DATA MART (dimensions first, then fact)
if os.path.exists("data/sales_mart.db"):
    os.remove("data/sales_mart.db")
con = sqlite3.connect("data/sales_mart.db")
con.executescript("""
CREATE TABLE dim_time(time_key INTEGER PRIMARY KEY, full_date TEXT, day INT, month INT,
                      month_name TEXT, quarter TEXT, year INT);
CREATE TABLE dim_product(product_key INTEGER PRIMARY KEY, product_id TEXT, product_name TEXT, category TEXT);
CREATE TABLE dim_store(store_key INTEGER PRIMARY KEY, store_id TEXT, store_name TEXT, city TEXT,
                       state TEXT, region TEXT);
CREATE TABLE dim_customer(customer_key INTEGER PRIMARY KEY, segment TEXT);
CREATE TABLE fact_sales(bill_no INT, time_key INT REFERENCES dim_time, product_key INT REFERENCES dim_product,
                        store_key INT REFERENCES dim_store, customer_key INT REFERENCES dim_customer,
                        qty INT, sales_amount REAL, profit REAL);
""")
for name, t in [
    ("dim_time", dim_time),
    ("dim_product", dim_product),
    ("dim_store", dim_store),
    ("dim_customer", dim_customer),
    ("fact_sales", fact_sales),
]:
    t.to_sql(name, con, if_exists="append", index=False)
print("\nLOADED SALES DATA MART (star schema):")
for t in ["dim_time", "dim_product", "dim_store", "dim_customer", "fact_sales"]:
    print(f"  {t:13s} rows = {con.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")
print(dim_store.to_string(index=False))

q = lambda sql: pd.read_sql(sql, con)
JOIN = """FROM fact_sales f JOIN dim_time t ON f.time_key=t.time_key
          JOIN dim_product p ON f.product_key=p.product_key
          JOIN dim_store s ON f.store_key=s.store_key
          JOIN dim_customer c ON f.customer_key=c.customer_key"""

# ---------------------------------------------------------------
# STEP 4: DATA CUBE (Time x Product x Location) -> base view
# ---------------------------------------------------------------
cube_df = q(
    f"SELECT t.year, t.quarter, t.month, p.category, p.product_name, s.region, s.city, c.segment, "
    f"f.qty, f.sales_amount, f.profit {JOIN}"
)
cube = (
    cube_df.pivot_table(
        index=["region", "city"], columns=["year", "quarter"], values="sales_amount", aggfunc="sum"
    )
    .div(1e5)
    .round(2)
)
print("\nDATA CUBE view: sales (Rs lakh) by Location x Time")
print(cube.to_string())

# ---------------------------------------------------------------
# STEP 5: OLAP OPERATIONS
# ---------------------------------------------------------------
print("\n(1) ROLL-UP  city -> region, quarter -> year  (SQL)")
r = q(
    f"SELECT s.region, t.year, ROUND(SUM(f.sales_amount)/1e5,2) AS sales_lakh {JOIN} "
    f"GROUP BY s.region, t.year ORDER BY s.region, t.year"
)
print(r.to_string(index=False))

print("\n    ROLL-UP with sub-totals (ROLLUP emulated with UNION ALL - SQLite has no ROLLUP)")
r2 = q(f"""SELECT region, city, ROUND(SUM(sales_amount)/1e5,2) AS sales_lakh FROM
    (SELECT s.region, s.city, f.sales_amount {JOIN}) GROUP BY region, city
  UNION ALL SELECT region, 'ALL', ROUND(SUM(sales_amount)/1e5,2) FROM (SELECT s.region, f.sales_amount {JOIN}) GROUP BY region
  UNION ALL SELECT 'ALL', 'ALL', ROUND(SUM(sales_amount)/1e5,2) FROM (SELECT f.sales_amount {JOIN})
  ORDER BY region, city""")
print(r2.to_string(index=False))

print("\n(2) DRILL-DOWN  year -> quarter -> month for 2025, category = Electronics")
d = q(
    f"SELECT t.quarter, t.month_name, ROUND(SUM(f.sales_amount)/1e5,2) AS sales_lakh {JOIN} "
    f"WHERE t.year=2025 AND p.category='Electronics' GROUP BY t.quarter, t.month ORDER BY t.month"
)
print(d.to_string(index=False))

print("\n(3) SLICE  Time = 2025-Q1 (one dimension fixed)  -> Location x Product")
sl = q(
    f"SELECT s.city, p.category, ROUND(SUM(f.sales_amount)/1e3,1) AS sales_thousand {JOIN} "
    f"WHERE t.year=2025 AND t.quarter='Q1' GROUP BY s.city, p.category"
)
sl_p = sl.pivot(index="city", columns="category", values="sales_thousand").fillna(0)
print(sl_p.to_string())

print("\n(4) DICE  city IN (Pune, Mumbai) AND quarter IN (Q1, Q2) AND category IN (Electronics, Clothing)")
di = q(
    f"SELECT t.year, t.quarter, s.city, p.category, SUM(f.qty) AS units, "
    f"ROUND(SUM(f.sales_amount)/1e3,1) AS sales_thousand {JOIN} "
    f"WHERE s.city IN ('Pune','Mumbai') AND t.quarter IN ('Q1','Q2') "
    f"AND p.category IN ('Electronics','Clothing') GROUP BY t.year, t.quarter, s.city, p.category "
    f"ORDER BY t.year, t.quarter, s.city"
)
print(di.to_string(index=False))

print("\n(5) PIVOT  rows = category, columns = region  (then rotated)")
pv = (
    cube_df.pivot_table(index="category", columns="region", values="sales_amount", aggfunc="sum")
    .div(1e5)
    .round(2)
)
print(pv.to_string())
print("\nRotated (rows = region, columns = category):")
print(pv.T.to_string())

# ---------------------------------------------------------------
# STEP 6: BUSINESS QUERIES ON THE DATA MART
# ---------------------------------------------------------------
print("\nQ1. Top 3 products by profit")
print(
    q(
        f"SELECT p.product_name, ROUND(SUM(f.profit)/1e3,1) AS profit_thousand {JOIN} "
        f"GROUP BY p.product_name ORDER BY SUM(f.profit) DESC LIMIT 3"
    ).to_string(index=False)
)
print("\nQ2. Year-on-year growth by region")
yoy = q(f"SELECT s.region, t.year, SUM(f.sales_amount) AS sales {JOIN} GROUP BY s.region, t.year")
yoy = yoy.pivot(index="region", columns="year", values="sales")
yoy["growth_%"] = ((yoy[2025] - yoy[2024]) / yoy[2024] * 100).round(1)
print((yoy[[2024, 2025]] / 1e5).round(2).join(yoy["growth_%"]).to_string())
print("\nQ3. Share of sales by customer segment")
seg = q(f"SELECT c.segment, SUM(f.sales_amount) AS sales {JOIN} GROUP BY c.segment")
seg["share_%"] = (seg.sales / seg.sales.sum() * 100).round(1)
print(seg[["segment", "share_%"]].to_string(index=False))

# ---------------------------------------------------------------
# STEP 7: VISUALISATION
# ---------------------------------------------------------------
C = ["#1F3A5F", "#2A9D8F", "#E9C46A", "#E76F51", "#8A4FBF"]
fig, ax = plt.subplots(1, 3, figsize=(9, 2.8), sharey=True)
cats = ["Electronics", "Clothing", "Grocery"]
for k, cat in enumerate(cats):
    m = (
        cube_df[cube_df.category == cat]
        .pivot_table(index="city", columns="quarter", values="sales_amount", aggfunc="sum")
        .div(1e5)
    )
    im = ax[k].imshow(m.values, cmap="YlGnBu", aspect="auto")
    ax[k].set_xticks(range(4))
    ax[k].set_xticklabels(m.columns)
    ax[k].set_yticks(range(len(m)))
    ax[k].set_yticklabels(m.index)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax[k].text(
                j,
                i,
                f"{m.values[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if m.values[i, j] > m.values.max() * 0.6 else "black",
            )
    ax[k].set_title(f"Item = {cat}")
fig.suptitle("Data cube shown as slices: City x Quarter (sales, Rs lakh) for each Item", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/a2_cube.png")
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
rr = r.pivot(index="region", columns="year", values="sales_lakh")
rr.plot(kind="bar", ax=ax[0], color=C[:2], rot=0)
ax[0].set_title("Roll-up: sales by Region x Year (lakh)")
ax[0].set_xlabel("")
ax[1].plot(d.month_name, d.sales_lakh, marker="o", color=C[3])
ax[1].set_title("Drill-down: Electronics 2025 by month")
ax[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig(f"{OUT}/a2_rollup_drill.png")
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
sl_p.plot(kind="barh", stacked=True, ax=ax[0], color=C[:3])
ax[0].set_title("Slice: 2025-Q1 sales (thousand)")
ax[0].set_ylabel("")
im = ax[1].imshow(pv.values, cmap="Oranges")
ax[1].set_xticks(range(len(pv.columns)))
ax[1].set_xticklabels(pv.columns)
ax[1].set_yticks(range(len(pv.index)))
ax[1].set_yticklabels(pv.index)
for i in range(pv.shape[0]):
    for j in range(pv.shape[1]):
        ax[1].text(j, i, pv.values[i, j], ha="center", va="center", fontsize=8)
ax[1].set_title("Pivot: Category x Region (lakh)")
plt.tight_layout()
plt.savefig(f"{OUT}/a2_slice_pivot.png")
plt.close()
con.close()
print("\nData mart saved to data/sales_mart.db")
