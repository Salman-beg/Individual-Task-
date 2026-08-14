"""
Case Studies in Data Science - Individual Task 1, Part 1.3
Data Analysis: Random Forest Regressor + k-Nearest Neighbours Regressor
applied to (A) Rossmann Store Sales and (B) UCI Online Retail II.

Both datasets are aggregated to a DAILY TOTAL SALES/REVENUE time series so
that the two independently-sourced retail businesses can be modelled with
the same feature schema (calendar + promotional signals known in advance)
and compared on a like-for-like basis, echoing a Power BI Developer's task
of building daily/weekly sales-performance dashboards.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json

UP = "/sessions/relaxed-clever-noether/mnt/uploads/"
OUT = "/sessions/relaxed-clever-noether/mnt/outputs/"

results = {}

# -------------------------------------------------------------------
# DATASET A: ROSSMANN STORE SALES -> daily total sales across all stores
# -------------------------------------------------------------------
train = pd.read_csv(UP + "train.csv", dtype={"StateHoliday": str}, low_memory=False)
store = pd.read_csv(UP + "store.csv")
train["Date"] = pd.to_datetime(train["Date"])

# keep only days the store was open (closed days trivially have Sales=0)
train_open = train[train["Open"] == 1].copy()

daily_a = train_open.groupby("Date").agg(
    TotalSales=("Sales", "sum"),
    TotalCustomers=("Customers", "sum"),
    NumStoresOpen=("Store", "nunique"),
    PromoShare=("Promo", "mean"),
    SchoolHolidayShare=("SchoolHoliday", "mean"),
    AnyStateHoliday=("StateHoliday", lambda s: int((s != "0").any())),
).reset_index().sort_values("Date")

daily_a["DayOfWeek"] = daily_a["Date"].dt.dayofweek
daily_a["Month"] = daily_a["Date"].dt.month
daily_a["IsWeekend"] = (daily_a["DayOfWeek"] >= 5).astype(int)
# Normalise out the mechanical effect of "how many stores happened to be open"
# (driven almost entirely by weekday store-closure rules) so the model reflects
# genuine demand drivers (promotions, seasonality) rather than network size.
daily_a["AvgSalesPerStore"] = daily_a["TotalSales"] / daily_a["NumStoresOpen"]

feat_a = ["DayOfWeek", "Month", "IsWeekend", "PromoShare", "SchoolHolidayShare",
          "AnyStateHoliday"]
target_a = "AvgSalesPerStore"

daily_a = daily_a.dropna(subset=feat_a + [target_a])
n_a = len(daily_a)
split_a = int(n_a * 0.8)
train_a, test_a = daily_a.iloc[:split_a], daily_a.iloc[split_a:]

Xtr_a, ytr_a = train_a[feat_a], train_a[target_a]
Xte_a, yte_a = test_a[feat_a], test_a[target_a]

scaler_a = StandardScaler().fit(Xtr_a)
Xtr_a_s, Xte_a_s = scaler_a.transform(Xtr_a), scaler_a.transform(Xte_a)

rf_a = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf_a.fit(Xtr_a, ytr_a)
pred_rf_a = rf_a.predict(Xte_a)

knn_a = KNeighborsRegressor(n_neighbors=7, weights="distance")
knn_a.fit(Xtr_a_s, ytr_a)
pred_knn_a = knn_a.predict(Xte_a_s)

def metrics(y_true, y_pred):
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MeanTarget": float(np.mean(y_true)),
    }

results["rossmann"] = {
    "n_days_total": int(n_a),
    "n_train_days": int(len(train_a)),
    "n_test_days": int(len(test_a)),
    "date_range": [str(daily_a["Date"].min().date()), str(daily_a["Date"].max().date())],
    "RandomForest": metrics(yte_a, pred_rf_a),
    "kNN": metrics(yte_a, pred_knn_a),
    "RF_feature_importance": dict(sorted(
        zip(feat_a, rf_a.feature_importances_.tolist()), key=lambda x: -x[1])),
}

# -------------------------------------------------------------------
# DATASET B: ONLINE RETAIL II -> daily total revenue across all countries
# -------------------------------------------------------------------
sheets = pd.read_excel(UP + "online_retail_II.xlsx", sheet_name=None)
retail = pd.concat(sheets.values(), ignore_index=True)
retail.columns = [c.strip() for c in retail.columns]

# remove cancellations (Invoice starting with 'C') and non-positive qty/price
retail["Invoice"] = retail["Invoice"].astype(str)
retail = retail[~retail["Invoice"].str.startswith("C")]
retail = retail[(retail["Quantity"] > 0) & (retail["Price"] > 0)]
retail["Revenue"] = retail["Quantity"] * retail["Price"]
retail["Date"] = pd.to_datetime(retail["InvoiceDate"]).dt.date
retail["Date"] = pd.to_datetime(retail["Date"])

daily_b = retail.groupby("Date").agg(
    TotalRevenue=("Revenue", "sum"),
    NumInvoices=("Invoice", "nunique"),
    NumCustomers=("Customer ID", "nunique"),
    UKShareRevenue=("Country", lambda s: None),  # placeholder, computed below
).reset_index().sort_values("Date")

# UK revenue share per day (proxy for market-mix signal, known pattern not leakage-prone)
uk_rev = retail[retail["Country"] == "United Kingdom"].groupby("Date")["Revenue"].sum()
tot_rev = retail.groupby("Date")["Revenue"].sum()
uk_share = (uk_rev / tot_rev).reindex(daily_b["Date"]).fillna(0).values
daily_b["UKShareRevenue"] = uk_share

daily_b["DayOfWeek"] = daily_b["Date"].dt.dayofweek
daily_b["Month"] = daily_b["Date"].dt.month
daily_b["IsWeekend"] = (daily_b["DayOfWeek"] >= 5).astype(int)
daily_b["IsDecember"] = (daily_b["Month"] == 12).astype(int)

feat_b = ["DayOfWeek", "Month", "IsWeekend", "IsDecember", "UKShareRevenue"]
target_b = "TotalRevenue"

daily_b = daily_b.dropna(subset=feat_b + [target_b])
n_b = len(daily_b)
split_b = int(n_b * 0.8)
train_b, test_b = daily_b.iloc[:split_b], daily_b.iloc[split_b:]

Xtr_b, ytr_b = train_b[feat_b], train_b[target_b]
Xte_b, yte_b = test_b[feat_b], test_b[target_b]

scaler_b = StandardScaler().fit(Xtr_b)
Xtr_b_s, Xte_b_s = scaler_b.transform(Xtr_b), scaler_b.transform(Xte_b)

rf_b = RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42, n_jobs=-1)
rf_b.fit(Xtr_b, ytr_b)
pred_rf_b = rf_b.predict(Xte_b)

knn_b = KNeighborsRegressor(n_neighbors=7, weights="distance")
knn_b.fit(Xtr_b_s, ytr_b)
pred_knn_b = knn_b.predict(Xte_b_s)

results["retail_ii"] = {
    "n_days_total": int(n_b),
    "n_train_days": int(len(train_b)),
    "n_test_days": int(len(test_b)),
    "date_range": [str(daily_b["Date"].min().date()), str(daily_b["Date"].max().date())],
    "RandomForest": metrics(yte_b, pred_rf_b),
    "kNN": metrics(yte_b, pred_knn_b),
    "RF_feature_importance": dict(sorted(
        zip(feat_b, rf_b.feature_importances_.tolist()), key=lambda x: -x[1])),
}

with open(OUT + "results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save the daily aggregated series + predictions for plotting
test_a_out = test_a[["Date", target_a]].copy()
test_a_out["RF_pred"] = pred_rf_a
test_a_out["kNN_pred"] = pred_knn_a
test_a_out.to_csv(OUT + "rossmann_test_predictions.csv", index=False)

test_b_out = test_b[["Date", target_b]].copy()
test_b_out["RF_pred"] = pred_rf_b
test_b_out["kNN_pred"] = pred_knn_b
test_b_out.to_csv(OUT + "retail_test_predictions.csv", index=False)

daily_a.to_csv(OUT + "rossmann_daily_agg.csv", index=False)
daily_b.to_csv(OUT + "retail_daily_agg.csv", index=False)

print(json.dumps(results, indent=2))
