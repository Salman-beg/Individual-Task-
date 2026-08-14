import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

OUT = "/sessions/relaxed-clever-noether/mnt/outputs/"

with open(OUT + "results.json") as f:
    res = json.load(f)

# --- Rossmann: actual vs predicted ---
ra = pd.read_csv(OUT + "rossmann_test_predictions.csv", parse_dates=["Date"])
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(ra["Date"], ra["AvgSalesPerStore"], label="Actual", color="#1b1b1b", linewidth=1.6)
ax.plot(ra["Date"], ra["RF_pred"], label="Random Forest", color="#2f7ed8", linewidth=1.3, alpha=0.85)
ax.plot(ra["Date"], ra["kNN_pred"], label="kNN", color="#d8572f", linewidth=1.3, alpha=0.85)
ax.set_title("Rossmann: Average Sales per Open Store (test period)")
ax.set_ylabel("EUR")
ax.set_xlabel("Date")
ax.legend()
fig.tight_layout()
fig.savefig(OUT + "fig_rossmann_actual_vs_pred.png", dpi=150)
plt.close(fig)

# --- Retail II: actual vs predicted ---
rb = pd.read_csv(OUT + "retail_test_predictions.csv", parse_dates=["Date"])
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(rb["Date"], rb["TotalRevenue"], label="Actual", color="#1b1b1b", linewidth=1.6)
ax.plot(rb["Date"], rb["RF_pred"], label="Random Forest", color="#2f7ed8", linewidth=1.3, alpha=0.85)
ax.plot(rb["Date"], rb["kNN_pred"], label="kNN", color="#d8572f", linewidth=1.3, alpha=0.85)
ax.set_title("Online Retail II: Total Daily Revenue (test period)")
ax.set_ylabel("GBP")
ax.set_xlabel("Date")
ax.legend()
fig.tight_layout()
fig.savefig(OUT + "fig_retail_actual_vs_pred.png", dpi=150)
plt.close(fig)

# --- Feature importance ---
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, key, title in zip(axes, ["rossmann", "retail_ii"],
                           ["Rossmann: RF Feature Importance", "Online Retail II: RF Feature Importance"]):
    fi = res[key]["RF_feature_importance"]
    names = list(fi.keys())
    vals = list(fi.values())
    ax.barh(names[::-1], vals[::-1], color="#2f7ed8")
    ax.set_title(title)
    ax.set_xlabel("Importance")
fig.tight_layout()
fig.savefig(OUT + "fig_feature_importance.png", dpi=150)
plt.close(fig)

print("plots done")
