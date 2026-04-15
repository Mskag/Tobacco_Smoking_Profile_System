# ============================================================
#  Tobacco Use and Mortality Analysis  |  2004–2015
#  Datasets: admissions, fatalities, metrics, prescriptions,
#            smokers
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="Set2")
plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

# ============================================================
#  1. LOAD DATA
# ============================================================

admissions    = pd.read_csv(""r"D:\dissertation\archive\admissions.csv")
fatalities    = pd.read_csv(""r"D:\dissertation\archive\fatalities.csv")
metrics       = pd.read_csv(""r"D:\dissertation\archive\metrics.csv")
prescriptions = pd.read_csv(""r"D:\dissertation\archive\prescriptions.csv")
smokers       = pd.read_csv(""r"D:\dissertation\archive\smokers.csv")

# Coerce Value columns to numeric (they may be read as strings)
for _df in [admissions, fatalities]:
    _df["Value"] = pd.to_numeric(_df["Value"].astype(str).str.replace(",", ""), errors="coerce")

# ── Clean column names (strip embedded newlines from metrics) ─
metrics.columns       = [c.replace("\n", " ").strip() for c in metrics.columns]
prescriptions.columns = [c.replace("\n", " ").strip() for c in prescriptions.columns]


# ============================================================
#  2. HELPER – extract a fiscal year start as integer
#     "2014/15" → 2014,  2014 → 2014
# ============================================================

def parse_year(series):
    return series.astype(str).str[:4].astype(int)


# ============================================================
#  3. ADMISSIONS ANALYSIS
# ============================================================

# 3a. Total smoking-attributable admissions over time (all sexes)
smoke_adm = admissions[
    (admissions["ICD10 Diagnosis"].str.contains("caused by smoking", case=False, na=False)) &
    (admissions["Metric"] == "Number of admissions") &
    (admissions["Sex"].isna())
].copy()
smoke_adm["YearInt"] = parse_year(smoke_adm["Year"])
smoke_adm = smoke_adm.sort_values("YearInt")

# 3b. Admissions by sex
smoke_adm_sex = admissions[
    (admissions["ICD10 Diagnosis"].str.contains("caused by smoking", case=False, na=False)) &
    (admissions["Metric"] == "Number of admissions") &
    (admissions["Sex"].notna())
].copy()
smoke_adm_sex["YearInt"] = parse_year(smoke_adm_sex["Year"])
smoke_adm_sex = smoke_adm_sex.sort_values("YearInt")

# 3c. Top 5 diagnosis categories (most recent year, total admissions, attributable)
latest_adm_year = smoke_adm["Year"].iloc[-1]
top_diag = admissions[
    (admissions["Year"] == latest_adm_year) &
    (admissions["Metric"] == "Attributable number") &
    (admissions["Sex"].isna()) &
    (~admissions["ICD10 Diagnosis"].str.contains("All", case=False, na=False))
].nlargest(6, "Value")


# ============================================================
#  4. FATALITIES ANALYSIS
# ============================================================

# 4a. Total smoking-attributable deaths over time
smoke_fat = fatalities[
    (fatalities["ICD10 Diagnosis"].str.contains("caused by smoking", case=False, na=False)) &
    (fatalities["Metric"] == "Number of observed deaths") &
    (fatalities["Sex"].isna())
].copy()
smoke_fat["YearInt"] = parse_year(smoke_fat["Year"].astype(str))
smoke_fat = smoke_fat.sort_values("YearInt")

# 4b. Deaths by sex
smoke_fat_sex = fatalities[
    (fatalities["ICD10 Diagnosis"].str.contains("caused by smoking", case=False, na=False)) &
    (fatalities["Metric"] == "Number of observed deaths") &
    (fatalities["Sex"].notna())
].copy()
smoke_fat_sex["YearInt"] = parse_year(smoke_fat_sex["Year"].astype(str))
smoke_fat_sex = smoke_fat_sex.sort_values("YearInt")

# 4c. Top cause-of-death categories (attributable, latest year)
latest_fat_year = fatalities["Year"].max()
top_cause = fatalities[
    (fatalities["Year"] == latest_fat_year) &
    (fatalities["Metric"] == "Attributable number") &
    (fatalities["Sex"].isna()) &
    (~fatalities["ICD10 Diagnosis"].str.contains("All", case=False, na=False))
].nlargest(6, "Value")


# ============================================================
#  5. SMOKER PREVALENCE
# ============================================================

prev = smokers[
    (smokers["Method"] == "Weighted") &
    (smokers["Sex"].isna())
].copy().sort_values("Year")

prev_sex = smokers[
    (smokers["Method"] == "Weighted") &
    (smokers["Sex"].notna())
].copy().sort_values("Year")


# ============================================================
#  6. PRESCRIPTIONS
# ============================================================

rx = prescriptions.copy()
rx["YearInt"] = parse_year(rx["Year"])
rx = rx.sort_values("YearInt")


# ============================================================
#  7. ECONOMIC METRICS
# ============================================================

eco = metrics.copy().sort_values("Year")


# ============================================================
#  8. CORRELATION – admissions vs fatalities (overlapping years)
# ============================================================

adm_annual = smoke_adm[["YearInt", "Value"]].rename(columns={"Value": "Admissions"})
fat_annual  = smoke_fat[["YearInt", "Value"]].rename(columns={"Value": "Deaths"})
combined    = pd.merge(adm_annual, fat_annual, on="YearInt")


# ============================================================
#  9. PLOT EVERYTHING  (3 figures, 10 panels total)
# ============================================================

# ── Figure 1 : Admissions & Fatalities ──────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9))
fig1.suptitle("Tobacco-Attributable Hospital Admissions & Deaths (England)", fontsize=13, fontweight="bold")

# Panel A – total admissions trend
ax = axes1[0, 0]
ax.plot(smoke_adm["YearInt"], smoke_adm["Value"] / 1_000, marker="o", linewidth=2, color="#2196F3")
ax.fill_between(smoke_adm["YearInt"], smoke_adm["Value"] / 1_000, alpha=0.15, color="#2196F3")
ax.set_title("A. Total Smoking-Attributable Admissions")
ax.set_xlabel("Year")
ax.set_ylabel("Admissions (thousands)")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Panel B – admissions by sex
ax = axes1[0, 1]
for sex, grp in smoke_adm_sex.groupby("Sex"):
    ax.plot(grp["YearInt"], grp["Value"] / 1_000, marker="o", linewidth=2, label=sex)
ax.set_title("B. Admissions by Sex")
ax.set_xlabel("Year")
ax.set_ylabel("Admissions (thousands)")
ax.legend()
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Panel C – total deaths trend
ax = axes1[1, 0]
ax.plot(smoke_fat["YearInt"], smoke_fat["Value"] / 1_000, marker="s", linewidth=2, color="#F44336")
ax.fill_between(smoke_fat["YearInt"], smoke_fat["Value"] / 1_000, alpha=0.15, color="#F44336")
ax.set_title("C. Total Smoking-Attributable Deaths")
ax.set_xlabel("Year")
ax.set_ylabel("Deaths (thousands)")
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Panel D – deaths by sex
ax = axes1[1, 1]
for sex, grp in smoke_fat_sex.groupby("Sex"):
    ax.plot(grp["YearInt"], grp["Value"] / 1_000, marker="s", linewidth=2, label=sex)
ax.set_title("D. Deaths by Sex")
ax.set_xlabel("Year")
ax.set_ylabel("Deaths (thousands)")
ax.legend()
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

fig1.tight_layout()


# ── Figure 2 : Disease Breakdown & Correlation ──────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6))
fig2.suptitle("Disease Breakdown & Admissions–Mortality Correlation", fontsize=13, fontweight="bold")

# Panel E – top admissions by diagnosis (horizontal bar)
ax = axes2[0]
if not top_diag.empty:
    ax.barh(top_diag["ICD10 Diagnosis"], top_diag["Value"] / 1_000, color="#42A5F5")
    ax.set_xlabel("Attributable Admissions (thousands)")
    ax.set_title(f"E. Top Diagnoses – Admissions\n({latest_adm_year})")
    ax.invert_yaxis()
else:
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("E. Top Diagnoses – Admissions")

# Panel F – top causes of death (horizontal bar)
ax = axes2[1]
if not top_cause.empty:
    ax.barh(top_cause["ICD10 Diagnosis"], top_cause["Value"] / 1_000, color="#EF5350")
    ax.set_xlabel("Attributable Deaths (thousands)")
    ax.set_title(f"F. Top Causes of Death\n({latest_fat_year})")
    ax.invert_yaxis()
else:
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("F. Top Causes of Death")

# Panel G – scatter: admissions vs deaths
ax = axes2[2]
if not combined.empty:
    corr = combined["Admissions"].corr(combined["Deaths"])
    ax.scatter(combined["Admissions"] / 1_000, combined["Deaths"] / 1_000,
               c=combined["YearInt"], cmap="viridis", s=80, zorder=3)
    # trend line
    z = combined[["Admissions", "Deaths"]].dropna()
    m, b = pd.Series(z["Deaths"].values).pipe(
        lambda _: (
            (z["Deaths"].cov(z["Admissions"]) / z["Admissions"].var()),
            z["Deaths"].mean() - (z["Deaths"].cov(z["Admissions"]) / z["Admissions"].var()) * z["Admissions"].mean()
        )
    )
    import numpy as np
    xr = np.linspace(z["Admissions"].min(), z["Admissions"].max(), 50)
    ax.plot(xr / 1_000, (m * xr + b) / 1_000, "--", color="grey", linewidth=1.5)
    ax.set_xlabel("Admissions (thousands)")
    ax.set_ylabel("Deaths (thousands)")
    ax.set_title(f"G. Admissions vs Deaths\n(r = {corr:.2f})")
else:
    ax.text(0.5, 0.5, "Insufficient overlapping years", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("G. Admissions vs Deaths")

fig2.tight_layout()


# ── Figure 3 : Smokers, Prescriptions, Economics ────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 6))
fig3.suptitle("Smoker Prevalence, Prescriptions & Tobacco Economics", fontsize=13, fontweight="bold")

# Panel H – smoking prevalence by age group (total)
ax = axes3[0]
age_cols = ["16 and Over", "16-24", "25-34", "35-49", "50-59", "60 and Over"]
for col in age_cols:
    if col in prev.columns:
        ax.plot(prev["Year"], prev[col], marker=".", linewidth=1.5, label=col)
ax.set_title("H. Smoking Prevalence by Age Group (%)")
ax.set_xlabel("Year")
ax.set_ylabel("Prevalence (%)")
ax.legend(fontsize=7, ncol=2)

# Panel I – prescriptions over time
ax = axes3[1]
rx_plot = rx.dropna(subset=["YearInt"])
ax.bar(rx_plot["YearInt"], rx_plot["All Pharmacotherapy Prescriptions"],
       color="#66BB6A", label="Total")
ax.plot(rx_plot["YearInt"],
        rx_plot["Nicotine Replacement Therapy (NRT) Prescriptions"],
        marker="o", color="#1B5E20", label="NRT", linewidth=2)
ax.plot(rx_plot["YearInt"],
        rx_plot["Varenicline (Champix) Prescriptions"],
        marker="^", color="#388E3C", label="Varenicline", linewidth=2)
ax.set_title("I. Smoking Cessation Prescriptions")
ax.set_xlabel("Fiscal Year Start")
ax.set_ylabel("Prescriptions (thousands)")
ax.legend(fontsize=8)
ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

# Panel J – tobacco affordability index vs prevalence (dual axis)
ax  = axes3[2]
ax2 = ax.twinx()
col_afford = "Affordability of Tobacco Index"
if col_afford in eco.columns and not prev.empty:
    merged_eco = pd.merge(
        eco[["Year", col_afford]],
        prev[["Year", "16 and Over"]],
        on="Year", how="inner"
    )
    l1, = ax.plot(merged_eco["Year"], merged_eco[col_afford],
                  color="#7E57C2", marker="o", linewidth=2, label="Affordability Index")
    l2, = ax2.plot(merged_eco["Year"], merged_eco["16 and Over"],
                   color="#FF7043", marker="s", linewidth=2, linestyle="--", label="Prevalence 16+ (%)")
    ax.set_ylabel("Affordability Index", color="#7E57C2")
    ax2.set_ylabel("Prevalence (%)", color="#FF7043")
    ax.set_title("J. Tobacco Affordability vs Smoking Prevalence")
    ax.set_xlabel("Year")
    lines = [l1, l2]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=8)

fig3.tight_layout()


# ============================================================
#  10. SUMMARY TABLE
# ============================================================

print("\n" + "=" * 60)
print("  SUMMARY: Smoking-Attributable Burden")
print("=" * 60)
print("\nAdmissions (thousands):")
print(smoke_adm[["YearInt", "Value"]].rename(columns={"YearInt": "Year", "Value": "Admissions"})
      .assign(Admissions=lambda x: (x["Admissions"] / 1_000).round(1))
      .set_index("Year").to_string())

print("\nDeaths (thousands):")
print(smoke_fat[["YearInt", "Value"]].rename(columns={"YearInt": "Year", "Value": "Deaths"})
      .assign(Deaths=lambda x: (x["Deaths"] / 1_000).round(1))
      .set_index("Year").to_string())

if not combined.empty:
    corr_val = combined["Admissions"].corr(combined["Deaths"])
    print(f"\nCorrelation (admissions vs deaths): r = {corr_val:.3f}")

print("\nSmoking Prevalence (16+, Weighted %):")
print(prev[["Year", "16 and Over"]].set_index("Year").to_string())

print("\nTotal Prescriptions by Year:")
print(rx[["YearInt", "All Pharmacotherapy Prescriptions"]]
      .rename(columns={"YearInt": "Year"})
      .set_index("Year").to_string())

print("\n" + "=" * 60)
print("  All charts rendered. Close figure windows to exit.")
print("=" * 60 + "\n")

plt.show()