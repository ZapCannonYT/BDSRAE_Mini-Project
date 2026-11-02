"""Generate notebooks/results.ipynb. Run it, then execute the notebook with nbconvert.

The notebook only reads CSVs in results/. Sections whose inputs do not exist are skipped.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.strip().splitlines(True)}


CELLS = [
    md("""
# Results

Every figure and table in the report is built here from the CSVs in `results/`. This notebook never
trains models or runs detectors itself; rerun the scripts in `experiments/` to refresh the inputs.
"""),
    code("""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd() if (Path.cwd() / "results").exists() else Path.cwd().parent
R, FIG = ROOT / "results", ROOT / "docs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
FAMILY = {"adwin": COLORS[0], "adwin_demand": COLORS[1], "psi": COLORS[2], "page_hinkley": COLORS[3], "kswin": COLORS[4]}
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "lines.linewidth": 1.6, "legend.frameon": False})
S = {}  # headline numbers, saved for the report and the slides

def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.show()
"""),
    md("## 1. Forecasting model against the baseline"),
    code("""
base = pd.read_csv(R / "model" / "baseline_metrics.csv")
display(base.round(2))
city = base[base.scope == "city"].set_index("model")
S["model_wape"], S["naive_wape"] = float(city.wape["lightgbm"]), float(city.wape["seasonal_naive"])
"""),
    md("## 2. Big data measurements"),
    code("""
p = R / "scaling" / "batch_scaling.csv"
if p.exists():
    b = pd.read_csv(p).groupby(["workers", "cores"], as_index=False).seconds.median()
    b["speedup"] = b.seconds.iloc[0] / b.seconds
    display(b.round(2))
    S["batch"] = b.round(2).to_dict("records")
    fig, ax = plt.subplots(figsize=(4.6, 2.8))
    ax.bar(b.workers.astype(str), b.seconds, color=COLORS[0], width=0.5)
    for x, v in zip(range(len(b)), b.seconds):
        ax.annotate(f"{v:.0f}s", (x, v), ha="center", va="bottom", xytext=(0, 2), textcoords="offset points")
    ax.set(xlabel="Spark workers (2 cores each)", ylabel="Seconds", title="Zone-hour batch build, 2019-2021")
    ax.grid(axis="x", visible=False)
    save(fig, "batch_scaling")
"""),
    code("""
for name in ["stream_throughput", "storage", "pruning", "detector_cost", "stream_latency"]:
    p = R / "scaling" / f"{name}.csv"
    if p.exists():
        d = pd.read_csv(p)
        if name == "pruning":
            d = d.groupby("query", as_index=False).seconds.median()
        if name == "stream_latency":
            d = d.groupby("speedup", as_index=False).processing_latency.describe()
        print(name)
        display(d.round(2))
        S[name] = json.loads(d.round(3).to_json(orient="records"))
p = R / "stream" / "spark_progress.csv"
if p.exists():
    sp = pd.read_csv(p)
    sp = sp[sp.input_rows > 0]
    S["stream_demo_rate_median"] = float(sp.processed_rate.median())
    S["stream_demo_rows"] = int(sp.input_rows.sum())
    print("demo stream: rows", S["stream_demo_rows"], "median processed rows/s", round(S["stream_demo_rate_median"]))
"""),
    md("## 3. Detector comparison (synthetic drift with a known change point)"),
    code("""
bench = pd.read_csv(R / "detectors" / "benchmark.csv")
# False alarms: no-drift runs, city stream
nd = bench[(bench.scenario == "no_drift") & (bench.stream_scope == "city")]
fa = nd.groupby(["family", "detector"], as_index=False).agg(alarms=("false_alarms", "sum"), hours=("hours_watched", "sum"))
fa["false_alarms_per_1000h"] = 1000 * fa.alarms / fa.hours
# Detection: city-wide drift, city stream
dr = bench[(bench.target == "city") & (bench.stream_scope == "city")]
det = dr.groupby("detector").agg(detection_rate=("detected", "mean"), median_delay_h=("delay", "median"))
table = fa.merge(det, on="detector")[["family", "detector", "false_alarms_per_1000h", "detection_rate", "median_delay_h"]]
display(table.round(2))
S["detectors"] = json.loads(table.round(3).to_json(orient="records"))

fig, ax = plt.subplots(figsize=(5.6, 3.2))
for fam, g in table.groupby("family"):
    ax.scatter(g.false_alarms_per_1000h, g.median_delay_h, s=38, color=FAMILY.get(fam, "#777"), label=fam,
               edgecolor="white", linewidth=1.2, zorder=3)
ax.set(xlabel="False alarms per 1000 hours (no-drift runs)", ylabel="Median detection delay (hours)",
       title="Detection delay against false alarms, city-wide drift")
ax.legend(fontsize=8)
save(fig, "detector_tradeoff")

by = dr.pivot_table(index="scenario", columns="family", values="delay", aggfunc="median").round(0)
rate = dr.pivot_table(index="scenario", columns="family", values="detected", aggfunc="mean").round(2)
print("median delay (h) by scenario"); display(by)
print("detection rate by scenario"); display(rate)
S["delay_by_scenario"] = json.loads(by.reset_index().to_json(orient="records"))
S["rate_by_scenario"] = json.loads(rate.reset_index().to_json(orient="records"))
"""),
    md("## 4. Real drift: COVID-19, fixed 2019 model"),
    code("""
cov = pd.read_csv(R / "detectors" / "covid_alarms.csv")
display(cov[cov.scope == "city"].round(1))
S["covid_city"] = json.loads(cov[cov.scope == "city"].round(1).to_json(orient="records"))
"""),
    md("## 5. Retraining policies"),
    code("""
summ = pd.read_csv(R / "retraining" / "summary.csv")
cost = summ[summ.period == "cost"][["policy", "retrains", "promoted", "retrain_seconds"]]
err = summ[summ.period != "cost"]
piv = err[err.scope == "city"].pivot(index="policy", columns="period", values="wape").round(2)
display(piv.join(cost.set_index("policy")).round(1))
S["policies"] = json.loads(piv.join(cost.set_index("policy")).round(2).reset_index().to_json(orient="records"))

order = ["never", "periodic", "adwin", "psi", "combined", "adwin_any_borough", "oracle"]
fig, ax = plt.subplots(figsize=(6.8, 3.2))
for i, name in enumerate(order):
    f = R / "retraining" / f"{name}_daily.csv"
    if not f.exists():
        continue
    d = pd.read_csv(f, parse_dates=["day"])
    m = d[d.scope == "city"].set_index("day")[["abs_err", "demand"]].resample("MS").sum()
    ax.plot(m.index, 100 * m.abs_err / m.demand, color=COLORS[i], label=name)
ax.axvline(pd.Timestamp("2020-03-22"), color="#52514e", linestyle=":", linewidth=1)
ax.annotate("NY PAUSE", (pd.Timestamp("2020-03-22"), ax.get_ylim()[1]), fontsize=8, ha="left", va="top", xytext=(3, 0), textcoords="offset points")
ax.set(ylabel="WAPE (%)", title="Monthly city-wide error by retraining policy")
ax.legend(ncol=4, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
save(fig, "policy_error")
"""),
    md("## 6. Fairness across boroughs"),
    code("""
# Does drift hurt some boroughs more? Fixed model, error before against during the crash.
never = err[(err.policy == "never") & (err.scope != "city")]
w = never.pivot(index="scope", columns="period", values="wape")
w["crash_over_before"] = w["crash (Mar-Jun 2020)"] / w["before (Oct 2019-Feb 2020)"]
display(w.round(2))
S["borough_never"] = json.loads(w.round(2).reset_index().to_json(orient="records"))

best = cost.merge(piv.reset_index(), on="policy")
best = best[~best.policy.isin(["oracle", "never"])].sort_values("crash (Mar-Jun 2020)").policy.iloc[0]
S["best_policy"] = best
period = "crash (Mar-Jun 2020)"
gap = err[(err.period == period) & (err.scope != "city")].pivot(index="policy", columns="scope", values="wape")
gap["worst_minus_best"] = gap.max(axis=1) - gap.min(axis=1)
display(gap.round(1))
S["gap"] = json.loads(gap.round(2).reset_index().to_json(orient="records"))

boroughs = list(w.index)
fig, ax = plt.subplots(figsize=(6.2, 3.1))
x = range(len(boroughs))
vals = [("before, no retraining", w["before (Oct 2019-Feb 2020)"], "#9ec5f4"), ("crash, no retraining", w[period], COLORS[1]),
        (f"crash, {best}", gap.loc[best, boroughs], COLORS[0])]
for i, (lab, v, c) in enumerate(vals):
    ax.bar([k + (i - 1) * 0.27 for k in x], v, width=0.25, label=lab, color=c)
ax.set_xticks(list(x), boroughs)
ax.set(ylabel="WAPE (%)", title="Error by borough")
ax.grid(axis="x", visible=False)
ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
save(fig, "borough_error")
"""),
    code("""
# Does a city-wide detector miss drift confined to one borough?
one = bench[(bench.target.isin(boroughs)) & (bench.target != "none")]
one = one.assign(detector_view=(one.stream_scope == "city").map({True: "city-wide detector", False: "borough's own detector"}))
adw = one[one.detector == "adwin_d0.002"]
psi_ = one[one.detector == "psi_t0.25"]
rows = []
for lab, part in [("adwin_d0.002", adw), ("psi_t0.25", psi_)]:
    t = part.groupby(["target", "detector_view"]).detected.mean().unstack()
    t["detector"] = lab
    rows.append(t.reset_index())
view = pd.concat(rows)
display(view.round(2))
S["city_vs_borough"] = json.loads(view.round(3).to_json(orient="records"))
pv = view[view.detector == "adwin_d0.002"].set_index("target").reindex(boroughs)
fig, ax = plt.subplots(figsize=(6.2, 2.9))
ax.bar([k - 0.17 for k in x], pv["city-wide detector"], width=0.32, label="city-wide detector", color=COLORS[1])
ax.bar([k + 0.17 for k in x], pv["borough's own detector"], width=0.32, label="borough's own detector", color=COLORS[0])
ax.set_xticks(list(x), boroughs)
ax.set(ylabel="Detection rate", ylim=(0, 1.05), title="Drift injected into one borough only (ADWIN, delta 0.002)")
ax.grid(axis="x", visible=False)
ax.legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.1))
save(fig, "city_vs_borough")
"""),
    md("## 7. Live pipeline run (Kafka and Spark, Feb-Apr 2020)"),
    code("""
p = R / "stream" / "monitor_hourly.csv"
if p.exists() and (R / "stream" / "monitor_alarms.csv").stat().st_size > 2:
    h = pd.read_csv(p, parse_dates=["hour"])
    al = pd.read_csv(R / "stream" / "monitor_alarms.csv", parse_dates=["timestamp"])
    fig, ax = plt.subplots(figsize=(6.8, 2.8))
    daily = h.set_index("hour").demand.resample("D").sum()
    ax.plot(daily.index, daily.values, color=COLORS[0])
    for det, c in [("adwin", COLORS[1]), ("psi", COLORS[2])]:
        for t in al[(al.detector == det) & (al.scope == "city")].timestamp:
            ax.axvline(t, color=c, linewidth=1, linestyle="--")
        ax.plot([], [], color=c, linestyle="--", label=f"{det} alarm")
    ax.set(ylabel="Pickups per day", title="Streamed demand and city-wide drift alarms")
    ax.legend(fontsize=8)
    save(fig, "live_run")
    S["live"] = {"hours": len(h), "alarms": len(al), "city_alarms": al[al.scope == "city"][["timestamp", "detector"]].astype(str).to_dict("records")}
    rp = R / "stream" / "monitor_retrains.csv"
    if rp.exists() and rp.stat().st_size > 2:
        r = pd.read_csv(rp)
        display(r.round(2))
        S["live"]["retrains"] = json.loads(r.round(2).to_json(orient="records"))
(R / "summary.json").write_text(json.dumps(S, indent=1, default=str))
"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
(ROOT / "notebooks" / "results.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote notebooks/results.ipynb")
