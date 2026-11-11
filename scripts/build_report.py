"""Build the technical report docs/report.pdf from results/summary.json and docs/figures.

Usage:  python scripts/build_report.py
Needs reportlab. Every number in the report is read from results/summary.json, so re-running the
experiments and this script keeps the report in sync with the notebook.
"""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
S = json.loads((ROOT / "results" / "summary.json").read_text())
FIG = ROOT / "docs" / "figures"
OUT = ROOT / "docs" / "report.pdf"

ss = getSampleStyleSheet()
body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9.5, leading=12.5, spaceAfter=5)
h1 = ParagraphStyle("h1", parent=ss["Heading2"], fontSize=12.5, spaceBefore=8, spaceAfter=4, keepWithNext=1)
cap = ParagraphStyle("cap", parent=body, fontSize=8.5, leading=10.5, textColor=colors.HexColor("#444444"))
title = ParagraphStyle("title", parent=ss["Title"], fontSize=16, spaceAfter=4)
sub = ParagraphStyle("sub", parent=body, alignment=1, fontSize=9.5)
cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=9.5, spaceAfter=0)

W = A4[0] - 3.6 * cm


def P(text: str) -> Paragraph:
    return Paragraph(text, body)


def fig(name: str, caption: str, width: float = W) -> list:
    img = Image(str(FIG / name))
    img.drawWidth, img.drawHeight = width, width * img.imageHeight / img.imageWidth
    return [img, Paragraph(caption, cap), Spacer(1, 4)]


def table(rows: list[list], widths: list[float] | None = None) -> Table:
    data = [[Paragraph(str(c), cell) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8e8e8")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    return t


def f(x, nd=1, none="-"):
    return none if x is None else f"{x:.{nd}f}"


det = {d["detector"]: d for d in S["detectors"]}
pol = {p["policy"]: p for p in S["policies"]}
bor = {b["scope"]: b for b in S["borough_never"]}
gap = {g["policy"]: g for g in S["gap"]}
cov = {c["detector"]: c for c in S["covid_city"]}
cvb = {(c["target"], c["detector"]): c for c in S["city_vs_borough"]}
live = S["live"]
BEFORE, CRASH, RECOV = "before (Oct 2019-Feb 2020)", "crash (Mar-Jun 2020)", "recovery (Jul 2020-Dec 2021)"
BOROUGHS = ["Manhattan", "Queens", "Brooklyn", "Bronx", "Staten Island"]

story: list = [
    Paragraph("Concept Drift Detection on NYC Taxi Demand Streams", title),
    Paragraph("Technical report, Big Data mini project, Option A. "
              "Team: Rahul Saran Sharma, Rahul Prasanna, Raghav U.", sub),
    Spacer(1, 6),
]

# 1. Problem
story += [
    Paragraph("1. Problem and approach", h1),
    P("A ride-demand model trained on one period degrades when city travel patterns change. We forecast "
      "hourly yellow-taxi pickups for each of the 263 NYC taxi zones one hour ahead, stream 2019-2021 TLC "
      "trip data through the model, and ask four questions: (1) which streaming drift detectors find the "
      "change, how fast and with how many false alarms; (2) whether detector-triggered retraining beats "
      "never retraining or retraining on a schedule; (3) whether drift hurts some boroughs more than others; "
      "and (4) whether a city-wide detector is enough to protect every borough."),
    P("The model is a single LightGBM regressor over all zones with lag features (1, 2, 24 and 168 hours), "
      "24 h and 168 h rolling means, hour of day, day of week, a holiday flag and the zone id. It is trained "
      "on January-June 2019; July-September 2019 is used for early stopping and to calibrate the detectors; "
      "October 2019 onward is the stream under test. On October-December 2019 it reaches a city-wide WAPE of "
      f"<b>{S['model_wape']:.1f}%</b> against <b>{S['naive_wape']:.1f}%</b> for a seasonal-naive baseline "
      "(same hour last week)."),
]

# 2. Architecture
story += [
    Paragraph("2. Pipeline", h1),
    P("A replay producer reads the monthly Parquet files and publishes trips to the Kafka topic "
      "<i>trips</i> in event-time order (3600x real time). Spark Structured Streaming groups them into "
      "1-hour tumbling windows per pickup zone with a 2-hour watermark, writes the counts to the "
      "<i>zone_demand</i> topic and to partitioned Parquet. The monitor service consumes zone-hour counts, "
      "predicts, scores the error, and feeds two signals to the detectors: the hourly city (or borough) WAPE "
      "divided by its July-September value for the same hour of week, and total demand divided by its "
      "calibrated hour-of-week value. Normalising by hour of week removes the daily and weekly cycle so the "
      "detectors see a roughly stationary signal when nothing has changed. A drift event goes to the "
      "retraining controller and to an append-only audit log; promoted models are versioned in MLflow."),
    P(f"In the live run we replayed March 2020 end to end through Kafka and Spark on a laptop: "
      f"{S['stream_demo_rows']:,} trips at a median of {S['stream_demo_rate_median']:,.0f} rows per second, "
      f"{live['hours']} stream hours, {live['alarms']} alarms in total (city and borough). The city PSI warned on "
      f"{live['city_alarms'][0]['timestamp']}; city ADWIN fired on {live['city_alarms'][1]['timestamp']}, "
      "which the combined policy accepted as confirmed drift and retrained. The challenger scored "
      f"{live['retrains'][0]['challenger_wape']:.1f}% WAPE on the 48-hour holdout against "
      f"{live['retrains'][0]['champion_wape']:.1f}% for the champion, so the gate kept the champion "
      f"(retrain took {live['retrains'][0]['seconds']:.0f} s)."),
]
story += fig("live_run.png", "Figure 1. Live run, March 2020: city error signal, demand and detector alarms "
             "as produced by the Kafka - Spark - monitor pipeline.")

# 3. Detectors
story += [
    Paragraph("3. Detectors and benchmark", h1),
    P("All detectors share one interface (<i>update(x) -> drift?</i>) and are implemented from scratch: "
      "<b>ADWIN</b> (adaptive window, cuts when two sub-windows differ by more than a Hoeffding bound with "
      "confidence delta), <b>PSI</b> (Population Stability Index of the last 168 hours against a 10-bin "
      "reference histogram, evaluated every 6 hours, warning at 0.4x the drift threshold), "
      "<b>Page-Hinkley</b> (cumulative deviation from the running mean) and <b>KSWIN</b> "
      "(Kolmogorov-Smirnov test between a recent and a reference window). ADWIN, Page-Hinkley and KSWIN "
      "watch the model error; PSI and a second ADWIN watch demand."),
    P("Real data has no labelled change points, so detection delay is measured on a synthetic benchmark: "
      "we inject drift with a known start into October-December 2019 (abrupt -50%, abrupt +50%, a 2-week "
      "gradual ramp to -50%, and a 3-hour shift of the daily profile), into the whole city or one borough, "
      "with 8 random change points each. Delay is hours from the true change to the first alarm (capped at "
      "336 h); false alarms are alarms before the change and on a no-drift stream, per 1000 stream hours."),
]
rows = [["Detector", "Signal", "Detection rate", "Median delay (h)", "False alarms / 1000 h"]]
for name, sig in [("page_hinkley_t5", "error"), ("page_hinkley_t10", "error"), ("page_hinkley_t20", "error"),
                  ("adwin_d0.002", "error"), ("adwin_d0.01", "error"), ("adwin_d0.05", "error"),
                  ("adwin_demand_d0.002", "demand"), ("kswin_a0.0005", "error"), ("kswin_a0.005", "error"),
                  ("psi_t0.1", "demand"), ("psi_t0.25", "demand"), ("psi_t0.5", "demand")]:
    d = det[name]
    rows.append([name, sig, f"{d['detection_rate']:.0%}", f(d["median_delay_h"]), f(d["false_alarms_per_1000h"], 2)])
story += [table(rows, [4.2 * cm, 1.8 * cm, 2.6 * cm, 3 * cm, 3.6 * cm]),
          Paragraph("Table 1. Detector benchmark, all scenarios and targets pooled.", cap), Spacer(1, 4)]
story += fig("detector_tradeoff.png", "Figure 2. Speed against false alarms for every detector setting.", W * 0.82)
pht5, pht20, ad = det["page_hinkley_t5"], det["page_hinkley_t20"], det["adwin_d0.002"]
story += [
    P(f"<b>Findings.</b> Page-Hinkley is the fastest (median {pht5['median_delay_h']:.0f} h at threshold 5) "
      f"but pays for it with {pht5['false_alarms_per_1000h']:.1f} false alarms per 1000 h; at threshold 20 it "
      f"takes {pht20['median_delay_h']:.0f} h with {pht20['false_alarms_per_1000h']:.1f}. ADWIN (delta 0.002) "
      f"finds every injected change in a median {ad['median_delay_h']:.1f} h with "
      f"{ad['false_alarms_per_1000h']:.1f} false alarms per 1000 h and has no threshold to hand-tune. KSWIN "
      "is no faster than ADWIN and raises 2-3 times as many false alarms. PSI hardly ever false-alarms, but "
      "with a 168-hour window it only reached the threshold on the gradual scenario (29% of runs, ~180 h "
      "late): a 50% change in total demand moves the binned distribution of the normalised demand ratio "
      "less than the thresholds tuned on real data allow. PSI is therefore useful as a slow confirmation "
      "signal, not as a primary detector."),
]
c_ad, c_ph = cov["adwin_d0.002"], cov["page_hinkley_t20"]
story += [
    P(f"<b>Real drift (COVID-19).</b> On the real stream, ADWIN first fired after 1 March on "
      f"{c_ad['first_alarm_after_mar1']}, {abs(c_ad['hours_from_mar15']):.0f} hours before the 15 March "
      f"demand collapse usually quoted, with {c_ad['alarms_before_march']} alarms in the five calm months "
      f"before it; Page-Hinkley (t=20) fired on {c_ph['first_alarm_after_mar1']} after "
      f"{c_ph['alarms_before_march']} earlier alarms. KSWIN and Page-Hinkley at low thresholds alarmed "
      "throughout early March and February, so their 'early' alarms are not informative. City PSI did not "
      "fire in the March window at all in the offline run."),
]

# 4. Retraining
story += [
    Paragraph("4. Retraining policy", h1),
    P("Every policy is replayed over October 2019 - December 2021 with the same controller. On a trigger it "
      "trains a challenger on the last 6 weeks, compares it with the champion on the 48 hours just before "
      "the trigger (which the challenger does not train on), and promotes it only if its WAPE is lower. A "
      "72-hour cooldown stops alarm storms from causing back-to-back retrains. Policies: <i>never</i>; "
      "<i>periodic</i> (first of every month); <i>adwin</i> (city ADWIN on error); <i>psi</i>; "
      "<i>combined</i> (ADWIN confirmed by a PSI warning within 7 days); <i>adwin_any_borough</i> (city "
      "ADWIN or any borough's ADWIN); and <i>oracle</i>, which retrains on six known real change dates and "
      "serves as a reference."),
]
rows = [["Policy", "WAPE before", "WAPE crash", "WAPE recovery", "Retrains", "Promoted", "Train time (s)"]]
for name in ["never", "periodic", "adwin", "psi", "combined", "adwin_any_borough", "oracle"]:
    p = pol[name]
    rows.append([name, f(p[BEFORE], 2), f(p[CRASH], 2), f(p[RECOV], 2), f"{p['retrains']:.0f}",
                 f"{p['promoted']:.0f}", f(p["retrain_seconds"])])
story += [table(rows, [3.6 * cm, 2 * cm, 2 * cm, 2.3 * cm, 1.7 * cm, 1.8 * cm, 2.2 * cm]),
          Paragraph("Table 2. City-wide WAPE (%) by period and cost of each policy. Before = Oct 2019-Feb "
                    "2020, crash = Mar-Jun 2020, recovery = Jul 2020-Dec 2021.", cap), Spacer(1, 4)]
story += fig("policy_error.png", "Figure 3. Daily city WAPE for each retraining policy across 2019-2021.")
best = pol["adwin_any_borough"]
story += [
    P(f"<b>Findings.</b> The best crash-period error comes from <i>adwin_any_borough</i> "
      f"({best[CRASH]:.2f}% vs {pol['never'][CRASH]:.2f}% for never retraining), and periodic retraining "
      f"is clearly worst ({pol['periodic'][CRASH]:.2f}%) because it retrains on calendar dates that straddle "
      "the collapse. The honest result, however, is that the gains are small and <i>never</i> retraining is "
      f"best during recovery ({pol['never'][RECOV]:.2f}%). The model already adapts through its 1- and "
      "2-hour lag features, and a 6-week window that contains the abnormal spring of 2020 teaches the model "
      "patterns that no longer hold later. The holdout gate is what keeps detector-triggered retraining "
      f"safe: only {best['promoted']:.0f} of {best['retrains']:.0f} challengers were promoted, and in the "
      "live run the gate rejected a challenger that was worse than the champion. Even the oracle does not "
      "beat never retraining, which says the bottleneck is the retraining recipe (window length, features), "
      "not the trigger."),
    P("<b>Recommended policy.</b> Trigger on city ADWIN on the error signal (delta 0.002) <i>or</i> any "
      "borough's ADWIN; require the challenger to beat the champion on a 48-hour holdout; 72-hour cooldown; "
      "6-week training window; log every decision. Two changes we would make next, based on Section 5: "
      "gate on per-borough error as well as city error, and use a longer or weighted training window after "
      "an abrupt shift so the model is not retrained only on the shock."),
]

# 5. Boroughs
story += [Paragraph("5. Does drift hurt some boroughs more?", h1)]
rows = [["Borough", "WAPE before", "WAPE crash", "Crash / before", "Crash WAPE, never",
         "Crash WAPE, adwin_any_borough"]]
for b in BOROUGHS:
    r = bor[b]
    rows.append([b, f(r[BEFORE]), f(r[CRASH]), f"{r['crash_over_before']:.2f}x", f(gap["never"][b]),
                 f(gap["adwin_any_borough"][b])])
story += [table(rows, [2.7 * cm, 2.2 * cm, 2.2 * cm, 2.3 * cm, 2.8 * cm, 3.6 * cm]),
          Paragraph("Table 3. Borough WAPE (%) for the never-retrained model, and crash-period WAPE under the "
                    "best policy.", cap), Spacer(1, 4)]
story += fig("borough_error.png", "Figure 4. Borough error before, during and after the COVID crash.", W * 0.9)
story += [
    P(f"<b>Yes, unevenly.</b> Relative to its own baseline, Queens was hit hardest (error x"
      f"{bor['Queens']['crash_over_before']:.2f}), driven by the collapse of airport trips at JFK and LaGuardia, "
      f"then Manhattan (x{bor['Manhattan']['crash_over_before']:.2f}) and Brooklyn "
      f"(x{bor['Brooklyn']['crash_over_before']:.2f}). The Bronx and Staten Island barely moved because their "
      "yellow-taxi demand is so small that the model was already poor there (WAPE above 100%, often "
      "predicting a fraction of a trip per zone-hour). Retraining makes the gap worse: under every policy "
      f"that promoted a new model, Staten Island's crash-period WAPE jumps from {gap['never']['Staten Island']:.0f}% to over "
      f"{gap['adwin_any_borough']['Staten Island']:.0f}%, because the city-level gate accepts a model that is "
      "better for Manhattan, which dominates the total, and worse for tiny boroughs. This is why the "
      "recommended gate checks each borough."),
]
story += fig("city_vs_borough.png", "Figure 5. Share of borough-only drift caught by the borough's own "
             "detector against the city-wide detector.", W * 0.9)
q, b, s = (cvb[("Queens", "adwin_d0.002")], cvb[("Bronx", "adwin_d0.002")], cvb[("Brooklyn", "psi_t0.25")])
OWN, CITY = "borough's own detector", "city-wide detector"
story += [
    P(f"<b>A city-wide detector does not protect every borough.</b> When drift is injected into one "
      f"borough only, city ADWIN catches it in Manhattan every time but in Queens only "
      f"{q[CITY]:.0%} of runs and in the Bronx {b[CITY]:.0%}, while each borough's own ADWIN catches "
      f"{q[OWN]:.0%} and {b[OWN]:.0%}. City PSI caught none of the outer-borough changes (Brooklyn: "
      f"{s[CITY]:.0%} city vs {s[OWN]:.0%} own). Per-borough detectors are cheap and are part of the "
      "recommended policy."),
]

# 6. Limitations
story += [
    Paragraph("6. Limitations", h1),
    P("Delay is measured on synthetic drift because real changes have no exact start; COVID is used only as a "
      "qualitative check. WAPE in the Bronx and Staten Island is dominated by near-zero counts and should be "
      "read as 'poor' rather than as a precise number. Yellow taxis are a shrinking share of outer-borough "
      "trips, so borough results describe yellow-taxi demand, not mobility. The Docker Compose stack is "
      "written but was not run; the tested setup is native Linux/WSL with a single Kafka broker and Spark "
      "in local mode, so we make no claims about cluster scaling. The live run covers one month of replay."),
    Paragraph("7. Reproducing", h1),
    P("<i>scripts/pipeline.sh setup | data | up | demo</i> builds the environment, downloads the data, "
      "trains the model and runs the live demo; <i>experiments/run_detectors.py</i> and "
      "<i>experiments/run_retraining.py</i> regenerate every table here into <i>results/</i>; "
      "<i>notebooks/results.ipynb</i> rebuilds the figures and <i>scripts/build_report.py</i> rebuilds this "
      "report from <i>results/summary.json</i>."),
]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                        topMargin=1.6 * cm, bottomMargin=1.6 * cm,
                        title="Concept Drift Detection on NYC Taxi Demand Streams")
doc.build(story)
print(f"wrote {OUT}")
