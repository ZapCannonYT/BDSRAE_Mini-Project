# Model Card: NYC Zone-Hour Demand Forecaster

Owner: Raghav U. Numbers come from `results/summary.json` and match `notebooks/results.ipynb` and `docs/report.pdf`.

## Model details
- Type: LightGBM regressor, one model for all zones. Baseline: seasonal naive (same hour last week).
- Input features (`model/features.py`): pickups 1, 2, 24 and 168 hours earlier; 24 h and 168 h rolling means; hour of day; day of week; NYC holiday flag; zone id (categorical).
- A day-ahead variant (`configs/day_ahead.yaml`, lags 24/48/168 h) is also trained; city WAPE 17.0% on Oct-Dec 2019.
- Output: predicted pickups in the next hour for one taxi zone.
- Versions: every live model is registered in MLflow; see the audit log for when and why each was promoted.

## Intended use
- Course project demonstrating drift detection and retraining. Not for operational dispatch or pricing.

## Training and evaluation data
- Train: Jan-Jun 2019. Early stopping and detector calibration: Jul-Sep 2019. Validation: Oct-Dec 2019. Drift evaluation: Oct 2019 - Dec 2021. See the data card.

## Metrics
City-wide WAPE is the headline metric. sMAPE is reported in `results/` but is dominated by zone-hours with zero or one pickup (it is above 100% for every model), so it is not used to compare models.

| Split | MAE (pickups) | WAPE | Notes |
|---|---|---|---|
| Validation Oct-Dec 2019, LightGBM | 4.71 | 13.1% | |
| Validation Oct-Dec 2019, seasonal naive | 7.89 | 22.0% | same hour last week |
| Crash Mar-Jun 2020, never retrained | 1.54 | 28.9% | |
| Crash Mar-Jun 2020, best policy (adwin_any_borough) | - | 28.6% | |
| Recovery Jul 2020-Dec 2021, never retrained | 2.02 | 18.4% | best of all policies |
| Recovery Jul 2020-Dec 2021, adwin_any_borough | - | 18.9% | |

## Performance by borough
WAPE (%) of the never-retrained model, and crash-period WAPE under the best policy:

| Borough | Before (Oct 2019-Feb 2020) | Crash (Mar-Jun 2020) | Crash / before | Crash, adwin_any_borough |
|---|---|---|---|---|
| Manhattan | 11.8 | 24.6 | 2.09x | 22.6 |
| Queens | 21.8 | 52.5 | 2.41x | 59.9 |
| Brooklyn | 56.8 | 100.5 | 1.77x | 124.0 |
| Bronx | 114.3 | 131.5 | 1.15x | 169.4 |
| Staten Island | 211.5 | 215.7 | 1.02x | 1154.8 |

Queens degrades most (airport trips collapsed). Retraining with a city-level gate helps Manhattan and hurts the small boroughs; see `results/retraining/summary.csv` for every policy.

## Drift monitoring and retraining
- Detectors: ADWIN (delta 0.002) on the hour-of-week-normalised error, city-wide and per borough; PSI (168 h window, 10 bins, warn 0.1, drift 0.25) on normalised demand as a confirmation signal.
- Trigger: city ADWIN or any borough's ADWIN (`adwin_any_borough` policy).
- Training window: last 6 weeks. Gate: challenger must beat the champion on the 48 hours before the trigger, otherwise the champion stays (this is the rollback). Cooldown: 72 hours.
- Every alarm, retrain and promotion decision is written to the audit log and every model is versioned in MLflow.

## Limitations and risks
- Zone-level counts hide within-zone differences.
- Quiet zones (Staten Island, EWR) have small counts, so percentage errors are noisy.
- Retraining gives only small gains; never retraining is best during the 2020-2021 recovery, because the 6-week window can contain only abnormal data.
- A city-level promotion gate can accept a model that is much worse for small boroughs (Staten Island). A per-borough gate is the recommended next step.
- City-wide detectors miss most drift confined to one outer borough; per-borough detectors are needed.

## Ethical considerations
- No personal data is used. Errors that are larger in some boroughs could lead to worse service there if a model like this were used for supply planning; this is measured and reported above.
