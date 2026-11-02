# Model Card: NYC Zone-Hour Demand Forecaster

Owner: Raghav U. Fill in each section as results come in; keep numbers in sync with the notebook.

## Model details
- Type: LightGBM regressor, one model for all zones. Baseline: seasonal naive (same hour last week).
- Input features: TODO (list from `model/features.py`).
- Output: predicted pickups in the next hour for one taxi zone.
- Versions: every live model is registered in MLflow; see the audit log for when and why each was promoted.

## Intended use
- Course project demonstrating drift detection and retraining. Not for operational dispatch or pricing.

## Training and evaluation data
- Train: Jan-Sep 2019. Validation: Oct-Dec 2019. Drift evaluation: 2020-2021. See the data card.

## Metrics
| Split | MAE | sMAPE | Notes |
|---|---|---|---|
| Validation 2019 | TODO | TODO | |
| 2020-2021, never retrained | TODO | TODO | |
| 2020-2021, best policy | TODO | TODO | |

## Performance by borough
TODO: table of sMAPE per borough before and after drift, for each retraining policy.

## Drift monitoring and retraining
TODO: detectors in use, thresholds, retraining policy (trigger, window, gate, cooldown, rollback).

## Limitations and risks
- Zone-level counts hide within-zone differences.
- Quiet zones (Staten Island, EWR) have small counts, so percentage errors are noisy.
- TODO: anything found during experiments.

## Ethical considerations
- No personal data is used. Errors that are larger in some boroughs could lead to worse service there if a model like this were used for supply planning; this is measured and reported above.
