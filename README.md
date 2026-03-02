# GeoLinear

**Boosted piecewise-linear models on cooperative geometry partitions.**

[![PyPI version](https://img.shields.io/pypi/v/geolinear)](https://pypi.org/project/geolinear/)
[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## What is GeoLinear?

GeoLinear discovers *cooperative geometry regimes* in your data — groups of observations
where features interact in similar ways — and fits an interpretable linear model inside
each regime. Predictions are accumulated across boosting rounds.

The key insight is that many real-world relationships are *piecewise-linear in cooperative
geometry*. A global linear model averages over regimes and loses the regime-specific signal.
GeoLinear finds the regimes automatically (via [HVRT](https://github.com/jake-peace/hvrt))
and lets the coefficients vary across them.

```
Round 1: HVRT partitions X → {cooperative, competitive, mixed} regimes
         Ridge fits within each partition on residuals
Round 2: New partitioning on updated residuals → new local Ridge models
...
Final prediction = Σ (learning_rate × stage_predictions) + intercept
```

Each partition's Ridge coefficients are *directly interpretable* as local relativities —
exactly the quantities actuaries file with regulators.

---

## Why it matters for actuaries

Insurance pricing models must be both *accurate* and *interpretable*. Regulators require
filed relativities; black-box models are inadmissible. The usual compromise — vanilla GLM —
leaves accuracy on the table when the true relationship is regime-switching.

GeoLinear bridges this gap:

1. **Fit GeoLinear** on the training portfolio. Each partition's Ridge coefficients are
   the relativities for that cooperative geometry segment.
2. **File a meta-GLM**: fit `OLS(X → ŷ_GL)` to compress the ensemble into a single
   interpretable GLM. The compression loss is typically small (R² drop < 2%).
3. **Audit trail**: every prediction can be traced to a specific partition and its
   local coefficient vector.

See `examples/insurance_pricing_demo.py` for a worked actuarial example including
relativities tables and the GLM bridge.

---

## Installation

```bash
pip install geolinear
```

Requires a C++17 compiler and CMake (automatically satisfied on most systems; the
`cmake` PyPI package is a reliable fallback).

For the examples:

```bash
pip install geolinear[examples]   # adds xgboost, optuna, matplotlib
```

---

## Quick start

### Regression

```python
from geolinear import GeoLinear
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

X, y = fetch_california_housing(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

model = GeoLinear(n_rounds=20, learning_rate=0.1, alpha=1.0)
model.fit(X_tr, y_tr)
print(r2_score(y_te, model.predict(X_te)))   # ~0.72 default params
```

### Classification

```python
from geolinear import GeoLinearClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0)

clf = GeoLinearClassifier(n_rounds=20, alpha=1.0)
clf.fit(X_tr, y_tr)
print(roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1]))  # ~0.993
```

### Pipeline + GridSearchCV

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from geolinear import GeoLinear

pipe = Pipeline([("scaler", StandardScaler()), ("gl", GeoLinear())])
grid = GridSearchCV(pipe, {"gl__alpha": [0.1, 1.0, 10.0], "gl__n_rounds": [10, 20]}, cv=5)
grid.fit(X_tr, y_tr)
```

### Optuna HPO

```python
import optuna
from geolinear import GeoLinear
from sklearn.model_selection import cross_val_score

def objective(trial):
    model = GeoLinear(
        n_rounds=trial.suggest_int("n_rounds", 10, 60),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        alpha=trial.suggest_float("alpha", 0.01, 20.0, log=True),
        y_weight=trial.suggest_float("y_weight", 0.3, 1.0),
        base_learner=trial.suggest_categorical("base_learner", ["ridge", "ols", "lasso"]),
        hvrt_n_partitions=trial.suggest_int("hvrt_n_partitions", 4, 12),
        min_samples_partition=trial.suggest_int("min_samples_partition", 5, 20),
        refit_interval=trial.suggest_categorical("refit_interval", [0, 1, 2]),
    )
    return cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=60)
```

---

## Ecosystem: which tool to use?

GeoLinear is part of a family of cooperative-geometry libraries. The right choice
depends on what "interpretable" means in your domain and the structure of your data.

| Domain | Primary need | Recommended |
|--------|-------------|-------------|
| **Insurance pricing** | Filed relativities; regulator-auditable linear tariff | **GeoLinear** |
| **Credit risk / scoring** | Scorecard linearity; SR 11-7 model risk compliance | **GeoLinear** |
| **Utility rate-setting** | Filed tariff schedules with linear rate factors | **GeoLinear** |
| **Healthcare / clinical** | XGBoost-level performance with auditable predictions | GeoXGB |
| **Ecology / environmental** | Nonlinear regime detection; SHAP-compatible explanations | GeoXGB |
| **Public policy** | Algorithmic accountability without linearity constraints | GeoXGB |
| **Personalised interventions** | Per-entity longitudinal data; individual treatment trajectories | AutoITE |

**Rule of thumb:** if your regulator or ethics board requires a *linear equation you can
file or defend*, use GeoLinear. If you need XGBoost-class accuracy with an interpretable
audit trail but no linearity constraint, use GeoXGB. If you have repeated observations
per individual and want to model how treatment effects evolve over time for each entity,
use AutoITE.

---

## Benchmark results

60-trial Optuna HPO, GeoLinear vs XGBoost.

### Regression R²

| DGP | GL-HPO | XGB-HPO | Gap |
|-----|--------|---------|-----|
| T-regime (regime-switching) | 0.331 | 0.336 | −0.005 (tie) |
| 3-regime | 0.387 | 0.464 | −0.077 |
| Friedman1 | 0.948 | 0.963 | −0.015 |
| **Linear** | **0.971** | 0.954 | **+0.017 (GL wins)** |
| CalHousing | 0.689 | 0.818 | −0.129 |
| CalHousing + log transform | 0.725 | 0.818 | −0.093 |

GL is competitive with XGBoost on regime-switching and linear DGPs and wins on
purely linear relationships. The gap on CalHousing narrows with feature engineering.

### Classification AUC

| Dataset | GL-Ridge (default) | GL-HPO | XGB-HPO |
|---------|--------------------|--------|---------|
| Synthetic | 0.940 | 0.957 | 0.981 |
| **BreastCancer** | **0.993** | 0.972 | 0.993 |

Default-parameter GL-Ridge matches XGBoost-HPO on BreastCancer (AUC 0.993 each).

---

## API reference

### `GeoLinear` (regressor)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_rounds` | int | 20 | Boosting rounds |
| `learning_rate` | float | 0.1 | Shrinkage per round |
| `y_weight` | float | 0.5 | HVRT outcome-blend (0 = geometry-only, 1 = y-driven) |
| `base_learner` | str | `"ridge"` | `"ridge"`, `"ols"`, or `"lasso"` |
| `alpha` | float | 1.0 | L2 regularisation within each partition |
| `min_samples_partition` | int | 5 | Minimum samples to fit a partition model |
| `hvrt_n_partitions` | int\|None | None | Target partitions (None = HVRT auto-tune) |
| `hvrt_min_samples_leaf` | int\|None | None | HVRT min leaf size |
| `hvrt_inner_rounds` | int | 1 | HVRT T-residual inner rounds per stage |
| `partition_inner_rounds` | int | 1 | Base-learner rounds within each partition |
| `refit_interval` | int | 0 | 0 = fresh HVRT each round; k>0 = refit every k rounds (faster) |
| `random_state` | int | 42 | Seed (incremented per round for diverse partitionings) |

`GeoLinearClassifier` accepts the same parameters.

### Key methods

```python
model.fit(X, y)                          # fit
model.predict(X)                         # regression predictions
clf.predict_proba(X)                     # class probabilities, shape (n, 2)
model.feature_importances(feature_names) # weighted mean |coef| across partitions/stages
model.stages_                            # list of (None, dict[partition_id, RidgeModel])
```

---

## License

GNU Affero General Public License v3.0 or later. See [LICENSE](LICENSE).
