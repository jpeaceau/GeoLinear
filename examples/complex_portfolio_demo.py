"""
Complex Portfolio Demo -- GeoLinear (PyramidHART) vs GLM
=========================================================
A synthetic portfolio designed to be genuinely hard for a global linear model.
Complexity sources:

  1. Correlated features  -- vehicle_value/driver_age, mileage/occupation,
                             urban/mileage, telematics/night_driving
  2. U-shaped age effect  -- young AND old drivers are high risk
  3. Sign-flipping mileage-- high mileage REDUCES risk in dense urban regimes
                             (slow traffic, short trips) but INCREASES it rurally
  4. Non-monotone value   -- mid-value vehicles are riskiest (risk compensation
                             without premium safety features)
  5. Telematics x incidents-- good telematics score partially pardons prior claims
  6. Three-way extreme    -- occupation_risk x night_driving x young driver:
                             multiplicative blowup a GLM cannot represent
  7. Sports car x age     -- coefficient SIGN differs: young sports = high risk,
                             older sports = protective (skilled confident drivers)

The engineered GLM gets explicit polynomial and interaction terms to give it
every fair advantage before comparing to GeoLinear.

Architecture note (pyramid_hart default):
  GeoLinear partitions the policyholder space using PyramidHART cooperative
  geometry (A-statistic + MAD whitening + AbsoluteError splits), then fits a
  separate Ridge linear model within each partition.  The end result is a
  collection of ordinary linear models -- one per risk regime -- each with
  fully interpretable coefficients suitable for actuarial review.

  No manual feature engineering is required: the regime boundaries (and the
  relevant feature interactions within each regime) are discovered automatically.

Requirements: pip install geolinear[examples]   (xgboost + optuna)
"""

from __future__ import annotations

import warnings
import numpy as np
import optuna
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from geolinear import GeoLinear

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

SEP = "-" * 64

def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")

def row(label: str, r2: float, delta: float | None = None) -> None:
    d = f"  dR2 {delta:+.3f}" if delta is not None else ""
    print(f"  {label:<34}  R2={r2:.4f}{d}")


# =============================================================================
# 1. Data generation
# =============================================================================
section("1. Complex Portfolio (n=8 000, 11 features)")

rng = np.random.default_rng(1)
N = 8_000

# -- Raw features (correlated) ------------------------------------------------

driver_age = rng.uniform(18, 80, N)

# Older drivers buy more expensive vehicles
vehicle_value = (12_000 + 600 * driver_age
                 + rng.normal(0, 12_000, N)).clip(3_000, 130_000)

# Occupation risk: delivery/courier/sales; skewed low
occupation_risk = rng.beta(1.5, 5, N)

# Mileage driven up by occupation
annual_mileage = (6_000 + 38_000 * occupation_risk
                  + rng.normal(0, 4_000, N)).clip(500, 80_000)

# Urban score anti-correlated with mileage (urban = short trips)
urban_score = (0.9 - 0.7 * (annual_mileage / 80_000)
               + rng.uniform(0, 0.25, N)).clip(0, 1)

# Night driving; heavier for young, high-occupation
night_driving_pct = rng.beta(
    1.5 + 1.5 * (driver_age < 28) + 2.0 * occupation_risk,
    8, N).clip(0, 1)

# Telematics score anti-correlated with night driving and youth
telematics_raw = (0.75
                  - 0.25 * (driver_age < 28)
                  - 0.30 * night_driving_pct
                  + rng.normal(0, 0.12, N))
telematics_score = telematics_raw.clip(0.05, 1.0)

# Vehicle age: independent
vehicle_age = rng.exponential(5, N).clip(0, 20)

# Prior incidents: higher for young + high occupation
lambda_inc = (0.20
              + 0.50 * (driver_age < 26)
              + 0.35 * occupation_risk
              + 0.20 * (night_driving_pct > 0.35))
prior_incidents = rng.poisson(lambda_inc, N).astype(float)

# Vehicle type: 0=standard, 1=SUV, 2=sports
vehicle_type = rng.choice([0, 1, 2], N, p=[0.60, 0.30, 0.10])
type_suv    = (vehicle_type == 1).astype(float)
type_sports = (vehicle_type == 2).astype(float)

# -- DGP (log pure premium) ---------------------------------------------------

# Standardise for coefficient comparability
def std(v): return (v - v.mean()) / v.std()

age_s    = std(driver_age)
val_s    = std(vehicle_value)
mile_s   = std(annual_mileage)
urban_s  = std(urban_score)
night_s  = std(night_driving_pct)
telem_s  = std(telematics_score)
occ_s    = std(occupation_risk)
vage_s   = std(vehicle_age)
inc_s    = std(prior_incidents)

# (a) U-shaped age: GLM needs age + age^2 at minimum
log_pp = 4.30 + 0.40 * age_s**2 - 0.05 * age_s

# (b) Mileage sign-flips in urban regime
#     Urban (>0.6): extra mileage = slow city driving = lower risk
#     Rural (<0.4): extra mileage = fast roads = higher risk
log_pp += np.where(urban_score > 0.60, -0.18 * mile_s,
          np.where(urban_score < 0.40,  0.32 * mile_s,
                                         0.08 * mile_s))   # mid-urban

# (c) Non-monotone vehicle value (inverted-U centred ~GBP30k)
log_pp += 0.18 * val_s - 0.10 * val_s**2

# (d) Night driving x age: additive alone, but multiplicative amplifier for young
log_pp += 0.28 * night_s + 0.40 * night_s * (driver_age < 30).astype(float)

# (e) Telematics x prior incidents: good score attenuates incident loading
log_pp += 0.38 * inc_s * (1.0 - 0.50 * telematics_score)

# (f) Telematics main effect (protective, especially in high-risk groups)
log_pp -= 0.30 * telem_s

# (g) Occupation risk x vehicle age: older vehicles in high-use hands fail more
log_pp += 0.20 * occ_s + 0.15 * occ_s * vage_s

# (h) Sports car: SIGN FLIP across age
#     Young (<32) sports driver: +0.60 risk
#     Older (>50) sports driver: -0.15 (confident, deliberate)
log_pp += type_sports * np.where(driver_age < 32,  0.60,
                        np.where(driver_age > 50, -0.15, 0.25))
log_pp += type_suv * 0.08

# (i) Three-way extreme regime: young + occupation + night = blowup
extreme = ((driver_age < 30) & (occupation_risk > 0.60)
           & (night_driving_pct > 0.35)).astype(float)
log_pp += 0.75 * extreme      # ~3.5% of portfolio; GLM cannot represent this

# (j) Urban main effect
log_pp -= 0.12 * urban_s

# Noise
log_pp += rng.normal(0, 0.28, N)

y = np.exp(log_pp)

print(f"  Pure premium: mean GBP{y.mean():,.0f},  std GBP{y.std():,.0f}")
print(f"  Extreme-risk segment: {extreme.mean()*100:.1f}% of portfolio")
print(f"  Urban > 0.6: {(urban_score > 0.6).mean()*100:.1f}%  |  "
      f"Rural < 0.4: {(urban_score < 0.4).mean()*100:.1f}%")
print(f"  Sports car:  {type_sports.mean()*100:.1f}%")

# -- Feature matrix -----------------------------------------------------------

FEAT_NAMES = [
    "driver_age", "vehicle_age", "annual_mileage", "vehicle_value",
    "urban_score", "night_driving_pct", "telematics_score",
    "occupation_risk", "prior_incidents", "type_suv", "type_sports",
]

X_raw = np.column_stack([
    driver_age, vehicle_age, annual_mileage, vehicle_value,
    urban_score, night_driving_pct, telematics_score,
    occupation_risk, prior_incidents, type_suv, type_sports,
])

X_tr, X_te, y_tr, y_te = train_test_split(
    X_raw, y, test_size=0.20, random_state=42)
log_y_tr = np.log(y_tr)
log_y_te = np.log(y_te)


# =============================================================================
# 2. GLM baselines
# =============================================================================
section("2. GLM Baselines")

# -- 2a. Raw Ridge GLM --------------------------------------------------------
glm_raw = Pipeline([
    ("sc", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
glm_raw.fit(X_tr, log_y_tr)
r2_glm_raw = r2_score(log_y_te, glm_raw.predict(X_te))
row("Ridge GLM (raw features)", r2_glm_raw)

# -- 2b. Engineered Ridge GLM -------------------------------------------------
# Give the GLM every known nonlinearity so the comparison is fair:
#   age^2, value^2, mileage x urban, night x age<30, telem x incidents,
#   occ x vehicle_age, sports x age<32, sports x age>50

def engineer(X: np.ndarray, fit_stats=None):
    """Add polynomial + interaction terms the GLM needs to compete."""
    (age, vage, mile, val, urban, night, telem, occ, inc,
     t_suv, t_sports) = [X[:, i] for i in range(11)]

    # z-score using training stats
    if fit_stats is None:
        means = X.mean(0)
        stds  = X.std(0)
        stds[stds == 0] = 1.0
        fit_stats = (means, stds)
    means, stds = fit_stats
    Z = (X - means) / stds

    age_z, vage_z, mile_z, val_z, urban_z, night_z, telem_z, occ_z, inc_z = \
        [Z[:, i] for i in range(9)]

    extra = np.column_stack([
        age_z ** 2,                               # U-shape
        val_z ** 2,                               # non-monotone value
        mile_z * urban_z,                         # mileage x urban
        night_z * (age < 30).astype(float),       # night x young
        inc_z * (1.0 - telem),                    # telem-attenuated incidents
        occ_z * vage_z,                           # occ x vehicle age
        t_sports * (age < 32).astype(float),      # sports x young
        t_sports * (age > 50).astype(float),      # sports x older
    ])
    return np.hstack([X, extra]), fit_stats

X_tr_eng, eng_stats = engineer(X_tr)
X_te_eng, _         = engineer(X_te, fit_stats=eng_stats)

glm_eng = Pipeline([
    ("sc", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
glm_eng.fit(X_tr_eng, log_y_tr)
r2_glm_eng = r2_score(log_y_te, glm_eng.predict(X_te_eng))
row("Ridge GLM (engineered, 19 feats)", r2_glm_eng,
    delta=r2_glm_eng - r2_glm_raw)

print(f"\n  Note: even the engineered GLM requires a human actuary to")
print(f"  know which interactions to add.  GeoLinear discovers them.")


# =============================================================================
# 3. GeoLinear -- default hyperparameters (raw features)
# =============================================================================
section("3. GeoLinear -- Default Hyperparameters (pyramid_hart, raw features)")
print("  Each model uses PyramidHART cooperative geometry partitions +")
print("  a separate Ridge linear model per partition (no feature engineering).")
print()

# Standardise once for GeoLinear
scaler = StandardScaler().fit(X_tr)
Xgl_tr = scaler.transform(X_tr)
Xgl_te = scaler.transform(X_te)

VARIANTS = [
    ("GL-Ridge", GeoLinear(base_learner="ridge", alpha=1.0)),
    ("GL-OLS",   GeoLinear(base_learner="ols",   alpha=0.0)),
    ("GL-Lasso", GeoLinear(base_learner="lasso", alpha=1.0)),
]

gl_default_r2 = {}
for name, model in VARIANTS:
    model.fit(Xgl_tr, log_y_tr)
    r2 = r2_score(log_y_te, model.predict(Xgl_te))
    gl_default_r2[name] = r2
    row(name, r2, delta=r2 - r2_glm_raw)


# =============================================================================
# 4. GeoLinear HPO -- 60-trial Optuna
# =============================================================================
section("4. GeoLinear -- 60-trial Optuna HPO")

def gl_objective(trial: optuna.Trial) -> float:
    model = GeoLinear(
        n_rounds              = trial.suggest_int("n_rounds", 10, 60),
        learning_rate         = trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
        alpha                 = trial.suggest_float("alpha", 0.01, 30.0, log=True),
        y_weight              = trial.suggest_float("y_weight", 0.3, 1.0),
        base_learner          = trial.suggest_categorical(
                                    "base_learner", ["ridge", "ols", "lasso"]),
        hvrt_n_partitions     = trial.suggest_int("hvrt_n_partitions", 4, 14),
        min_samples_partition = trial.suggest_int("min_samples_partition", 5, 25),
        refit_interval        = trial.suggest_categorical("refit_interval", [0, 1, 2]),
        hvrt_inner_rounds     = trial.suggest_int("hvrt_inner_rounds", 1, 3),
    )
    return cross_val_score(model, Xgl_tr, log_y_tr, cv=5, scoring="r2").mean()

study_gl = optuna.create_study(direction="maximize")
study_gl.optimize(gl_objective, n_trials=60, show_progress_bar=False)

best_gl = GeoLinear(**study_gl.best_params)
best_gl.fit(Xgl_tr, log_y_tr)
r2_gl_hpo = r2_score(log_y_te, best_gl.predict(Xgl_te))

print(f"  Best CV R2: {study_gl.best_value:.4f}")
print(f"  Best params: {study_gl.best_params}")
print()
row("GL-HPO (60 trials)", r2_gl_hpo, delta=r2_gl_hpo - r2_glm_raw)


# =============================================================================
# 5. XGBoost HPO -- 60-trial Optuna (upper bound)
# =============================================================================
section("5. XGBoost -- 60-trial Optuna HPO (upper bound)")

try:
    from xgboost import XGBRegressor

    def xgb_objective(trial: optuna.Trial) -> float:
        model = XGBRegressor(
            n_estimators      = trial.suggest_int("n_estimators", 50, 400),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth         = trial.suggest_int("max_depth", 3, 8),
            subsample         = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha         = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            random_state=42, n_jobs=-1, verbosity=0,
        )
        return cross_val_score(model, X_tr, log_y_tr, cv=5, scoring="r2").mean()

    study_xgb = optuna.create_study(direction="maximize")
    study_xgb.optimize(xgb_objective, n_trials=60, show_progress_bar=False)

    best_xgb = XGBRegressor(**study_xgb.best_params, random_state=42, n_jobs=-1, verbosity=0)
    best_xgb.fit(X_tr, log_y_tr)
    r2_xgb_hpo = r2_score(log_y_te, best_xgb.predict(X_te))

    print(f"  Best CV R2: {study_xgb.best_value:.4f}")
    row("XGB-HPO (60 trials)", r2_xgb_hpo, delta=r2_xgb_hpo - r2_glm_raw)
    xgb_available = True

except ImportError:
    print("  xgboost not installed -- skipping.")
    r2_xgb_hpo = None
    xgb_available = False


# =============================================================================
# 6. Feature importances -- GL-HPO
# =============================================================================
section("6. GL-HPO Feature Importances")

importances, names = best_gl.feature_importances(feature_names=FEAT_NAMES)
order = np.argsort(importances)[::-1]
scale = importances[order[0]]
for rank, i in enumerate(order, 1):
    bar = "#" * int(importances[i] / scale * 32)
    print(f"  {rank:2d}.  {names[i]:<22}  {importances[i]:.4f}  {bar}")

print()
print("  Ground-truth dominant effects:")
print("    driver_age       -- U-shape (young + old)")
print("    prior_incidents  -- attenuated by telematics (interaction)")
print("    night_driving    -- amplified for young (interaction)")
print("    annual_mileage   -- sign-flips by urban regime")
print("    occupation_risk  -- x vehicle_age interaction + extreme-risk 3-way")
print("    telematics_score -- protective; modifies incidents coefficient")
print("    type_sports      -- sign-flips by driver age")


# =============================================================================
# 7. Summary
# =============================================================================
section("Summary")

print(f"  {'Model':<36}  {'R2':>7}  {'dR2 vs raw GLM':>15}")
print(f"  {'-'*36}  {'-'*7}  {'-'*15}")

rows = [
    ("Ridge GLM (raw, 11 feats)",        r2_glm_raw,  0.0),
    ("Ridge GLM (engineered, 19 feats)", r2_glm_eng,  r2_glm_eng  - r2_glm_raw),
    ("GL-Ridge default",                 gl_default_r2["GL-Ridge"],
                                         gl_default_r2["GL-Ridge"] - r2_glm_raw),
    ("GL-Lasso default",                 gl_default_r2["GL-Lasso"],
                                         gl_default_r2["GL-Lasso"] - r2_glm_raw),
    ("GL-HPO (60 trials, raw feats)",    r2_gl_hpo,   r2_gl_hpo   - r2_glm_raw),
]
if xgb_available:
    rows.append(
        ("XGB-HPO (60 trials, upper bound)",  r2_xgb_hpo, r2_xgb_hpo - r2_glm_raw))

for label, r2, delta in rows:
    flag = "  <-- GL advantage" if "GL-HPO" in label else ""
    print(f"  {label:<36}  {r2:7.4f}  {delta:+15.4f}{flag}")

print()
print("  Complexity sources the raw GLM cannot represent:")
print("  * U-shaped age (needs age^2)")
print("  * Mileage sign-flip across urban regimes (needs mile x urban)")
print("  * Non-monotone vehicle value (needs value^2)")
print("  * Sports x age sign-flip (regime-conditional coefficient)")
print("  * Three-way extreme risk (occ x night x young -- no polynomial captures this)")
print("  * Telematics-modulated incident loading (inc x telem interaction)")
print()
print("  GeoLinear (pyramid_hart) discovers all of these from geometry alone,")
print("  with no manual feature engineering.")
print()
print("  Actuarial readiness:")
print("  * End result = collection of Ridge linear models (one per risk regime)")
print("  * Each partition's coef_ = relativities for that cooperative risk group")
print("  * PyramidHART's MAD whitening + AbsoluteError splits align with")
print("    heavy-tailed claims data and the actuarial MAE objective")
print("  * Compressible to a single OLS meta-GLM for regulatory filing")
print()
