"""
GeoLinear vs XGBoost — comprehensive predictive benchmark (NMAE metric).

Default hvrt_model is now "pyramid_hart" (A-statistic + MAD whitening +
AbsoluteError splits).  GL-OLS / GL-Ridge / GL-Lasso use this default.
GL-HPO additionally searches over hvrt_model variants.

NMAE_norm = MAE / MAD(y_true)  where MAD = mean(|y - mean(y)|).

This equals NMAE / NMAE_ceiling, since std(y) cancels:
  NMAE          = MAE / std(y)
  NMAE_ceiling  = MAD / std(y)   [= NMAE of predicting the mean]
  NMAE_norm     = NMAE / NMAE_ceiling = MAE / MAD

Interpretation: 1.0 = no better than predicting the mean (noise floor);
                0.0 = perfect predictions.  Lower is better.
Comparable across datasets because the noise floor is always 1.0.

Regression (5-fold CV, NMAE_norm):
  DGPs: T-regime, 3-regime, Friedman #1, Linear, California Housing
  Models:
    Ridge, Ridge+TSQ                         -- linear baselines
    GL-OLS, GL-Ridge, GL-Lasso              -- GeoLinear default (pyramid_hart)
    GL-HPO                                   -- Optuna TPE (searches hvrt_model too)
    XGBoost, XGBoost-HPO

CalHousing feature engineering (GL-HPO vs XGB-HPO):
  Variants: raw / log / raw+TSQ / log+TSQ

Classification (5-fold CV, accuracy + AUC):
  Datasets: make_classification (synthetic), breast_cancer
  Models: GL-Ridge, GL-HPO, XGB, XGB-HPO

HPO budget: N_HPO Optuna TPE trials, 3-fold inner CV, same budget GL vs XGB.

Run:
    python benchmarks/benchmark_vs_xgb_nmae.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing, load_breast_cancer, make_classification
import xgboost as xgb

from geolinear import GeoLinear, GeoLinearClassifier, augment_TSQ

# ── Settings ──────────────────────────────────────────────────────────────────

RNG_SEED = 42
N        = 2000
D        = 10
RHO      = 0.7
N_FOLDS  = 5
N_HPO    = 60
CAL_N    = 5000


# ── DGPs ──────────────────────────────────────────────────────────────────────

def compute_T(X):
    Z = (X - X.mean(0)) / np.where(X.std(0) < 1e-8, 1.0, X.std(0))
    S = Z.sum(1);  Q = (Z**2).sum(1)
    return S**2 - Q


def dgp_t_regime(n, d, rng):
    X = rng.standard_normal((n, d))
    T = compute_T(X)
    y = (2 + 3 * np.sign(T)) * X[:, 0] + rng.standard_normal(n) * 0.5
    return X, y


def dgp_3regime(n, d, rng):
    X = rng.standard_normal((n, d))
    T = compute_T(X)
    t33, t67 = np.percentile(T, 33), np.percentile(T, 67)
    y = np.zeros(n)
    m0 = T < t33;  m1 = (T >= t33) & (T < t67);  m2 = T >= t67
    y[m0] = 3 * X[m0, 0] + rng.standard_normal(m0.sum()) * 0.5
    y[m1] = 3 * X[m1, 1] + rng.standard_normal(m1.sum()) * 0.5
    y[m2] = 3 * X[m2, 2] + rng.standard_normal(m2.sum()) * 0.5
    return X, y


def dgp_friedman1(n, d, rng, rho=RHO):
    from scipy.special import ndtr
    cov = np.array([[rho ** abs(i-j) for j in range(d)] for i in range(d)])
    X   = rng.standard_normal((n, d)) @ np.linalg.cholesky(cov).T
    Xu  = ndtr(X)
    y   = (10 * np.sin(np.pi * Xu[:, 0] * Xu[:, 1])
           + 20 * (Xu[:, 2] - 0.5)**2
           + 10 * Xu[:, 3] + 5 * Xu[:, 4]
           + rng.standard_normal(n))
    return X, y


def dgp_linear(n, d, rng):
    beta = rng.standard_normal(d)
    X    = rng.standard_normal((n, d))
    y    = X @ beta + rng.standard_normal(n) * 0.5
    return X, y


def dgp_california(n, d, rng):
    X_full, y_full = fetch_california_housing(return_X_y=True)
    idx = rng.choice(len(X_full), size=min(n, len(X_full)), replace=False)
    return X_full[idx], y_full[idx]


REGRESSION_DGPS = {
    "T-regime"   : dgp_t_regime,
    "3-regime"   : dgp_3regime,
    "Friedman1"  : dgp_friedman1,
    "Linear"     : dgp_linear,
    "CalHousing" : dgp_california,
}


# ── CalHousing feature engineering ────────────────────────────────────────────
# Features: MedInc(0), HouseAge(1), AveRooms(2), AveBedrms(3),
#           Population(4), AveOccup(5), Latitude(6), Longitude(7)
# Right-skewed columns amenable to log1p: 0, 2, 4, 5

CAL_LOG_COLS = [0, 2, 4, 5]


def cal_log_transform(X):
    X = X.copy()
    for c in CAL_LOG_COLS:
        X[:, c] = np.log1p(np.clip(X[:, c], 0, None))
    return X


def preprocess_cal(X, log=False, tsq=False):
    if log:
        X = cal_log_transform(X)
    if tsq:
        X = augment_TSQ(X)
    return X


# ── Metrics ───────────────────────────────────────────────────────────────────

def nmae(y_true, y_pred):
    """NMAE normalised by its ceiling = MAE / MAD(y_true).

    MAD = mean(|y - mean(y)|) is the NMAE of predicting the mean,
    so this ratio is always 1.0 for a null predictor and 0.0 for a
    perfect one.  Comparable across datasets regardless of scale.
    """
    mad = float(np.mean(np.abs(y_true - y_true.mean())))
    if mad < 1e-8:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)) / mad)


# ── Regression fitters ────────────────────────────────────────────────────────

def fit_ridge(X_tr, y_tr, X_te, augment=False):
    if augment:
        X_tr, X_te = augment_TSQ(X_tr), augment_TSQ(X_te)
    sc = StandardScaler()
    m  = Ridge(alpha=1.0).fit(sc.fit_transform(X_tr), y_tr)
    return m.predict(sc.transform(X_te))


def fit_geolinear(X_tr, y_tr, X_te, **kw):
    return GeoLinear(**kw).fit(X_tr, y_tr).predict(X_te)


_XGB_REG_DEFAULTS = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8)

def fit_xgb(X_tr, y_tr, X_te, augment=False, **kw):
    if augment:
        X_tr, X_te = augment_TSQ(X_tr), augment_TSQ(X_te)
    params = {**_XGB_REG_DEFAULTS, **kw}
    m = xgb.XGBRegressor(random_state=RNG_SEED, verbosity=0, n_jobs=-1, **params)
    return m.fit(X_tr, y_tr).predict(X_te)


# ── HPO ───────────────────────────────────────────────────────────────────────

def _cv_nmae(X, y, fn, params, n_splits=3, seed=RNG_SEED):
    """Inner CV NMAE for HPO (lower is better → minimise)."""
    scores = []
    for tr, va in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X):
        try:
            scores.append(nmae(y[va], fn(X[tr], y[tr], X[va], **params)))
        except Exception:
            scores.append(1e6)
    return float(np.mean(scores))


def hpo_geolinear_reg(X, y, n_trials=N_HPO, seed=RNG_SEED):
    def objective(trial):
        return _cv_nmae(X, y, fit_geolinear, dict(
            n_rounds               = trial.suggest_int("n_rounds", 10, 50),
            learning_rate          = trial.suggest_float("learning_rate", 0.02, 0.5, log=True),
            y_weight               = trial.suggest_float("y_weight", 0.0, 1.0),
            base_learner           = trial.suggest_categorical("base_learner",
                                                               ["ols", "ridge", "lasso"]),
            alpha                  = trial.suggest_float("alpha", 1e-4, 100.0, log=True),
            min_samples_partition  = trial.suggest_int("min_samples_partition", 3, 20),
            hvrt_n_partitions      = trial.suggest_int("hvrt_n_partitions", 4, 30),
            hvrt_inner_rounds      = trial.suggest_int("hvrt_inner_rounds", 1, 3),
            partition_inner_rounds = trial.suggest_int("partition_inner_rounds", 1, 3),
            refit_interval         = trial.suggest_categorical("refit_interval", [0, 1, 2, 5]),
            hvrt_model             = trial.suggest_categorical("hvrt_model",
                                         ["pyramid_hart", "hvrt", "hart",
                                          "fast_hvrt", "fast_hart"]),
            random_state           = seed,
        ), seed=seed)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def hpo_xgb_reg(X, y, n_trials=N_HPO, seed=RNG_SEED):
    def objective(trial):
        return _cv_nmae(X, y, fit_xgb, dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 600),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth        = trial.suggest_int("max_depth", 3, 9),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            gamma            = trial.suggest_float("gamma", 0.0, 3.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        ), seed=seed)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ── Regression benchmark ──────────────────────────────────────────────────────

REG_MODELS = ["Ridge", "Ridge+TSQ",
              "GL-OLS", "GL-Ridge", "GL-Lasso",
              "GL-HPO", "XGBoost", "XGB-HPO"]

GL_DEFAULTS = dict(random_state=RNG_SEED)


def evaluate_regression_dgp(dgp_name, dgp_fn, hpo_cache):
    rng  = np.random.default_rng(RNG_SEED)
    n    = CAL_N if dgp_name == "CalHousing" else N
    X, y = dgp_fn(n, D, rng)

    kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    results = {k: [] for k in REG_MODELS}
    gl_params = xgb_params = None

    for fold_i, (tr, te) in enumerate(kf.split(X)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        results["Ridge"].append(nmae(y_te, fit_ridge(X_tr, y_tr, X_te)))
        results["Ridge+TSQ"].append(nmae(y_te, fit_ridge(X_tr, y_tr, X_te, augment=True)))
        results["GL-OLS"].append(nmae(y_te, fit_geolinear(X_tr, y_tr, X_te,
                                                           base_learner="ols", **GL_DEFAULTS)))
        results["GL-Ridge"].append(nmae(y_te, fit_geolinear(X_tr, y_tr, X_te,
                                                             base_learner="ridge", **GL_DEFAULTS)))
        results["GL-Lasso"].append(nmae(y_te, fit_geolinear(X_tr, y_tr, X_te,
                                                             base_learner="lasso", **GL_DEFAULTS)))
        results["XGBoost"].append(nmae(y_te, fit_xgb(X_tr, y_tr, X_te)))

        if fold_i == 0:
            gl_params  = hpo_cache.setdefault(f"gl_{dgp_name}",
                         _hpo_with_log("GL-HPO",  dgp_name, hpo_geolinear_reg, X_tr, y_tr))
            xgb_params = hpo_cache.setdefault(f"xgb_{dgp_name}",
                         _hpo_with_log("XGB-HPO", dgp_name, hpo_xgb_reg,       X_tr, y_tr))

        results["GL-HPO"].append(nmae(y_te, fit_geolinear(X_tr, y_tr, X_te, **gl_params)))
        results["XGB-HPO"].append(nmae(y_te, fit_xgb(X_tr, y_tr, X_te, **xgb_params)))

    return {k: (float(np.mean(v)), float(np.std(v))) for k, v in results.items()}


def _hpo_with_log(label, dgp_name, hpo_fn, X_tr, y_tr):
    print(f"    [{label:8s}] {N_HPO} trials for {dgp_name}...", flush=True)
    params = hpo_fn(X_tr, y_tr)
    print(f"    [{label:8s}] best: {params}", flush=True)
    return params


# ── CalHousing feature engineering section ────────────────────────────────────

CAL_VARIANTS = {
    "raw"      : dict(log=False, tsq=False),
    "log"      : dict(log=True,  tsq=False),
    "raw+TSQ"  : dict(log=False, tsq=True),
    "log+TSQ"  : dict(log=True,  tsq=True),
}


def evaluate_cal_feature_engineering(hpo_cache):
    """
    For each feature variant: run GL-HPO (searched on that variant) and XGB-HPO.
    XGB-HPO params are reused from the main run (trees are invariant to
    monotone feature transforms; TSQ augmentation may shift partitions slightly).
    """
    rng     = np.random.default_rng(RNG_SEED)
    X0, y0  = dgp_california(CAL_N, D, rng)

    kf      = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    results = {}

    for variant, flags in CAL_VARIANTS.items():
        print(f"\n  [CalHousing/{variant}]", flush=True)
        X = preprocess_cal(X0, **flags)

        gl_key  = f"gl_cal_fe_{variant}"
        xgb_key = f"xgb_cal_fe_{variant}"

        fold_scores_gl  = []
        fold_scores_xgb = []
        gl_params = xgb_params = None

        for fold_i, (tr, te) in enumerate(kf.split(X)):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y0[tr], y0[te]

            if fold_i == 0:
                gl_params  = hpo_cache.setdefault(gl_key,
                             _hpo_with_log("GL-HPO",  f"CalHousing/{variant}", hpo_geolinear_reg, X_tr, y_tr))
                xgb_params = hpo_cache.setdefault(xgb_key,
                             _hpo_with_log("XGB-HPO", f"CalHousing/{variant}", hpo_xgb_reg,       X_tr, y_tr))

            fold_scores_gl.append(nmae(y_te,  fit_geolinear(X_tr, y_tr, X_te, **gl_params)))
            fold_scores_xgb.append(nmae(y_te, fit_xgb(X_tr, y_tr, X_te, **xgb_params)))

        results[variant] = {
            "GL-HPO" : (float(np.mean(fold_scores_gl)),  float(np.std(fold_scores_gl))),
            "XGB-HPO": (float(np.mean(fold_scores_xgb)), float(np.std(fold_scores_xgb))),
        }

    return results


# ── Classification ────────────────────────────────────────────────────────────

CLF_MODELS = ["GL-Ridge", "GL-HPO", "XGB", "XGB-HPO"]


def fit_gl_clf(X_tr, y_tr, X_te, **kw):
    return GeoLinearClassifier(**kw).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


_XGB_CLF_DEFAULTS = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8)

def fit_xgb_clf(X_tr, y_tr, X_te, **kw):
    params = {**_XGB_CLF_DEFAULTS, **kw}
    m = xgb.XGBClassifier(random_state=RNG_SEED, verbosity=0, n_jobs=-1,
                           eval_metric="logloss", **params)
    return m.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


def _cv_auc(X, y, fn, params, n_splits=3, seed=RNG_SEED):
    scores = []
    for tr, va in StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=seed).split(X, y):
        try:
            scores.append(roc_auc_score(y[va], fn(X[tr], y[tr], X[va], **params)))
        except Exception:
            scores.append(0.5)
    return float(np.mean(scores))


def hpo_gl_clf(X, y, n_trials=N_HPO, seed=RNG_SEED):
    def objective(trial):
        return _cv_auc(X, y, fit_gl_clf, dict(
            n_rounds               = trial.suggest_int("n_rounds", 10, 50),
            learning_rate          = trial.suggest_float("learning_rate", 0.02, 0.5, log=True),
            y_weight               = trial.suggest_float("y_weight", 0.0, 1.0),
            base_learner           = trial.suggest_categorical("base_learner",
                                                               ["ols", "ridge", "lasso"]),
            alpha                  = trial.suggest_float("alpha", 1e-4, 100.0, log=True),
            min_samples_partition  = trial.suggest_int("min_samples_partition", 3, 20),
            hvrt_n_partitions      = trial.suggest_int("hvrt_n_partitions", 4, 30),
            hvrt_inner_rounds      = trial.suggest_int("hvrt_inner_rounds", 1, 3),
            partition_inner_rounds = trial.suggest_int("partition_inner_rounds", 1, 3),
            refit_interval         = trial.suggest_categorical("refit_interval", [0, 1, 2, 5]),
            hvrt_model             = trial.suggest_categorical("hvrt_model",
                                         ["pyramid_hart", "hvrt", "hart",
                                          "fast_hvrt", "fast_hart"]),
            random_state           = seed,
        ), seed=seed)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def hpo_xgb_clf(X, y, n_trials=N_HPO, seed=RNG_SEED):
    def objective(trial):
        return _cv_auc(X, y, fit_xgb_clf, dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 600),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_depth        = trial.suggest_int("max_depth", 3, 9),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            min_child_weight = trial.suggest_int("min_child_weight", 1, 10),
            gamma            = trial.suggest_float("gamma", 0.0, 3.0),
            reg_lambda       = trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
        ), seed=seed)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def evaluate_classification(ds_name, X, y, hpo_cache):
    skf     = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RNG_SEED)
    results = {k: {"acc": [], "auc": []} for k in CLF_MODELS}
    gl_params = xgb_params = None

    for fold_i, (tr, te) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]

        # GL default (Ridge)
        p = fit_gl_clf(X_tr, y_tr, X_te, random_state=RNG_SEED)
        results["GL-Ridge"]["acc"].append(float(((p >= 0.5).astype(int) == y_te).mean()))
        results["GL-Ridge"]["auc"].append(float(roc_auc_score(y_te, p)))

        # XGB default
        p = fit_xgb_clf(X_tr, y_tr, X_te)
        results["XGB"]["acc"].append(float(((p >= 0.5).astype(int) == y_te).mean()))
        results["XGB"]["auc"].append(float(roc_auc_score(y_te, p)))

        if fold_i == 0:
            gl_params  = hpo_cache.setdefault(f"gl_clf_{ds_name}",
                         _hpo_with_log("GL-HPO",  ds_name, hpo_gl_clf,  X_tr, y_tr))
            xgb_params = hpo_cache.setdefault(f"xgb_clf_{ds_name}",
                         _hpo_with_log("XGB-HPO", ds_name, hpo_xgb_clf, X_tr, y_tr))

        for model, fn, params in [("GL-HPO",  fit_gl_clf,  gl_params),
                                   ("XGB-HPO", fit_xgb_clf, xgb_params)]:
            p = fn(X_tr, y_tr, X_te, **params)
            results[model]["acc"].append(float(((p >= 0.5).astype(int) == y_te).mean()))
            results[model]["auc"].append(float(roc_auc_score(y_te, p)))

    return {k: {m: (float(np.mean(v[m])), float(np.std(v[m]))) for m in ("acc","auc")}
            for k, v in results.items()}


# ── Formatting ────────────────────────────────────────────────────────────────

def fmt(mu, sd):
    return f"{mu:.3f}±{sd:.2f}"


def section(title):
    print(f"\n{'='*80}\n{title}\n{'='*80}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    hpo_cache = {}

    # ── Regression ────────────────────────────────────────────────────────────
    section("REGRESSION  --  5-fold CV NMAE_norm  (mean +/- std, lower is better)")
    print(f"  NMAE_norm = MAE / MAD(y_true)  [= NMAE / NMAE_ceiling]")
    print(f"  1.0 = null predictor (predict mean); 0.0 = perfect; comparable across datasets")
    print(f"  Synthetic: n={N}, d={D}   CalHousing: n={CAL_N} (subsampled), 8 features")
    print(f"  HPO: {N_HPO} Optuna TPE trials, 3-fold inner CV -- minimises NMAE_norm, searches learner too")

    all_reg = {}
    for dgp_name, dgp_fn in REGRESSION_DGPS.items():
        print(f"\n[{dgp_name}]", flush=True)
        all_reg[dgp_name] = evaluate_regression_dgp(dgp_name, dgp_fn, hpo_cache)

    # Table
    col_w = 12
    print()
    header = f"{'DGP':<13}" + "".join(f"{m:>{col_w}}" for m in REG_MODELS)
    print(header)
    print("-" * len(header))
    for dgp, res in all_reg.items():
        row = f"{dgp:<13}" + "".join(f"{fmt(*res[m]):>{col_w}}" for m in REG_MODELS)
        print(row)

    # Gap table (NMAE: negative gap = GL is better)
    print(f"\n{'DGP':<13}  {'GL-OLS':>8}  {'GL-Ridge':>8}  {'GL-Lasso':>8}  "
          f"{'GL-HPO':>8}  {'XGB-HPO':>8}  {'Gap HPO':>9}")
    print("  (Gap = GL-HPO - XGB-HPO; negative = GL wins)")
    print("-" * 70)
    for dgp, res in all_reg.items():
        gl_ols   = res["GL-OLS"][0]
        gl_ridge = res["GL-Ridge"][0]
        gl_lasso = res["GL-Lasso"][0]
        gl_hpo   = res["GL-HPO"][0]
        xhpo     = res["XGB-HPO"][0]
        gap      = gl_hpo - xhpo
        print(f"{dgp:<13}  {gl_ols:>8.3f}  {gl_ridge:>8.3f}  {gl_lasso:>8.3f}  "
              f"{gl_hpo:>8.3f}  {xhpo:>8.3f}  {gap:>+9.3f}")

    # ── CalHousing feature engineering ────────────────────────────────────────
    section("CALHOUSING FEATURE ENGINEERING  --  GL-HPO vs XGB-HPO (NMAE_norm, lower is better)")
    print(f"  Log cols: MedInc, AveRooms, Population, AveOccup (log1p)")
    print(f"  TSQ: appends T=S²-Q, S, Q cooperative statistics")
    print(f"  Each variant gets its own HPO run ({N_HPO} trials)")

    cal_fe = evaluate_cal_feature_engineering(hpo_cache)

    print(f"\n{'Variant':<12}  {'GL-HPO':>14}  {'XGB-HPO':>14}  {'Gap':>9}  {'GL<XGB?':>8}")
    print("  (Gap = GL-HPO - XGB-HPO; negative = GL wins)")
    print("-" * 62)
    for variant, res in cal_fe.items():
        gl_mu, gl_sd   = res["GL-HPO"]
        xgb_mu, xgb_sd = res["XGB-HPO"]
        gap = gl_mu - xgb_mu
        beat = "YES" if gap <= 0 else f"{gap:+.3f}"
        print(f"{variant:<12}  {fmt(gl_mu,gl_sd):>14}  {fmt(xgb_mu,xgb_sd):>14}  "
              f"{gap:>+9.3f}  {beat:>8}")

    # ── Classification ────────────────────────────────────────────────────────
    section("CLASSIFICATION  —  5-fold stratified CV  (accuracy + AUC, higher is better)")
    print(f"  HPO: {N_HPO} Optuna TPE trials, inner AUC objective, includes learner choice")

    clf_data = {
        "Synthetic"  : make_classification(n_samples=2000, n_features=10,
                                           n_informative=6, random_state=RNG_SEED),
        "BreastCancer": load_breast_cancer(return_X_y=True),
    }
    all_clf = {}
    for ds, (X_c, y_c) in clf_data.items():
        print(f"\n[{ds}]  n={len(y_c)}, d={X_c.shape[1]}, prevalence={y_c.mean():.2f}",
              flush=True)
        all_clf[ds] = evaluate_classification(ds, X_c, y_c, hpo_cache)

    print(f"\n{'Dataset/metric':<24}" + "".join(f"{m:>14}" for m in CLF_MODELS))
    print("-" * (24 + len(CLF_MODELS) * 14))
    for ds, res in all_clf.items():
        for metric in ("acc", "auc"):
            label = f"{ds}/{metric}"
            row = f"{label:<24}" + "".join(f"{fmt(*res[m][metric]):>14}" for m in CLF_MODELS)
            print(row)

    # ── HPO params summary ────────────────────────────────────────────────────
    section("BEST HPO PARAMETERS")
    for key in sorted(hpo_cache):
        print(f"\n  {key}:")
        for k, v in hpo_cache[key].items():
            print(f"    {k:<32} = {v}")


if __name__ == "__main__":
    main()
