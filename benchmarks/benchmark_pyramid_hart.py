"""
PyramidHART: linear boosting vs single-stage linear models.

Compares all 5 hvrt_model variants (hvrt, hart, fast_hvrt, fast_hart,
pyramid_hart) across two modes:

  boosted  : n_rounds=20, learning_rate=0.1  (standard GeoLinear boosting)
  single   : n_rounds=1,  learning_rate=1.0  (one partition -> one Ridge fit)

Note: "pyramid_hart" is the GeoLinear default as of v0.3.
      "hvrt" is the original (variance whitening, T-statistic) and is
      included here for comparison.

Plus Ridge and XGBoost baselines for reference context.

Metric: NMAE_norm = MAE / MAD(y_true)
  1.0 = null predictor (predict mean); 0.0 = perfect; lower is better.
  Comparable across datasets because the noise floor is always 1.0.

Datasets: T-regime, 3-regime, Friedman1, Linear, CalHousing
          (5-fold CV, same splits across all models)

Run:
    python benchmarks/benchmark_pyramid_hart.py
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

from geolinear import GeoLinear

# ── Settings ──────────────────────────────────────────────────────────────────

RNG_SEED = 42
N        = 2000   # synthetic DGPs
D        = 10
RHO      = 0.7
N_FOLDS  = 5
CAL_N    = 5000

# Fixed GL hyperparams (no HPO — isolates architecture effect)
# hvrt_n_partitions=None means HVRT auto-tunes partitions.
GL_SHARED = dict(
    y_weight               = 0.5,
    base_learner           = "ridge",
    alpha                  = 1.0,
    min_samples_partition  = 5,
    hvrt_n_partitions      = None,
    hvrt_min_samples_leaf  = None,
    random_state           = RNG_SEED,
)

HVRT_MODELS = ["hvrt", "hart", "fast_hvrt", "fast_hart", "pyramid_hart"]

# ── DGPs ──────────────────────────────────────────────────────────────────────

def compute_T(X):
    Z = (X - X.mean(0)) / np.where(X.std(0) < 1e-8, 1.0, X.std(0))
    S = Z.sum(1); Q = (Z**2).sum(1)
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
    m0 = T < t33; m1 = (T >= t33) & (T < t67); m2 = T >= t67
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


DGPS = {
    "T-regime"  : dgp_t_regime,
    "3-regime"  : dgp_3regime,
    "Friedman1" : dgp_friedman1,
    "Linear"    : dgp_linear,
    "CalHousing": dgp_california,
}

# ── Metric ────────────────────────────────────────────────────────────────────

def nmae(y_true, y_pred):
    """MAE / MAD(y_true).  1.0 = null predictor, 0.0 = perfect, lower is better."""
    mad = float(np.mean(np.abs(y_true - y_true.mean())))
    if mad < 1e-8:
        return 0.0
    return float(np.mean(np.abs(y_true - y_pred)) / mad)


# ── Model definitions ─────────────────────────────────────────────────────────

def build_models():
    """
    Returns ordered dict: model_name → callable(X_tr, y_tr, X_te) → predictions.
    """
    models = {}

    # ── Baselines ─────────────────────────────────────────────────────────────
    def ridge(X_tr, y_tr, X_te):
        sc = StandardScaler()
        m  = Ridge(alpha=1.0).fit(sc.fit_transform(X_tr), y_tr)
        return m.predict(sc.transform(X_te))

    models["Ridge"] = ridge

    try:
        import xgboost as xgb
        _XGB = dict(n_estimators=300, learning_rate=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    random_state=RNG_SEED, verbosity=0, n_jobs=-1)
        def xgboost(X_tr, y_tr, X_te):
            return xgb.XGBRegressor(**_XGB).fit(X_tr, y_tr).predict(X_te)
        models["XGBoost"] = xgboost
    except ImportError:
        pass

    # ── GeoLinear variants ────────────────────────────────────────────────────
    for hm in HVRT_MODELS:
        tag = hm  # e.g. "hvrt", "pyramid_hart"

        # Boosted: 20 rounds, lr=0.1
        def _boosted(X_tr, y_tr, X_te, _hm=hm):
            return (GeoLinear(n_rounds=20, learning_rate=0.1,
                              hvrt_model=_hm, **GL_SHARED)
                    .fit(X_tr, y_tr).predict(X_te))
        models[f"{tag}-boost"] = _boosted

        # Single-stage: 1 round, lr=1.0
        def _single(X_tr, y_tr, X_te, _hm=hm):
            return (GeoLinear(n_rounds=1, learning_rate=1.0,
                              hvrt_model=_hm, **GL_SHARED)
                    .fit(X_tr, y_tr).predict(X_te))
        models[f"{tag}-single"] = _single

    return models


# ── CV evaluation ─────────────────────────────────────────────────────────────

def cv_nmae(X, y, model_fn, n_folds=N_FOLDS, seed=RNG_SEED):
    """5-fold CV, returns (mean_nmae, std_nmae, mean_seconds_per_fold)."""
    scores, times = [], []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        t0 = time.perf_counter()
        try:
            pred = model_fn(X[tr], y[tr], X[te])
            scores.append(nmae(y[te], pred))
        except Exception as e:
            scores.append(1.0)
            print(f"      ERROR: {e}")
        times.append(time.perf_counter() - t0)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(times))


# ── Run benchmark ─────────────────────────────────────────────────────────────

def run():
    models = build_models()
    model_names = list(models.keys())

    # results[dgp][model] = (mean, std, seconds)
    all_results = {}

    for dgp_name, dgp_fn in DGPS.items():
        rng  = np.random.default_rng(RNG_SEED)
        n    = CAL_N if dgp_name == "CalHousing" else N
        X, y = dgp_fn(n, D, rng)

        print(f"\n[{dgp_name}]  n={n}, d={X.shape[1]}", flush=True)
        all_results[dgp_name] = {}

        for mname, mfn in models.items():
            mu, sd, sec = cv_nmae(X, y, mfn)
            all_results[dgp_name][mname] = (mu, sd, sec)
            tag = f"  {mname:<24} NMAE={mu:.3f}±{sd:.3f}  ({sec:.2f}s/fold)"
            print(tag, flush=True)

    # ── Tables ────────────────────────────────────────────────────────────────
    dgp_names = list(DGPS.keys())
    COL = 12

    def fmt(mu, sd):
        return f"{mu:.3f}±{sd:.2f}"

    def section(title):
        print(f"\n{'='*80}\n{title}\n{'='*80}")

    # ── Full results table ─────────────────────────────────────────────────────
    section("NMAE_norm  (lower is better;  1.0 = null predictor,  0.0 = perfect)")
    print(f"  Synthetic: n={N}, d={D}   CalHousing: n={CAL_N} (random subsample), 8 features")
    print(f"  GeoLinear fixed params: y_weight=0.5, base_learner=ridge, alpha=1.0")
    print(f"  -boost = 20 rounds / lr=0.1   |   -single = 1 round / lr=1.0 (no boosting)")
    print()

    header = f"{'Model':<26}" + "".join(f"{d:>{COL}}" for d in dgp_names)
    print(header)
    print("-" * len(header))

    # Print in groups: baselines, then each hvrt_model pair
    groups = [
        ["Ridge"] + (["XGBoost"] if "XGBoost" in model_names else []),
    ] + [
        [f"{hm}-boost", f"{hm}-single"] for hm in HVRT_MODELS
    ]

    for group in groups:
        for mname in group:
            if mname not in model_names:
                continue
            row = f"{mname:<26}"
            for dgp in dgp_names:
                mu, sd, _ = all_results[dgp][mname]
                row += f"{fmt(mu,sd):>{COL}}"
            print(row)
        print()

    # ── Boost gain table  (single - boost, negative = boosting helps) ──────────
    section("BOOST GAIN  =  NMAE(single-stage) - NMAE(boosted)   (positive = boosting helps)")
    print(f"  Reading: +0.05 means boosting reduced NMAE by 0.05 on this DGP")
    print()

    header2 = f"{'Model':<20}" + "".join(f"{d:>{COL}}" for d in dgp_names)
    print(header2)
    print("-" * len(header2))

    for hm in HVRT_MODELS:
        bname = f"{hm}-boost"
        sname = f"{hm}-single"
        row = f"{hm:<20}"
        for dgp in dgp_names:
            b_mu = all_results[dgp][bname][0]
            s_mu = all_results[dgp][sname][0]
            gain = s_mu - b_mu  # positive = single is worse = boosting helps
            row += f"{gain:>+{COL}.3f}"
        print(row)

    # ── Variant comparison vs hvrt-boost (positive = variant is better) ───────
    section("IMPROVEMENT vs hvrt-boost  =  NMAE(hvrt-boost) - NMAE(variant-boost)")
    print(f"  Reading: +0.02 means this variant has 0.02 lower NMAE than standard hvrt-boost")
    print()

    header3 = f"{'Model':<22}" + "".join(f"{d:>{COL}}" for d in dgp_names)
    print(header3)
    print("-" * len(header3))

    for hm in HVRT_MODELS:
        if hm == "hvrt":
            continue
        bname = f"{hm}-boost"
        row = f"{hm}-boost{'':<12}"
        for dgp in dgp_names:
            base = all_results[dgp]["hvrt-boost"][0]
            this = all_results[dgp][bname][0]
            delta = base - this  # positive = this variant is better
            row += f"{delta:>+{COL}.3f}"
        print(row)

    print()
    for hm in HVRT_MODELS:
        if hm == "hvrt":
            continue
        sname = f"{hm}-single"
        row = f"{hm}-single{'':<11}"
        for dgp in dgp_names:
            base = all_results[dgp]["hvrt-single"][0]
            this = all_results[dgp][sname][0]
            delta = base - this
            row += f"{delta:>+{COL}.3f}"
        print(row)

    # ── Timing table ──────────────────────────────────────────────────────────
    section("WALL-CLOCK  (seconds per fold, 5-fold CV)")
    print()

    header4 = f"{'Model':<26}" + "".join(f"{d:>{COL}}" for d in dgp_names)
    print(header4)
    print("-" * len(header4))

    for group in groups:
        for mname in group:
            if mname not in model_names:
                continue
            row = f"{mname:<26}"
            for dgp in dgp_names:
                _, _, sec = all_results[dgp][mname]
                row += f"{sec:>{COL}.2f}"
            print(row)
        print()


if __name__ == "__main__":
    run()
