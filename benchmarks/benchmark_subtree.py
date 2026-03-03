"""
Benchmark: HVRT/PyramidHART partitions + MAE sub-tree base learner.

Architecture variants tested:

  GL-Ridge   : HVRT partitions → Ridge per partition  (existing GeoLinear)
  GL-Tree    : HVRT partitions → MAE decision-tree per partition → leaf medians
  GL-TreeRidge: HVRT partitions → MAE decision-tree per partition → Ridge per leaf

All three are boosted over n_rounds=20 stages.  The question:
does replacing the Ridge linear model with an MAE sub-tree (with leaf medians
or leaf Ridges) improve over the existing partition-local linear model?

Sub-tree variants are implemented in pure Python on top of the C++ HVRT
partitions exposed via apply_stage() and get_stage_partition_ids().

Also includes for reference:
  - Ridge (global linear)
  - GBT-MAE: sklearn GradientBoostingRegressor(loss='absolute_error')
  - XGBoost

Metric: NMAE_norm = MAE / MAD(y)  [1.0 = null predictor, 0.0 = perfect]
Datasets: T-regime, 3-regime, Friedman1, Linear, CalHousing
          5-fold CV, same splits across all models.

Run:
    python benchmarks/benchmark_subtree.py
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

from geolinear import GeoLinear

# ── Settings ──────────────────────────────────────────────────────────────────

RNG_SEED   = 42
N, D, RHO  = 2000, 10, 0.7
N_FOLDS    = 5
CAL_N      = 5000

# Shared GeoLinear hyperparams (no HPO)
GL_KW = dict(
    y_weight              = 0.5,
    base_learner          = "ridge",
    alpha                 = 1.0,
    min_samples_partition = 5,
    hvrt_n_partitions     = None,  # auto-tune
    random_state          = RNG_SEED,
)

HVRT_MODEL = "pyramid_hart"   # best-performing variant from previous benchmark


# ── DGPs ──────────────────────────────────────────────────────────────────────

def compute_T(X):
    Z = (X - X.mean(0)) / np.where(X.std(0) < 1e-8, 1.0, X.std(0))
    return (Z.sum(1))**2 - (Z**2).sum(1)

def dgp_t_regime(n, d, rng):
    X = rng.standard_normal((n, d))
    y = (2 + 3*np.sign(compute_T(X)))*X[:,0] + rng.standard_normal(n)*0.5
    return X, y

def dgp_3regime(n, d, rng):
    X = rng.standard_normal((n, d))
    T = compute_T(X)
    t33, t67 = np.percentile(T, 33), np.percentile(T, 67)
    y = np.zeros(n)
    m0,m1,m2 = T<t33, (T>=t33)&(T<t67), T>=t67
    y[m0] = 3*X[m0,0] + rng.standard_normal(m0.sum())*0.5
    y[m1] = 3*X[m1,1] + rng.standard_normal(m1.sum())*0.5
    y[m2] = 3*X[m2,2] + rng.standard_normal(m2.sum())*0.5
    return X, y

def dgp_friedman1(n, d, rng):
    from scipy.special import ndtr
    cov = np.array([[RHO**abs(i-j) for j in range(d)] for i in range(d)])
    X   = rng.standard_normal((n, d)) @ np.linalg.cholesky(cov).T
    Xu  = ndtr(X)
    y   = (10*np.sin(np.pi*Xu[:,0]*Xu[:,1]) + 20*(Xu[:,2]-0.5)**2
           + 10*Xu[:,3] + 5*Xu[:,4] + rng.standard_normal(n))
    return X, y

def dgp_linear(n, d, rng):
    beta = rng.standard_normal(d)
    X    = rng.standard_normal((n, d))
    return X, X@beta + rng.standard_normal(n)*0.5

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
    mad = float(np.mean(np.abs(y_true - y_true.mean())))
    return float(np.mean(np.abs(y_true - y_pred)) / mad) if mad > 1e-8 else 0.0


# ── GeoLinear sub-tree: Python-level boosted model ────────────────────────────
#
# Each round:
#   1. One-round GeoLinear on pseudo-residuals g → HVRT partitions (structure only)
#   2. Within each HVRT partition: fit DecisionTreeRegressor(criterion="absolute_error")
#       leaf_mode="median" : use tree prediction directly (sklearn DT uses leaf median
#                            when criterion="absolute_error")
#       leaf_mode="ridge"  : get leaf assignments, fit Ridge on each sub-leaf
#   3. Accumulate F += lr * predictions
#
# Prediction uses apply_stage(0, X_test) on each stored one-round GL to get
# per-stage partition IDs for test data, then routes through stored sub-trees.

class GeoLinearSubtree:
    """
    HVRT cooperative-geometry partitions + MAE decision sub-tree per partition.

    Parameters
    ----------
    n_rounds : int
        Boosting rounds.
    learning_rate : float
    hvrt_model : str
        HVRT variant ("hvrt", "pyramid_hart", etc.)
    max_leaf_nodes : int
        Maximum leaves in the sub-tree within each HVRT partition.
    min_samples_leaf : int
        Minimum samples per leaf in the sub-tree.
    leaf_mode : str
        "median"  — leaf prediction is the leaf median (standard MAE tree)
        "ridge"   — fit Ridge(alpha=1) on the X values within each leaf
    """
    def __init__(self, n_rounds=20, learning_rate=0.1,
                 hvrt_model="pyramid_hart",
                 max_leaf_nodes=4, min_samples_leaf=5,
                 leaf_mode="median",
                 random_state=42, **gl_kw):
        self.n_rounds         = n_rounds
        self.learning_rate    = learning_rate
        self.hvrt_model       = hvrt_model
        self.max_leaf_nodes   = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.leaf_mode        = leaf_mode
        self.random_state     = random_state
        self.gl_kw            = gl_kw

    # ── helpers ───────────────────────────────────────────────────────────────

    def _fit_sub_tree(self, X_p, g_p, seed):
        """Fit MAE decision tree on one partition's (X, g)."""
        dt = DecisionTreeRegressor(
            criterion="absolute_error",
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf,
            random_state=seed,
        ).fit(X_p, g_p)
        return dt

    def _fit_leaf_ridges(self, dt, X_p, g_p):
        """Fit Ridge on each leaf of dt. Returns dict: leaf_id → (coef, intercept)."""
        leaf_ids = dt.apply(X_p)
        ridges = {}
        for lid in np.unique(leaf_ids):
            mask = leaf_ids == lid
            X_l, g_l = X_p[mask], g_p[mask]
            if len(X_l) >= max(3, X_l.shape[1]):
                m = Ridge(alpha=1.0).fit(X_l, g_l)
                ridges[lid] = (m.coef_.copy(), float(m.intercept_))
            else:
                ridges[lid] = (None, float(np.median(g_l)))
        return ridges

    def _predict_partition(self, dt, ridges, X_p):
        """Predict for samples in one partition using the sub-tree + optional Ridge."""
        if self.leaf_mode == "median":
            return dt.predict(X_p)
        # leaf_mode == "ridge"
        leaf_ids = dt.apply(X_p)
        p = np.zeros(len(X_p))
        for lid in np.unique(leaf_ids):
            mask = leaf_ids == lid
            coef, intercept = ridges.get(lid, (None, 0.0))
            if coef is not None:
                p[mask] = X_p[mask] @ coef + intercept
            else:
                p[mask] = intercept
        return p

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n = len(y)

        self.F0_ = float(y.mean())
        F = np.full(n, self.F0_)
        self._stages = []

        for r in range(self.n_rounds):
            g = y - F  # pseudo-residuals

            # One-round GeoLinear on g → only used for its HVRT partition structure
            gl_r = GeoLinear(
                n_rounds=1, learning_rate=1.0,
                hvrt_model=self.hvrt_model,
                random_state=self.random_state + r,
                **self.gl_kw
            )
            gl_r.fit(X, g)

            # Training-time partition IDs
            pids = np.array(gl_r._backend.get_stage_partition_ids(0))

            # Fit sub-trees per partition
            sub_trees  = {}
            leaf_ridges = {}

            for pid in np.unique(pids):
                mask = pids == pid
                X_p, g_p = X[mask], g[mask]
                if len(X_p) < self.min_samples_leaf:
                    continue
                dt = self._fit_sub_tree(X_p, g_p, self.random_state + r + pid)
                sub_trees[pid] = dt
                if self.leaf_mode == "ridge":
                    leaf_ridges[pid] = self._fit_leaf_ridges(dt, X_p, g_p)

            # Training predictions
            pred = np.zeros(n)
            for pid in np.unique(pids):
                mask = pids == pid
                if pid not in sub_trees:
                    continue
                pred[mask] = self._predict_partition(
                    sub_trees[pid],
                    leaf_ridges.get(pid, {}),
                    X[mask]
                )

            F += self.learning_rate * pred
            self._stages.append({
                'gl'         : gl_r,
                'sub_trees'  : sub_trees,
                'leaf_ridges': leaf_ridges,
            })

        return self

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = len(X)
        F = np.full(n, self.F0_)

        for stage in self._stages:
            pids = np.array(stage['gl']._backend.apply_stage(0, X))
            pred = np.zeros(n)
            for pid in np.unique(pids):
                mask = pids == pid
                if pid not in stage['sub_trees']:
                    continue
                pred[mask] = self._predict_partition(
                    stage['sub_trees'][pid],
                    stage['leaf_ridges'].get(pid, {}),
                    X[mask]
                )
            F += self.learning_rate * pred

        return F

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))


# ── Model factory ─────────────────────────────────────────────────────────────

def build_models():
    models = {}

    # Baseline: global Ridge
    def ridge(X_tr, y_tr, X_te):
        sc = StandardScaler()
        return Ridge(alpha=1.0).fit(sc.fit_transform(X_tr), y_tr).predict(sc.transform(X_te))
    models["Ridge"] = ridge

    # sklearn GBT with MAE loss (reference tree ensemble)
    def gbt_mae(X_tr, y_tr, X_te):
        return (GradientBoostingRegressor(
            loss="absolute_error", n_estimators=100,
            learning_rate=0.1, max_depth=3, random_state=RNG_SEED
        ).fit(X_tr, y_tr).predict(X_te))
    models["GBT-MAE"] = gbt_mae

    # XGBoost reference
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

    # Existing GeoLinear (boosted Ridge, best variant from prior benchmark)
    def gl_ridge(X_tr, y_tr, X_te):
        return GeoLinear(n_rounds=20, learning_rate=0.1, hvrt_model=HVRT_MODEL,
                         **GL_KW).fit(X_tr, y_tr).predict(X_te)
    models["GL-Ridge"] = gl_ridge

    # Sub-tree variants, varying max_leaf_nodes
    for max_leaves in [2, 4, 8]:
        tag = f"GL-Tree-{max_leaves}"
        def _tree(X_tr, y_tr, X_te, _ml=max_leaves):
            return (GeoLinearSubtree(
                n_rounds=20, learning_rate=0.1, hvrt_model=HVRT_MODEL,
                max_leaf_nodes=_ml, min_samples_leaf=5,
                leaf_mode="median", **GL_KW
            ).fit(X_tr, y_tr).predict(X_te))
        models[tag] = _tree

        tag_r = f"GL-TreeRidge-{max_leaves}"
        def _tree_ridge(X_tr, y_tr, X_te, _ml=max_leaves):
            return (GeoLinearSubtree(
                n_rounds=20, learning_rate=0.1, hvrt_model=HVRT_MODEL,
                max_leaf_nodes=_ml, min_samples_leaf=5,
                leaf_mode="ridge", **GL_KW
            ).fit(X_tr, y_tr).predict(X_te))
        models[tag_r] = _tree_ridge

    return models


# ── CV evaluation ─────────────────────────────────────────────────────────────

def cv_nmae(X, y, model_fn, n_folds=N_FOLDS, seed=RNG_SEED):
    scores, times = [], []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for tr, te in kf.split(X):
        t0 = time.perf_counter()
        try:
            pred = model_fn(X[tr], y[tr], X[te])
            scores.append(nmae(y[te], pred))
        except Exception as e:
            scores.append(1.0)
            print(f"      ERROR: {e}", flush=True)
        times.append(time.perf_counter() - t0)
    return float(np.mean(scores)), float(np.std(scores)), float(np.mean(times))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    models     = build_models()
    model_names = list(models.keys())
    dgp_names   = list(DGPS.keys())

    # Collect results
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
            print(f"  {mname:<22} NMAE={mu:.3f}+-{sd:.3f}  ({sec:.2f}s/fold)", flush=True)

    # ── Tables ────────────────────────────────────────────────────────────────
    COL = 13

    def fmt(mu, sd, _sec=None):
        return f"{mu:.3f}+-{sd:.2f}"

    def section(title):
        print(f"\n{'='*80}\n{title}\n{'='*80}")

    section("NMAE_norm  (lower is better;  1.0 = null predictor,  0.0 = perfect)")
    print(f"  hvrt_model={HVRT_MODEL!r}  n_rounds=20  lr=0.1  y_weight=0.5  alpha=1.0")
    print(f"  GL-Tree-k   : partition-local MAE tree (k leaves), leaf = median")
    print(f"  GL-TreeRidge-k: same tree, leaf = Ridge(alpha=1) on X within each leaf")
    print()

    header = f"{'Model':<24}" + "".join(f"{d:>{COL}}" for d in dgp_names)
    print(header)
    print("-" * len(header))

    groups = [
        ["Ridge", "GBT-MAE"] + (["XGBoost"] if "XGBoost" in model_names else []),
        ["GL-Ridge"],
        [f"GL-Tree-{k}"      for k in [2, 4, 8]],
        [f"GL-TreeRidge-{k}" for k in [2, 4, 8]],
    ]
    for group in groups:
        for mname in group:
            if mname not in model_names:
                continue
            row = f"{mname:<24}" + "".join(
                f"{fmt(*all_results[d][mname]):>{COL}}" for d in dgp_names)
            print(row)
        print()

    # ── Delta vs GL-Ridge (negative = improvement) ────────────────────────────
    section("DELTA vs GL-Ridge  (negative = improvement over existing baseline)")
    print(f"  -0.05 means NMAE is 0.05 lower than GL-Ridge = 5% better relative to noise floor")
    print()

    header2 = f"{'Model':<24}" + "".join(f"{d:>{COL}}" for d in dgp_names)
    print(header2)
    print("-" * len(header2))

    ref_name = "GL-Ridge"
    for group in groups[2:]:  # only sub-tree variants
        for mname in group:
            if mname not in model_names:
                continue
            row = f"{mname:<24}"
            for d in dgp_names:
                delta = all_results[d][mname][0] - all_results[d][ref_name][0]
                row += f"{delta:>+{COL}.3f}"
            print(row)
        print()

    # ── Timing ────────────────────────────────────────────────────────────────
    section("WALL-CLOCK  (seconds per fold, 5-fold CV)")
    print()

    header3 = f"{'Model':<24}" + "".join(f"{d:>{COL}}" for d in dgp_names)
    print(header3)
    print("-" * len(header3))

    for group in groups:
        for mname in group:
            if mname not in model_names:
                continue
            row = f"{mname:<24}" + "".join(
                f"{all_results[d][mname][2]:>{COL}.2f}" for d in dgp_names)
            print(row)
        print()


if __name__ == "__main__":
    main()
