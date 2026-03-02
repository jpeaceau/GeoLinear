"""
GeoLinear — boosted ensemble of partition-local Ridge regression and logistic
classification models fitted within HVRT cooperative geometry partitions.

The core insight: many feature-outcome relationships are piecewise-linear
in cooperative geometry.  Features interact differently depending on whether
they cooperate (T > 0) or compete (T < 0), and the coefficients governing
their relationship to the outcome change across cooperative regimes.  A
global linear model averages over these regimes.  GeoLinear discovers the
regimes via HVRT and fits interpretable Ridge models within each.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from geolinear._cpp_backend import (
    CppGeoLinearRegressor,
    CppGeoLinearClassifier,
    GeoLinearConfig,
)


# ── Thin Python wrapper around PartitionCoeffs ────────────────────────────────

class _RidgeModelView:
    """Lightweight view of a per-partition Ridge model's coefficients."""

    def __init__(self, pc):
        self.coef_      = np.array(pc.coef, dtype=np.float64)
        self.intercept_ = float(pc.intercept)
        self.fallback_  = bool(pc.fallback)
        self.n_samples_ = int(pc.n_samples)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_

    def __repr__(self):
        return (f"<RidgeModel n={self.n_samples_} "
                f"intercept={self.intercept_:.4f} "
                f"fallback={self.fallback_}>")


# ── GeoLinear ─────────────────────────────────────────────────────────────────

class GeoLinear(BaseEstimator, RegressorMixin):
    """
    Boosted partition-local Ridge regression on HVRT cooperative geometry.

    At each boosting round, HVRT partitions the feature space into cooperative
    geometry regions.  A separate Ridge model is fitted within each partition
    on the current residuals.  Predictions are summed across rounds.

    Fully sklearn-compatible: works with Pipeline, GridSearchCV, clone(), etc.

    Parameters
    ----------
    n_rounds : int, default=20
        Number of boosting rounds.
    learning_rate : float, default=0.1
        Shrinkage per round.
    y_weight : float, default=0.5
        HVRT blending parameter. 0.5 = outcome-informed (recommended for
        prediction), 1.0 = fully y-driven.
    alpha : float, default=1.0
        Ridge L2 regularisation within each partition.
    min_samples_partition : int, default=5
        Minimum samples required to fit a per-partition model. Partitions
        below this threshold predict 0 (contribute nothing to residuals).
    hvrt_n_partitions : int or None, default=None
        Target number of HVRT partitions. None = HVRT auto-tune.
    hvrt_min_samples_leaf : int or None, default=None
        Minimum samples per HVRT partition tree leaf. None = HVRT auto-tune.
    random_state : int, default=42
        Random seed. Incremented per round to ensure diverse partitionings.

    Attributes
    ----------
    intercept_ : float
        Mean of training y (initial prediction).
    stages_ : list of (None, dict[int, _RidgeModelView])
        One entry per boosting stage.
    n_features_in_ : int
        Number of features seen at fit time.
    """

    def __init__(
        self,
        n_rounds:               int   = 20,
        learning_rate:          float = 0.1,
        y_weight:               float = 0.5,
        base_learner:           str   = "ridge",
        alpha:                  float = 1.0,
        min_samples_partition:  int   = 5,
        hvrt_n_partitions:      int | None = None,
        hvrt_min_samples_leaf:  int | None = None,
        hvrt_inner_rounds:      int   = 1,
        partition_inner_rounds: int   = 1,
        refit_interval:         int   = 0,
        random_state:           int   = 42,
    ):
        self.n_rounds               = n_rounds
        self.learning_rate          = learning_rate
        self.y_weight               = y_weight
        self.base_learner           = base_learner
        self.alpha                  = alpha
        self.min_samples_partition  = min_samples_partition
        self.hvrt_n_partitions      = hvrt_n_partitions
        self.hvrt_min_samples_leaf  = hvrt_min_samples_leaf
        self.hvrt_inner_rounds      = hvrt_inner_rounds
        self.partition_inner_rounds = partition_inner_rounds
        self.refit_interval         = refit_interval
        self.random_state           = random_state

    # ── Internal ──────────────────────────────────────────────────────────────

    def _make_config(self) -> GeoLinearConfig:
        cfg = GeoLinearConfig()
        cfg.n_rounds               = int(self.n_rounds)
        cfg.learning_rate          = float(self.learning_rate)
        cfg.y_weight               = float(self.y_weight)
        cfg.base_learner           = str(self.base_learner)
        cfg.alpha                  = float(self.alpha)
        cfg.min_samples_partition  = int(self.min_samples_partition)
        cfg.hvrt_n_partitions      = int(self.hvrt_n_partitions) if self.hvrt_n_partitions is not None else -1
        cfg.hvrt_min_samples_leaf  = int(self.hvrt_min_samples_leaf) if self.hvrt_min_samples_leaf is not None else -1
        cfg.hvrt_inner_rounds      = int(self.hvrt_inner_rounds)
        cfg.partition_inner_rounds = int(self.partition_inner_rounds)
        cfg.refit_interval         = int(self.refit_interval)
        cfg.random_state           = int(self.random_state)
        return cfg

    def _build_stages(self):
        self.stages_ = []
        for s in range(self._backend.n_stages()):
            pcs = self._backend.get_stage_coeffs(s)
            partition_models = {pc.partition_id: _RidgeModelView(pc) for pc in pcs}
            self.stages_.append((None, partition_models))

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, X, y):
        """
        Fit GeoLinear to (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        if len(y) != X.shape[0]:
            raise ValueError("X and y must have the same number of rows")

        self.n_features_in_ = X.shape[1]

        self._backend = CppGeoLinearRegressor(self._make_config())
        self._backend.fit(X, y)

        self.intercept_ = self._backend.intercept()
        self._build_stages()

        return self

    def predict(self, X):
        """
        Predict continuous target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)
        return self._backend.predict(X)

    # ── Interpretability ──────────────────────────────────────────────────────

    def feature_importances(self, feature_names=None):
        """
        Global feature importance: mean absolute coefficient across all
        partitions and stages, weighted by partition size.

        Parameters
        ----------
        feature_names : list of str, optional

        Returns
        -------
        importances : ndarray of shape (n_features,)
        names : list of str
        """
        check_is_fitted(self)
        d = self.n_features_in_
        weighted_sum = np.zeros(d)
        total_weight = 0.0

        for _, partition_models in self.stages_:
            for ridge in partition_models.values():
                w = float(ridge.n_samples_)
                weighted_sum += w * np.abs(ridge.coef_)
                total_weight += w

        importances = weighted_sum / max(total_weight, 1.0)
        names = feature_names if feature_names is not None else [f"x{i}" for i in range(d)]
        return importances, names

    def __repr__(self):
        fitted = hasattr(self, '_backend') and self._backend is not None and self._backend.is_fitted()
        return (f"GeoLinear(n_rounds={self.n_rounds}, "
                f"learning_rate={self.learning_rate}, "
                f"alpha={self.alpha}, "
                f"y_weight={self.y_weight}, "
                f"fitted={fitted})")


# ── GeoLinearClassifier ───────────────────────────────────────────────────────

class GeoLinearClassifier(BaseEstimator, ClassifierMixin):
    """
    Boosted logistic classifier on HVRT cooperative geometry partitions.

    Uses IRLS (Newton) steps: pseudo-residuals = y − sigmoid(F),
    weights = sigmoid(F) · (1 − sigmoid(F)).  Each round fits a weighted
    Ridge model per partition, updating the raw score F.

    Fully sklearn-compatible: works with Pipeline, GridSearchCV, clone(), etc.

    Parameters
    ----------
    n_rounds : int, default=20
    learning_rate : float, default=0.1
    y_weight : float, default=0.5
    alpha : float, default=1.0
    min_samples_partition : int, default=5
    hvrt_n_partitions : int or None, default=None
    hvrt_min_samples_leaf : int or None, default=None
    random_state : int, default=42

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Unique class labels seen at fit time.
    n_features_in_ : int
    """

    def __init__(
        self,
        n_rounds:               int   = 20,
        learning_rate:          float = 0.1,
        y_weight:               float = 0.5,
        base_learner:           str   = "ridge",
        alpha:                  float = 1.0,
        min_samples_partition:  int   = 5,
        hvrt_n_partitions:      int | None = None,
        hvrt_min_samples_leaf:  int | None = None,
        hvrt_inner_rounds:      int   = 1,
        partition_inner_rounds: int   = 1,
        refit_interval:         int   = 0,
        random_state:           int   = 42,
    ):
        self.n_rounds               = n_rounds
        self.learning_rate          = learning_rate
        self.y_weight               = y_weight
        self.base_learner           = base_learner
        self.alpha                  = alpha
        self.min_samples_partition  = min_samples_partition
        self.hvrt_n_partitions      = hvrt_n_partitions
        self.hvrt_min_samples_leaf  = hvrt_min_samples_leaf
        self.hvrt_inner_rounds      = hvrt_inner_rounds
        self.partition_inner_rounds = partition_inner_rounds
        self.refit_interval         = refit_interval
        self.random_state           = random_state

    def _make_config(self) -> GeoLinearConfig:
        cfg = GeoLinearConfig()
        cfg.n_rounds               = int(self.n_rounds)
        cfg.learning_rate          = float(self.learning_rate)
        cfg.y_weight               = float(self.y_weight)
        cfg.base_learner           = str(self.base_learner)
        cfg.alpha                  = float(self.alpha)
        cfg.min_samples_partition  = int(self.min_samples_partition)
        cfg.hvrt_n_partitions      = int(self.hvrt_n_partitions) if self.hvrt_n_partitions is not None else -1
        cfg.hvrt_min_samples_leaf  = int(self.hvrt_min_samples_leaf) if self.hvrt_min_samples_leaf is not None else -1
        cfg.hvrt_inner_rounds      = int(self.hvrt_inner_rounds)
        cfg.partition_inner_rounds = int(self.partition_inner_rounds)
        cfg.refit_interval         = int(self.refit_interval)
        cfg.random_state           = int(self.random_state)
        return cfg

    def fit(self, X, y):
        """
        Fit GeoLinearClassifier to binary (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
            Binary labels (any two distinct values).

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).ravel()

        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(
                f"GeoLinearClassifier supports binary classification only "
                f"(got {len(self.classes_)} classes)."
            )

        y_enc = (y == self.classes_[1]).astype(np.float64)
        self.n_features_in_ = X.shape[1]

        self._backend = CppGeoLinearClassifier(self._make_config())
        self._backend.fit(X, y_enc)

        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Column 0 = P(class 0), column 1 = P(class 1).
        """
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)
        p1 = self._backend.predict_proba(X)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        check_is_fitted(self)
        X = np.asarray(X, dtype=np.float64)
        p1 = self._backend.predict_proba(X)
        idx = (p1 >= 0.5).astype(int)
        return self.classes_[idx]

    def __repr__(self):
        fitted = hasattr(self, '_backend') and self._backend is not None and self._backend.is_fitted()
        return (f"GeoLinearClassifier(n_rounds={self.n_rounds}, "
                f"learning_rate={self.learning_rate}, "
                f"alpha={self.alpha}, "
                f"y_weight={self.y_weight}, "
                f"fitted={fitted})")
