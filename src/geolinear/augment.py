"""
augment_TSQ — cooperative geometry feature augmentation.

Appends the scalar cooperative statistics T, S, Q to any feature matrix.
These three features encode aggregate cooperative structure orthogonal to
the individual feature axes (Q-aligned geometry).

    Z = standardise(X)       # zero mean, unit variance per column
    S = Σ_j z_j              # cooperative projection (row sum)
    Q = Σ_j z_j²             # distance axis (row squared norm)
    T = S² − Q = 2·Σ_{i<j} z_i·z_j   # cooperative deviation

By Theorem 1 (T⊥Q orthogonality), T is geometrically orthogonal to Q
and cannot be recovered from individual feature values or their distances.
Appending T, S, Q to X gives any downstream estimator access to cooperative
structure without changing the estimator itself.
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def augment_TSQ(X, return_stats=False):
    """
    Append cooperative statistics T, S, Q to feature matrix X.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Raw feature matrix. Must contain at least 2 features.
    return_stats : bool, default=False
        If True, also return a dict with {'T': ..., 'S': ..., 'Q': ...}.

    Returns
    -------
    X_aug : ndarray of shape (n_samples, n_features + 3)
        Original features with T, S, Q appended as the last three columns.
    stats : dict, optional
        Returned only when return_stats=True. Keys: 'T', 'S', 'Q',
        each an ndarray of shape (n_samples,).

    Notes
    -----
    Standardisation uses training-set statistics (per-column mean and std).
    To avoid data leakage in cross-validation, standardise with training-set
    parameters and apply to test sets. See ``TSQTransformer`` for an
    sklearn-compatible fit/transform interface.

    Examples
    --------
    >>> import numpy as np
    >>> from geolinear import augment_TSQ
    >>> X = np.random.randn(100, 5)
    >>> X_aug = augment_TSQ(X)
    >>> X_aug.shape
    (100, 8)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must be 2-dimensional")
    if X.shape[1] < 2:
        raise ValueError("X must have at least 2 features for T to be non-trivial")

    # Standardise
    mu  = X.mean(axis=0)
    sig = X.std(axis=0, ddof=0)
    sig = np.where(sig < 1e-8, 1.0, sig)   # avoid division by zero
    Z   = (X - mu) / sig

    S = Z.sum(axis=1)               # (n,)
    Q = (Z ** 2).sum(axis=1)        # (n,)
    T = S ** 2 - Q                  # (n,)  cooperative deviation

    X_aug = np.column_stack([X, T, S, Q])

    if return_stats:
        return X_aug, {"T": T, "S": S, "Q": Q}
    return X_aug


class TSQTransformer(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible transformer that appends T, S, Q to X.

    Fits standardisation parameters on X_train, then applies them
    consistently to X_test, preventing data leakage.

    Parameters
    ----------
    copy : bool, default=True
        Whether to copy X before transforming.

    Examples
    --------
    >>> from geolinear import TSQTransformer
    >>> t = TSQTransformer().fit(X_train)
    >>> X_train_aug = t.transform(X_train)
    >>> X_test_aug  = t.transform(X_test)
    """

    def __init__(self, copy=True):
        self.copy = copy
        self._mu  = None
        self._sig = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self._mu  = X.mean(axis=0)
        self._sig = X.std(axis=0, ddof=0)
        self._sig = np.where(self._sig < 1e-8, 1.0, self._sig)
        return self

    def transform(self, X):
        if self._mu is None:
            raise RuntimeError("TSQTransformer: call fit() before transform()")
        X = np.array(X, dtype=np.float64, copy=self.copy)
        Z = (X - self._mu) / self._sig
        S = Z.sum(axis=1)
        Q = (Z ** 2).sum(axis=1)
        T = S ** 2 - Q
        return np.column_stack([X, T, S, Q])

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = [f"x{i}" for i in range(len(self._mu))]
        return list(input_features) + ["T", "S", "Q"]
