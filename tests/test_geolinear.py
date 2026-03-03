"""
Tests for GeoLinear v0.2.0.

Run with:
    pytest tests/test_geolinear.py -v
"""

import numpy as np
import pytest
from sklearn.datasets import make_regression, make_classification

from geolinear import GeoLinear, GeoLinearClassifier, augment_TSQ, TSQTransformer


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(0)
    n, d = 300, 6
    X = rng.standard_normal((n, d))
    Z = (X - X.mean(0)) / X.std(0)
    S = Z.sum(axis=1)
    Q = (Z ** 2).sum(axis=1)
    T = S ** 2 - Q
    y = 3.0 * np.sign(T) * X[:, 0] + rng.standard_normal(n) * 0.5
    return X, y


@pytest.fixture(scope="module")
def binary_data():
    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    return X, y


# ── augment_TSQ ───────────────────────────────────────────────────────────────

class TestAugmentTSQ:
    def test_shape(self):
        X = np.random.randn(50, 4)
        X_aug = augment_TSQ(X)
        assert X_aug.shape == (50, 7)

    def test_values(self):
        X = np.random.randn(20, 5)
        X_aug, stats = augment_TSQ(X, return_stats=True)
        T, S, Q = stats["T"], stats["S"], stats["Q"]
        np.testing.assert_allclose(T, S ** 2 - Q, atol=1e-10)
        np.testing.assert_array_equal(X_aug[:, -3], T)
        np.testing.assert_array_equal(X_aug[:, -2], S)
        np.testing.assert_array_equal(X_aug[:, -1], Q)

    def test_requires_2d(self):
        with pytest.raises(ValueError):
            augment_TSQ(np.ones(10))

    def test_requires_2_features(self):
        with pytest.raises(ValueError):
            augment_TSQ(np.ones((10, 1)))


class TestTSQTransformer:
    def test_fit_transform(self):
        X_tr = np.random.randn(40, 3)
        X_te = np.random.randn(10, 3)
        t = TSQTransformer()
        X_tr_aug = t.fit_transform(X_tr)
        X_te_aug = t.transform(X_te)
        assert X_tr_aug.shape == (40, 6)
        assert X_te_aug.shape == (10, 6)

    def test_not_fitted_raises(self):
        with pytest.raises(RuntimeError):
            TSQTransformer().transform(np.ones((5, 3)))

    def test_feature_names(self):
        t = TSQTransformer().fit(np.ones((5, 2)))
        names = t.get_feature_names_out(["a", "b"])
        assert names == ["a", "b", "T", "S", "Q"]

    def test_sklearn_clone(self):
        from sklearn.base import clone
        t = TSQTransformer(copy=False)
        t2 = clone(t)
        assert t2.copy == False  # noqa: E712


# ── GeoLinear (regressor) ─────────────────────────────────────────────────────

class TestGeoLinear:
    def test_fit_predict_shape(self, regression_data):
        X, y = regression_data
        model = GeoLinear(n_rounds=3, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (len(y),)

    def test_is_fitted(self, regression_data):
        X, y = regression_data
        model = GeoLinear(n_rounds=3, random_state=0)
        assert not hasattr(model, '_backend') or not model._backend
        model.fit(X, y)
        assert model._backend.is_fitted()

    def test_predict_before_fit_raises(self):
        with pytest.raises(Exception):
            GeoLinear().predict(np.ones((5, 3)))

    def test_stages_structure(self, regression_data):
        X, y = regression_data
        n_rounds = 4
        model = GeoLinear(n_rounds=n_rounds, random_state=0)
        model.fit(X, y)
        assert len(model.stages_) == n_rounds
        for _, partition_models in model.stages_:
            assert isinstance(partition_models, dict)
            for pid, ridge in partition_models.items():
                assert isinstance(pid, int)
                assert hasattr(ridge, "coef_")
                assert hasattr(ridge, "intercept_")
                assert ridge.coef_.shape == (X.shape[1],)

    def test_intercept(self, regression_data):
        X, y = regression_data
        model = GeoLinear(n_rounds=3, random_state=0)
        model.fit(X, y)
        np.testing.assert_allclose(model.intercept_, y.mean(), atol=1e-8)

    def test_r2_positive(self, regression_data):
        X, y = regression_data
        model = GeoLinear(n_rounds=10, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.2, f"R² = {r2:.3f} too low"

    def test_feature_importances(self, regression_data):
        X, y = regression_data
        model = GeoLinear(n_rounds=3, random_state=0)
        model.fit(X, y)
        imp, names = model.feature_importances()
        assert len(imp) == X.shape[1]
        assert (imp >= 0).all()

    def test_sklearn_score(self, regression_data):
        X, y = regression_data
        model = GeoLinear(n_rounds=5, random_state=0)
        model.fit(X, y)
        s = model.score(X, y)
        assert isinstance(s, float)
        assert s > 0.0

    def test_default_hyperparams(self):
        model = GeoLinear()
        assert model.n_rounds == 20
        assert model.learning_rate == 0.1


# ── GeoLinearClassifier ───────────────────────────────────────────────────────

class TestGeoLinearClassifier:
    def test_fit_predict_shape(self, binary_data):
        X, y = binary_data
        clf = GeoLinearClassifier(n_rounds=3, random_state=0)
        clf.fit(X, y)
        pred = clf.predict(X)
        assert pred.shape == (len(y),)

    def test_predict_labels_binary(self, binary_data):
        X, y = binary_data
        clf = GeoLinearClassifier(n_rounds=3, random_state=0)
        clf.fit(X, y)
        pred = clf.predict(X)
        assert set(pred).issubset(set(np.unique(y)))

    def test_predict_proba_shape(self, binary_data):
        X, y = binary_data
        clf = GeoLinearClassifier(n_rounds=3, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)

    def test_predict_proba_sums_to_one(self, binary_data):
        X, y = binary_data
        clf = GeoLinearClassifier(n_rounds=3, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-10)

    def test_predict_proba_in_range(self, binary_data):
        X, y = binary_data
        clf = GeoLinearClassifier(n_rounds=3, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_accuracy_above_baseline(self, binary_data):
        X, y = binary_data
        clf = GeoLinearClassifier(n_rounds=10, random_state=0)
        clf.fit(X, y)
        acc = (clf.predict(X) == y).mean()
        assert acc > 0.6, f"Accuracy = {acc:.3f} too low"

    def test_classes_attribute(self, binary_data):
        X, y = binary_data
        clf = GeoLinearClassifier(n_rounds=3, random_state=0)
        clf.fit(X, y)
        assert len(clf.classes_) == 2
        assert set(clf.classes_) == set(np.unique(y))

    def test_not_fitted_raises(self):
        with pytest.raises(Exception):
            GeoLinearClassifier().predict(np.ones((5, 3)))

    def test_multiclass_raises(self):
        X = np.random.randn(30, 3)
        y = np.array([0, 1, 2] * 10)
        with pytest.raises(ValueError):
            GeoLinearClassifier(n_rounds=2).fit(X, y)

    def test_sklearn_score(self, binary_data):
        X, y = binary_data
        clf = GeoLinearClassifier(n_rounds=5, random_state=0)
        clf.fit(X, y)
        s = clf.score(X, y)
        assert isinstance(s, float)
        assert 0.0 <= s <= 1.0

    def test_default_hyperparams(self):
        clf = GeoLinearClassifier()
        assert clf.n_rounds == 20
        assert clf.learning_rate == 0.1


# ── Sklearn compatibility ─────────────────────────────────────────────────────

class TestSklearnCompat:
    def test_pipeline_regressor(self, regression_data):
        from sklearn.pipeline import Pipeline
        X, y = regression_data
        pipe = Pipeline([("tsq", TSQTransformer()), ("model", GeoLinear(n_rounds=3))])
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert pred.shape == (len(y),)

    def test_pipeline_classifier(self, binary_data):
        from sklearn.pipeline import Pipeline
        X, y = binary_data
        pipe = Pipeline([("tsq", TSQTransformer()), ("clf", GeoLinearClassifier(n_rounds=3))])
        pipe.fit(X, y)
        pred = pipe.predict(X)
        assert pred.shape == (len(y),)

    def test_gridsearch_regressor(self, regression_data):
        from sklearn.model_selection import GridSearchCV
        X, y = regression_data
        gs = GridSearchCV(GeoLinear(), {"n_rounds": [3, 5]}, cv=2)
        gs.fit(X, y)
        assert gs.best_estimator_ is not None

    def test_clone_regressor(self):
        from sklearn.base import clone
        model = GeoLinear(n_rounds=5, alpha=0.5)
        cloned = clone(model)
        assert cloned.n_rounds == 5
        assert cloned.alpha == 0.5

    def test_clone_classifier(self):
        from sklearn.base import clone
        clf = GeoLinearClassifier(n_rounds=7, alpha=2.0)
        cloned = clone(clf)
        assert cloned.n_rounds == 7
        assert cloned.alpha == 2.0

    def test_get_params_regressor(self):
        model = GeoLinear(n_rounds=15, alpha=2.0)
        params = model.get_params()
        assert params["n_rounds"] == 15
        assert params["alpha"] == 2.0

    def test_set_params_regressor(self):
        model = GeoLinear()
        model.set_params(n_rounds=8, alpha=3.0)
        assert model.n_rounds == 8
        assert model.alpha == 3.0


# ── PyramidHART smoke tests ───────────────────────────────────────────────────

class TestPyramidHART:
    MODELS = ["hvrt", "hart", "fast_hvrt", "fast_hart", "pyramid_hart"]

    @pytest.fixture(scope="class")
    def small_data(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((200, 5))
        y = X[:, 0] + X[:, 1] * X[:, 2] + rng.standard_normal(200) * 0.1
        return X, y

    @pytest.mark.parametrize("model_name", MODELS)
    def test_fit_predict(self, small_data, model_name):
        X, y = small_data
        gl = GeoLinear(n_rounds=5, hvrt_model=model_name, random_state=0)
        gl.fit(X, y)
        pred = gl.predict(X)
        assert pred.shape == (len(y),)

    @pytest.mark.parametrize("model_name", MODELS)
    def test_r2_positive(self, small_data, model_name):
        X, y = small_data
        gl = GeoLinear(n_rounds=10, hvrt_model=model_name, random_state=0)
        gl.fit(X, y)
        r2 = gl.score(X, y)
        assert r2 > 0.0, f"hvrt_model={model_name!r}: R²={r2:.3f} not positive"

    def test_default_model_unchanged(self, small_data):
        """Default hvrt_model must be 'pyramid_hart' and produce the same result."""
        X, y = small_data
        gl_default = GeoLinear(n_rounds=5, random_state=0)
        gl_explicit = GeoLinear(n_rounds=5, hvrt_model="pyramid_hart", random_state=0)
        assert gl_default.hvrt_model == "pyramid_hart"
        gl_default.fit(X, y)
        gl_explicit.fit(X, y)
        np.testing.assert_allclose(
            gl_default.predict(X), gl_explicit.predict(X), atol=1e-10,
            err_msg="Default hvrt_model must equal explicit 'pyramid_hart'"
        )

    def test_pyramid_hart_classifier(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=1)
        clf = GeoLinearClassifier(n_rounds=5, hvrt_model="pyramid_hart", random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_clone_preserves_hvrt_model(self):
        from sklearn.base import clone
        model = GeoLinear(n_rounds=3, hvrt_model="pyramid_hart")
        cloned = clone(model)
        assert cloned.hvrt_model == "pyramid_hart"


# ── Latent signal amplification ───────────────────────────────────────────────

class TestLatentAmplification:
    @pytest.fixture(scope="class")
    def small_data(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((300, 6))
        Z = (X - X.mean(0)) / X.std(0)
        S = Z.sum(axis=1)
        Q = (Z ** 2).sum(axis=1)
        T = S ** 2 - Q
        y = 3.0 * np.sign(T) * X[:, 0] + rng.standard_normal(300) * 0.5
        return X, y

    def test_coop_weights_fits_and_predicts(self, small_data):
        X, y = small_data
        model = GeoLinear(n_rounds=5, use_coop_weights=True, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (len(y),)

    def test_t_feature_fits_and_predicts(self, small_data):
        X, y = small_data
        model = GeoLinear(n_rounds=5, use_t_feature=True, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (len(y),)

    def test_both_combined(self, small_data):
        X, y = small_data
        model = GeoLinear(n_rounds=5, use_coop_weights=True, use_t_feature=True, random_state=0)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (len(y),)

    def test_coop_weights_r2_positive(self, small_data):
        X, y = small_data
        model = GeoLinear(n_rounds=10, use_coop_weights=True, random_state=0)
        model.fit(X, y)
        r2 = model.score(X, y)
        assert r2 > 0.0, f"use_coop_weights R²={r2:.3f} not positive"

    def test_t_feature_r2_vs_baseline(self, small_data):
        """use_t_feature R² should be within 0.02 of baseline (may improve or be neutral)."""
        X, y = small_data
        baseline = GeoLinear(n_rounds=10, random_state=0)
        baseline.fit(X, y)
        r2_base = baseline.score(X, y)

        model = GeoLinear(n_rounds=10, use_t_feature=True, random_state=0)
        model.fit(X, y)
        r2_new = model.score(X, y)

        assert r2_new >= r2_base - 0.02, (
            f"use_t_feature R²={r2_new:.3f} dropped more than 0.02 below baseline {r2_base:.3f}"
        )

    def test_sklearn_compat(self):
        from sklearn.base import clone
        model = GeoLinear(n_rounds=3, use_coop_weights=True, use_t_feature=True)
        cloned = clone(model)
        assert cloned.use_coop_weights is True
        assert cloned.use_t_feature is True
        params = model.get_params()
        assert params["use_coop_weights"] is True
        assert params["use_t_feature"] is True
        model.set_params(use_coop_weights=False)
        assert model.use_coop_weights is False

    def test_feature_importances_with_t_feature(self, small_data):
        """feature_importances should not error and return shape (n_features,) even with use_t_feature."""
        X, y = small_data
        model = GeoLinear(n_rounds=5, use_t_feature=True, random_state=0)
        model.fit(X, y)
        imp, names = model.feature_importances()
        assert imp.shape == (X.shape[1],)
        assert (imp >= 0).all()

    def test_classifier_coop_weights(self):
        X, y = make_classification(n_samples=200, n_features=5, random_state=1)
        clf = GeoLinearClassifier(n_rounds=5, use_coop_weights=True, random_state=0)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert (proba >= 0).all() and (proba <= 1).all()
