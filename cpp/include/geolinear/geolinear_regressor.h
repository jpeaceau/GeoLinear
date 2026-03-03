#pragma once
#include <Eigen/Dense>
#include <vector>
#include <map>
#include <memory>
#include "geolinear/types.h"
#include "hvrt/hvrt.h"

namespace geolinear {

// ── Per-partition Ridge model ─────────────────────────────────────────────────

struct RidgeModel {
    Eigen::VectorXd coef;       // (d,) fitted coefficients
    double          intercept;  // fitted intercept
    bool            fallback;   // true = partition too small, predicts 0

    double predict_one(const Eigen::VectorXd& x) const {
        if (fallback) return 0.0;
        return coef.dot(x) + intercept;
    }

    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const {
        if (fallback) return Eigen::VectorXd::Zero(X.rows());
        return (X * coef).array() + intercept;
    }
};

// ── One boosting stage ────────────────────────────────────────────────────────
// hvrt_trees: K inner trees (K=1 for default behaviour).
//   tree_0 is fitted on blended(T, g); trees 1..K-1 on T-residuals.
// n_parts: max partition ID + 1 for each inner tree (used for stride computation
//   when encoding cross-product partition IDs).

struct Stage {
    std::vector<std::shared_ptr<hvrt::HVRT>> hvrt_trees;
    std::vector<int>                          n_parts;
    std::map<int, RidgeModel>                models;
    // Latent amplification (empty / zero when feature disabled)
    Eigen::VectorXd                           feat_weights;  // (d,) coop weights
    double                                    t_mean = 0.0;
    double                                    t_std  = 1.0;
};

// ── GeoLinearBase ─────────────────────────────────────────────────────────────
//
// Shared boosting infrastructure for regression and classification.
// Subclasses supply init_F, pseudo_residuals, and (optionally) hessian_weights.

class GeoLinearBase {
public:
    // ── Coefficient inspection ─────────────────────────────────────────────────
    struct PartitionCoeffs {
        int                 partition_id;
        std::vector<double> coef;
        double              intercept;
        bool                fallback;
        int                 n_samples;
    };

    bool   is_fitted() const { return fitted_; }
    int    n_stages()  const { return static_cast<int>(stages_.size()); }
    double f0()        const { return F0_; }   // initial F (mean for regressor, log-odds for classifier)

    std::vector<PartitionCoeffs> get_stage_coeffs(int stage_idx) const;
    std::vector<int>             get_stage_partition_ids(int stage_idx) const;
    std::vector<int>             apply_stage(int stage_idx,
                                              const Eigen::MatrixXd& X) const;

protected:
    explicit GeoLinearBase(GeoLinearConfig cfg = GeoLinearConfig{});

    void fit_boosting(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

    // Subclass hooks
    virtual double          init_F(const Eigen::VectorXd& y) const = 0;
    virtual Eigen::VectorXd pseudo_residuals(const Eigen::VectorXd& y,
                                              const Eigen::VectorXd& F) const = 0;
    virtual Eigen::VectorXd hessian_weights(const Eigen::VectorXd& F) const;
    // Default: all-ones (regression). Classifier overrides with p*(1-p).

    Eigen::VectorXd predict_raw(const Eigen::MatrixXd& X) const;

    // Dispatch to fit_weighted_ridge or fit_lasso based on cfg_.base_learner.
    RidgeModel fit_partition_model(const Eigen::MatrixXd& X_sub,
                                    const Eigen::VectorXd& g_sub,
                                    const Eigen::VectorXd& w_sub) const;

    // Weighted Ridge: solves (Xc'WXc + αI) β = Xc'Wg via LDLT.
    // W = diag(w_sub). alpha is explicit so OLS can pass 1e-8.
    RidgeModel fit_weighted_ridge(const Eigen::MatrixXd& X_sub,
                                   const Eigen::VectorXd& g_sub,
                                   const Eigen::VectorXd& w_sub,
                                   double alpha) const;

    // Weighted Lasso via coordinate descent.
    // Objective: (1/2)||W^{1/2}(gc - Xc β)||² + α||β||₁
    RidgeModel fit_lasso(const Eigen::MatrixXd& X_sub,
                          const Eigen::VectorXd& g_sub,
                          const Eigen::VectorXd& w_sub) const;

    hvrt::HVRTConfig make_hvrt_config(int seed_offset) const;

    GeoLinearConfig    cfg_;
    bool               fitted_ = false;
    double             F0_     = 0.0;
    std::vector<Stage> stages_;
    std::vector<std::map<int, int>> stage_partition_sizes_;

    // Refit path: persistent HVRT re-used across rounds (nullptr when refit disabled)
    std::shared_ptr<hvrt::HVRT> base_hvrt_;
};

// ── GeoLinearRegressor ────────────────────────────────────────────────────────
//
// Boosted ensemble of partition-local Ridge models.
// init: F₀ = mean(y).  residuals: y − F.  weights: ones.

class GeoLinearRegressor : public GeoLinearBase {
public:
    explicit GeoLinearRegressor(GeoLinearConfig cfg = GeoLinearConfig{});

    GeoLinearRegressor& fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::VectorXd     predict(const Eigen::MatrixXd& X) const;

    double intercept() const { return F0_; }

protected:
    double          init_F(const Eigen::VectorXd& y) const override;
    Eigen::VectorXd pseudo_residuals(const Eigen::VectorXd& y,
                                      const Eigen::VectorXd& F) const override;
    // hessian_weights() inherits all-ones default
};

// ── GeoLinearClassifier ───────────────────────────────────────────────────────
//
// Boosted logistic classifier using IRLS (Newton) steps.
// init: F₀ = clamp(log(p̄/(1−p̄)), −4, 4).
// residuals: y − sigmoid(F).  weights: sigmoid(F)·(1−sigmoid(F)).

class GeoLinearClassifier : public GeoLinearBase {
public:
    explicit GeoLinearClassifier(GeoLinearConfig cfg = GeoLinearConfig{});

    GeoLinearClassifier& fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);
    Eigen::VectorXd      predict(const Eigen::MatrixXd& X) const;        // 0/1 labels
    Eigen::VectorXd      predict_proba(const Eigen::MatrixXd& X) const;  // sigmoid(F)

protected:
    double          init_F(const Eigen::VectorXd& y) const override;
    Eigen::VectorXd pseudo_residuals(const Eigen::VectorXd& y,
                                      const Eigen::VectorXd& F) const override;
    Eigen::VectorXd hessian_weights(const Eigen::VectorXd& F) const override;
};

} // namespace geolinear
