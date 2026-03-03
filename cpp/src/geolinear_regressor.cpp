#include "geolinear/geolinear_regressor.h"
#include "hvrt/types.h"
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <cmath>
#include <set>
#include <unordered_map>

namespace geolinear {

// ── Helpers ───────────────────────────────────────────────────────────────────

static inline double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

static inline double soft_threshold(double x, double thresh) {
    if (x >  thresh) return x - thresh;
    if (x < -thresh) return x + thresh;
    return 0.0;
}

// ── Inner-boosting helpers ────────────────────────────────────────────────────

// Leaf-mean predictor: for each sample return the mean of `target` within its partition.
static Eigen::VectorXd compute_leaf_means(const Eigen::VectorXi& pids,
                                           const Eigen::VectorXd& target)
{
    const int n = static_cast<int>(pids.size());
    std::unordered_map<int, double> sum_map, cnt_map;
    sum_map.reserve(32);
    cnt_map.reserve(32);
    for (int i = 0; i < n; ++i) {
        int p = pids[i];
        sum_map[p] += target[i];
        cnt_map[p] += 1.0;
    }
    Eigen::VectorXd means(n);
    for (int i = 0; i < n; ++i) {
        int p = pids[i];
        means[i] = sum_map[p] / cnt_map[p];
    }
    return means;
}

// Cross-product partition encoding.
// Each sample gets an ID = sum_k( leaf_k[i] * stride_k ) where
//   stride_k = product( n_parts[0..k-1] ),  n_parts[k] = max_leaf_id_k + 1.
// n_parts_out is filled with max_partition_id+1 for each tree.
static Eigen::VectorXi compute_combo_pids(
    const std::vector<std::shared_ptr<hvrt::HVRT>>& trees,
    std::vector<int>& n_parts_out)
{
    const int n = static_cast<int>(trees[0]->partition_ids().size());
    n_parts_out.clear();
    n_parts_out.reserve(trees.size());
    for (const auto& t : trees)
        n_parts_out.push_back(t->partition_ids().maxCoeff() + 1);

    if (trees.size() == 1)
        return trees[0]->partition_ids();

    Eigen::VectorXi combo(n);
    combo.setZero();
    int stride = 1;
    for (int k = 0; k < static_cast<int>(trees.size()); ++k) {
        const Eigen::VectorXi& pk = trees[k]->partition_ids();
        for (int i = 0; i < n; ++i)
            combo[i] += pk[i] * stride;
        stride *= n_parts_out[k];
    }
    return combo;
}

// Apply all inner trees of a stage to new data X and return combined partition IDs.
static Eigen::VectorXi apply_combo_stage(const Stage& stage,
                                          const Eigen::MatrixXd& X)
{
    const int n = static_cast<int>(X.rows());
    if (stage.hvrt_trees.size() == 1)
        return stage.hvrt_trees[0]->apply(X);

    Eigen::VectorXi combo(n);
    combo.setZero();
    int stride = 1;
    for (int k = 0; k < static_cast<int>(stage.hvrt_trees.size()); ++k) {
        Eigen::VectorXi pk = stage.hvrt_trees[k]->apply(X);
        for (int i = 0; i < n; ++i)
            combo[i] += pk[i] * stride;
        stride *= stage.n_parts[k];
    }
    return combo;
}

// ── GeoLinearBase ─────────────────────────────────────────────────────────────

GeoLinearBase::GeoLinearBase(GeoLinearConfig cfg)
    : cfg_(std::move(cfg)) {}

hvrt::HVRTConfig GeoLinearBase::make_hvrt_config(int seed_offset) const {
    hvrt::HVRTConfig hcfg;
    hcfg.y_weight     = static_cast<float>(cfg_.y_weight);
    hcfg.n_bins       = 32;
    hcfg.random_state = cfg_.random_state + seed_offset;
    hcfg.auto_tune    = true;

    if (cfg_.hvrt_min_samples_leaf > 0)
        hcfg.min_samples_leaf = cfg_.hvrt_min_samples_leaf;

    if (cfg_.hvrt_n_partitions > 0) {
        hcfg.n_partitions = cfg_.hvrt_n_partitions;
        hcfg.auto_tune    = false;
    }

    // Map hvrt_model string → geometry_mode, whitening_mode, split_criterion
    const std::string& m = cfg_.hvrt_model;
    if (m == "pyramid_hart") {
        hcfg.geometry_mode   = hvrt::GeometryMode::A;
        hcfg.whitening_mode  = hvrt::WhiteningMode::MAD;
        hcfg.split_criterion = hvrt::SplitCriterion::AbsoluteError;
    } else if (m == "hart") {
        hcfg.geometry_mode   = hvrt::GeometryMode::T;
        hcfg.whitening_mode  = hvrt::WhiteningMode::MAD;
        hcfg.split_criterion = hvrt::SplitCriterion::AbsoluteError;
    } else if (m == "fast_hvrt") {
        hcfg.geometry_mode   = hvrt::GeometryMode::S;
        hcfg.whitening_mode  = hvrt::WhiteningMode::Variance;
        hcfg.split_criterion = hvrt::SplitCriterion::Variance;
    } else if (m == "fast_hart") {
        hcfg.geometry_mode   = hvrt::GeometryMode::S;
        hcfg.whitening_mode  = hvrt::WhiteningMode::MAD;
        hcfg.split_criterion = hvrt::SplitCriterion::AbsoluteError;
    } else {
        // default: "hvrt" — existing behavior unchanged
        hcfg.geometry_mode   = hvrt::GeometryMode::T;
        hcfg.whitening_mode  = hvrt::WhiteningMode::Variance;
        hcfg.split_criterion = hvrt::SplitCriterion::Variance;
    }

    return hcfg;
}

// ── fit_weighted_ridge ────────────────────────────────────────────────────────
// Weighted Ridge with centering.
//
// Computes weighted means, centers X and g, then solves:
//   (Xc'WXc + αI) β = Xc'Wgc   via LDLT
//   intercept = ḡ_w − x̄_w · β
//
// where W = diag(w_sub). For regression w=ones, reducing to ordinary centering.

RidgeModel GeoLinearBase::fit_weighted_ridge(const Eigen::MatrixXd& X_sub,
                                              const Eigen::VectorXd& g_sub,
                                              const Eigen::VectorXd& w_sub,
                                              double alpha) const
{
    const int n = static_cast<int>(X_sub.rows());
    const int d = static_cast<int>(X_sub.cols());

    if (n < cfg_.min_samples_partition) {
        RidgeModel m;
        m.coef      = Eigen::VectorXd::Zero(d);
        m.intercept = 0.0;
        m.fallback  = true;
        return m;
    }

    double w_sum = w_sub.sum();
    if (w_sum < 1e-10) {
        RidgeModel m;
        m.coef      = Eigen::VectorXd::Zero(d);
        m.intercept = 0.0;
        m.fallback  = true;
        return m;
    }

    // Weighted means
    Eigen::VectorXd x_mean = (X_sub.array().colwise() * w_sub.array()).colwise().sum() / w_sum;
    double          g_mean = w_sub.dot(g_sub) / w_sum;

    // Center
    Eigen::MatrixXd Xc = X_sub.rowwise() - x_mean.transpose();
    Eigen::VectorXd gc = g_sub.array() - g_mean;

    // sqrt-weighted: Xw = sqrt(w) ⊙ Xc, gw = sqrt(w) ⊙ gc
    Eigen::VectorXd sqrt_w = w_sub.array().sqrt();
    Eigen::MatrixXd Xw     = Xc.array().colwise() * sqrt_w.array();
    Eigen::VectorXd gw     = gc.array() * sqrt_w.array();

    // Normal equations: Xw'Xw + αI
    Eigen::MatrixXd XtWX = Xw.transpose() * Xw;
    XtWX.diagonal().array() += alpha;

    Eigen::VectorXd XtWg = Xw.transpose() * gw;

    Eigen::VectorXd coef = XtWX.ldlt().solve(XtWg);

    RidgeModel m;
    m.coef      = coef;
    m.intercept = g_mean - x_mean.dot(coef);
    m.fallback  = false;
    return m;
}

// ── fit_lasso ─────────────────────────────────────────────────────────────────
// Weighted Lasso via coordinate descent.
// Centers X and g with weighted means, then minimises:
//   (1/2) ||W^{1/2}(gc - Xc β)||² + α ||β||₁
// Uses soft-thresholding with efficient residual maintenance.

RidgeModel GeoLinearBase::fit_lasso(const Eigen::MatrixXd& X_sub,
                                     const Eigen::VectorXd& g_sub,
                                     const Eigen::VectorXd& w_sub) const
{
    const int n = static_cast<int>(X_sub.rows());
    const int d = static_cast<int>(X_sub.cols());

    if (n < cfg_.min_samples_partition) {
        RidgeModel m;
        m.coef = Eigen::VectorXd::Zero(d); m.intercept = 0.0; m.fallback = true;
        return m;
    }

    double w_sum = w_sub.sum();
    if (w_sum < 1e-10) {
        RidgeModel m;
        m.coef = Eigen::VectorXd::Zero(d); m.intercept = 0.0; m.fallback = true;
        return m;
    }

    // Weighted centering
    Eigen::VectorXd x_mean = (X_sub.array().colwise() * w_sub.array()).colwise().sum() / w_sum;
    double          g_mean = w_sub.dot(g_sub) / w_sum;

    Eigen::MatrixXd Xc = X_sub.rowwise() - x_mean.transpose();
    Eigen::VectorXd gc = g_sub.array() - g_mean;

    // Sqrt-weighted matrices
    Eigen::VectorXd sqrt_w = w_sub.array().sqrt();
    Eigen::MatrixXd Xw     = Xc.array().colwise() * sqrt_w.array();
    Eigen::VectorXd gw     = gc.array() * sqrt_w.array();

    // Pre-compute column norms squared: z_j = ||Xw_j||²
    Eigen::VectorXd z = Xw.colwise().squaredNorm();

    // Coordinate descent
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(d);
    Eigen::VectorXd r    = gw;            // r = gw - Xw*beta; beta=0 so r=gw

    const double alpha    = cfg_.alpha;
    const double tol      = 1e-6;
    const int    max_iter = 1000;

    for (int it = 0; it < max_iter; ++it) {
        double max_change = 0.0;

        for (int j = 0; j < d; ++j) {
            if (z[j] < 1e-10) continue;

            // Partial correlation — restore beta_j's contribution to r
            double rho_j    = Xw.col(j).dot(r) + z[j] * beta[j];
            double beta_new = soft_threshold(rho_j, alpha) / z[j];
            double delta    = beta_new - beta[j];

            if (std::abs(delta) > 1e-12) {
                r.noalias() -= delta * Xw.col(j);
                beta[j]      = beta_new;
                max_change   = std::max(max_change, std::abs(delta));
            }
        }
        if (max_change < tol) break;
    }

    RidgeModel m;
    m.coef      = beta;
    m.intercept = g_mean - x_mean.dot(beta);
    m.fallback  = false;
    return m;
}

// ── fit_partition_model ───────────────────────────────────────────────────────
// Dispatches to the learner selected by cfg_.base_learner.

RidgeModel GeoLinearBase::fit_partition_model(const Eigen::MatrixXd& X_sub,
                                               const Eigen::VectorXd& g_sub,
                                               const Eigen::VectorXd& w_sub) const
{
    if (cfg_.base_learner == "lasso")
        return fit_lasso(X_sub, g_sub, w_sub);
    // OLS: Ridge with effectively-zero regularisation
    const double a = (cfg_.base_learner == "ols") ? 1e-8 : cfg_.alpha;
    return fit_weighted_ridge(X_sub, g_sub, w_sub, a);
}

// ── hessian_weights (base default) ───────────────────────────────────────────

Eigen::VectorXd GeoLinearBase::hessian_weights(const Eigen::VectorXd& F) const {
    return Eigen::VectorXd::Ones(F.size());
}

// ── fit_boosting (shared loop) ────────────────────────────────────────────────

void GeoLinearBase::fit_boosting(const Eigen::MatrixXd& X, const Eigen::VectorXd& y)
{
    const int  n         = static_cast<int>(X.rows());
    const bool use_refit = (cfg_.refit_interval > 0);

    stages_.clear();
    stage_partition_sizes_.clear();
    stages_.reserve(cfg_.n_rounds);
    stage_partition_sizes_.reserve(cfg_.n_rounds);

    F0_ = init_F(y);
    Eigen::VectorXd F = Eigen::VectorXd::Constant(n, F0_);

    base_hvrt_.reset();

    for (int r = 0; r < cfg_.n_rounds; ++r) {

        // ── Pseudo-residuals and IRLS weights ─────────────────────────────────
        Eigen::VectorXd g = pseudo_residuals(y, F);
        Eigen::VectorXd w = hessian_weights(F);

        // ── Build h0: first inner HVRT tree (blended T + g) ───────────────────
        std::shared_ptr<hvrt::HVRT> h0;
        if (!use_refit) {
            h0 = std::make_shared<hvrt::HVRT>(make_hvrt_config(r + 1));
            h0->fit(X, g);
        } else {
            if (!base_hvrt_) {
                base_hvrt_ = std::make_shared<hvrt::HVRT>(make_hvrt_config(0));
                base_hvrt_->fit(X, g);
            } else if (r % cfg_.refit_interval == 0) {
                base_hvrt_->refit(g);
            }
            h0 = std::make_shared<hvrt::HVRT>(*base_hvrt_);
        }

        // ── Inner T-residual boosting (hvrt_inner_rounds > 1) ─────────────────
        // Each subsequent tree is fitted on the residual cooperative geometry
        // signal T - accumulated_leaf_means.  Pure geometry target (no g blend):
        // the y-signal is reserved for the linear model fitting below.
        std::vector<std::shared_ptr<hvrt::HVRT>> inner_trees;
        inner_trees.reserve(cfg_.hvrt_inner_rounds);
        inner_trees.push_back(h0);

        if (cfg_.hvrt_inner_rounds > 1) {
            Eigen::VectorXd T_res = h0->geometry_target();  // T before y-blend
            T_res -= compute_leaf_means(h0->partition_ids(), T_res);

            for (int k = 1; k < cfg_.hvrt_inner_rounds; ++k) {
                auto hk = std::make_shared<hvrt::HVRT>(*h0);  // reuse whitening/binning
                hk->refit_with_target(T_res);
                T_res -= compute_leaf_means(hk->partition_ids(), T_res);
                inner_trees.push_back(std::move(hk));
            }
        }

        // ── Cross-product partition IDs ────────────────────────────────────────
        std::vector<int> n_parts_per_tree;
        Eigen::VectorXi combo_pids = compute_combo_pids(inner_trees, n_parts_per_tree);

        std::set<int>    upids_set(combo_pids.data(), combo_pids.data() + n);
        std::vector<int> upids(upids_set.begin(), upids_set.end());

        // ── Fit per-partition linear models against outer residuals g ──────────
        Stage              stage;
        stage.hvrt_trees = std::move(inner_trees);
        stage.n_parts    = std::move(n_parts_per_tree);
        std::map<int, int> psizes;

        // ── Latent-signal feature transforms ──────────────────────────────────
        const bool any_transform = cfg_.use_coop_weights || cfg_.use_t_feature;
        Eigen::MatrixXd X_fit;

        if (any_transform) {
            const Eigen::MatrixXd& Xz = stage.hvrt_trees[0]->X_z();
            const int D = static_cast<int>(Xz.cols());
            Eigen::VectorXd S_vec = Xz.rowwise().sum();

            X_fit = X;

            if (cfg_.use_coop_weights) {
                Eigen::VectorXd fw(D);
                for (int k = 0; k < D; ++k) {
                    Eigen::VectorXd S_mk  = S_vec - Xz.col(k);
                    Eigen::VectorXd dk    = Xz.col(k).array() - Xz.col(k).mean();
                    Eigen::VectorXd dsmk  = S_mk.array() - S_mk.mean();
                    double cov    = dk.dot(dsmk) / n;
                    double std_k  = std::sqrt(dk.squaredNorm() / n) + 1e-8;
                    double std_mk = std::sqrt(dsmk.squaredNorm() / n) + 1e-8;
                    double corr   = cov / (std_k * std_mk);
                    fw(k) = corr * corr;
                }
                double max_w = fw.maxCoeff();
                if (max_w > 0)
                    fw = (fw / max_w).array() + 0.1;
                else
                    fw = Eigen::VectorXd::Constant(D, 1.0);
                X_fit = X.array().rowwise() * fw.transpose().array();
                stage.feat_weights = std::move(fw);
            }

            if (cfg_.use_t_feature) {
                Eigen::VectorXd Q_vec = Xz.rowwise().squaredNorm();
                Eigen::VectorXd t_raw = S_vec.array().square() - Q_vec.array();
                double tm = t_raw.mean();
                double ts = std::sqrt((t_raw.array() - tm).square().mean()) + 1e-8;
                stage.t_mean = tm;
                stage.t_std  = ts;
                Eigen::VectorXd t_norm = (t_raw.array() - tm) / ts;

                Eigen::MatrixXd X_aug(n, X_fit.cols() + 1);
                X_aug.leftCols(X_fit.cols()) = X_fit;
                X_aug.col(X_fit.cols()) = t_norm;
                X_fit = std::move(X_aug);
            }
        }

        const Eigen::MatrixXd& X_ref = any_transform ? X_fit : X;

        for (int pid : upids) {
            std::vector<int> idx;
            idx.reserve(n / static_cast<int>(upids.size()) + 1);
            for (int i = 0; i < n; ++i)
                if (combo_pids[i] == pid) idx.push_back(i);

            const int np = static_cast<int>(idx.size());
            psizes[pid] = np;

            if (np < cfg_.min_samples_partition) {
                RidgeModel m;
                m.coef      = Eigen::VectorXd::Zero(X_ref.cols());
                m.intercept = 0.0;
                m.fallback  = true;
                stage.models[pid] = std::move(m);
                continue;
            }

            Eigen::MatrixXd X_p(np, X_ref.cols());
            Eigen::VectorXd g_p(np);
            Eigen::VectorXd w_p(np);
            for (int k = 0; k < np; ++k) {
                X_p.row(k) = X_ref.row(idx[k]);
                g_p[k]     = g[idx[k]];
                w_p[k]     = w[idx[k]];
            }

            // ── Inner linear boosting within this intersection partition ────
            // Each round fits the base learner on the running inner residual,
            // accumulating coef and intercept into a single effective model.
            // For OLS: round 1 sees a near-zero residual (already explained).
            // For Ridge/Lasso: each round corrects residual regularisation bias.
            {
                RidgeModel acc;
                acc.coef      = Eigen::VectorXd::Zero(X_ref.cols());
                acc.intercept = 0.0;
                acc.fallback  = false;

                Eigen::VectorXd r = g_p;  // inner residual, updated each round

                for (int m = 0; m < cfg_.partition_inner_rounds; ++m) {
                    RidgeModel fm = fit_partition_model(X_p, r, w_p);
                    if (fm.fallback) { acc.fallback = true; break; }
                    r             -= fm.predict(X_p);
                    acc.coef      += fm.coef;
                    acc.intercept += fm.intercept;
                }

                stage.models[pid] = std::move(acc);
            }
        }

        // ── Update F ──────────────────────────────────────────────────────────
        Eigen::VectorXd pred(n);
        for (int i = 0; i < n; ++i) {
            auto it = stage.models.find(combo_pids[i]);
            pred[i] = (it != stage.models.end())
                       ? it->second.predict_one(X_ref.row(i))
                       : 0.0;
        }

        F.noalias() += cfg_.learning_rate * pred;

        stage_partition_sizes_.push_back(std::move(psizes));
        stages_.push_back(std::move(stage));
    }

    fitted_ = true;
}

// ── predict_raw ───────────────────────────────────────────────────────────────

Eigen::VectorXd GeoLinearBase::predict_raw(const Eigen::MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("model not fitted");

    const int n = static_cast<int>(X.rows());
    Eigen::VectorXd F = Eigen::VectorXd::Constant(n, F0_);

    const bool any_transform = cfg_.use_coop_weights || cfg_.use_t_feature;

    for (const Stage& stage : stages_) {
        Eigen::VectorXi pids = apply_combo_stage(stage, X);

        Eigen::MatrixXd X_fit;
        if (any_transform) {
            Eigen::MatrixXd X_cur = X;
            if (cfg_.use_coop_weights && stage.feat_weights.size() > 0)
                X_cur = X.array().rowwise() * stage.feat_weights.transpose().array();
            if (cfg_.use_t_feature) {
                Eigen::MatrixXd Xz = stage.hvrt_trees[0]->to_z(X);
                Eigen::VectorXd S  = Xz.rowwise().sum();
                Eigen::VectorXd Q  = Xz.rowwise().squaredNorm();
                Eigen::VectorXd t  = ((S.array().square() - Q.array()) - stage.t_mean) / stage.t_std;
                Eigen::MatrixXd X_aug(n, X_cur.cols() + 1);
                X_aug.leftCols(X_cur.cols()) = X_cur;
                X_aug.col(X_cur.cols()) = t;
                X_fit = std::move(X_aug);
            } else {
                X_fit = std::move(X_cur);
            }
        }

        const Eigen::MatrixXd& X_use = any_transform ? X_fit : X;

        for (int i = 0; i < n; ++i) {
            auto it = stage.models.find(pids[i]);
            if (it != stage.models.end())
                F[i] += cfg_.learning_rate * it->second.predict_one(X_use.row(i));
        }
    }

    return F;
}

// ── get_stage_coeffs ──────────────────────────────────────────────────────────

std::vector<GeoLinearBase::PartitionCoeffs>
GeoLinearBase::get_stage_coeffs(int stage_idx) const
{
    if (stage_idx < 0 || stage_idx >= static_cast<int>(stages_.size()))
        throw std::out_of_range("stage_idx out of range");

    const Stage& stage  = stages_[stage_idx];
    const auto&  psizes = stage_partition_sizes_[stage_idx];

    std::vector<PartitionCoeffs> result;
    result.reserve(stage.models.size());

    for (const auto& [pid, model] : stage.models) {
        PartitionCoeffs pc;
        pc.partition_id = pid;
        pc.coef.assign(model.coef.data(), model.coef.data() + model.coef.size());
        pc.intercept    = model.intercept;
        pc.fallback     = model.fallback;
        auto sit        = psizes.find(pid);
        pc.n_samples    = (sit != psizes.end()) ? sit->second : 0;
        result.push_back(std::move(pc));
    }

    return result;
}

// ── get_stage_partition_ids ───────────────────────────────────────────────────
// Returns combined (cross-product) partition IDs for training data.
// Reconstructed from each inner tree's stored training-time partition_ids_.

std::vector<int> GeoLinearBase::get_stage_partition_ids(int stage_idx) const {
    if (stage_idx < 0 || stage_idx >= static_cast<int>(stages_.size()))
        throw std::out_of_range("stage_idx out of range");

    const Stage& stage = stages_[stage_idx];
    if (stage.hvrt_trees.size() == 1) {
        const Eigen::VectorXi& pids = stage.hvrt_trees[0]->partition_ids();
        return std::vector<int>(pids.data(), pids.data() + pids.size());
    }

    // Reconstruct cross-product from stored training partition IDs
    const int n = static_cast<int>(stage.hvrt_trees[0]->partition_ids().size());
    Eigen::VectorXi combo(n);
    combo.setZero();
    int stride = 1;
    for (int k = 0; k < static_cast<int>(stage.hvrt_trees.size()); ++k) {
        const Eigen::VectorXi& pk = stage.hvrt_trees[k]->partition_ids();
        for (int i = 0; i < n; ++i)
            combo[i] += pk[i] * stride;
        stride *= stage.n_parts[k];
    }
    return std::vector<int>(combo.data(), combo.data() + combo.size());
}

// ── apply_stage ───────────────────────────────────────────────────────────────

std::vector<int> GeoLinearBase::apply_stage(int stage_idx,
                                              const Eigen::MatrixXd& X) const
{
    if (stage_idx < 0 || stage_idx >= static_cast<int>(stages_.size()))
        throw std::out_of_range("stage_idx out of range");

    Eigen::VectorXi pids = apply_combo_stage(stages_[stage_idx], X);
    return std::vector<int>(pids.data(), pids.data() + pids.size());
}

// ── GeoLinearRegressor ────────────────────────────────────────────────────────

GeoLinearRegressor::GeoLinearRegressor(GeoLinearConfig cfg)
    : GeoLinearBase(std::move(cfg)) {}

double GeoLinearRegressor::init_F(const Eigen::VectorXd& y) const {
    return y.mean();
}

Eigen::VectorXd GeoLinearRegressor::pseudo_residuals(const Eigen::VectorXd& y,
                                                       const Eigen::VectorXd& F) const {
    return y - F;
}

GeoLinearRegressor& GeoLinearRegressor::fit(const Eigen::MatrixXd& X,
                                             const Eigen::VectorXd& y) {
    fit_boosting(X, y);
    return *this;
}

Eigen::VectorXd GeoLinearRegressor::predict(const Eigen::MatrixXd& X) const {
    return predict_raw(X);
}

// ── GeoLinearClassifier ───────────────────────────────────────────────────────

GeoLinearClassifier::GeoLinearClassifier(GeoLinearConfig cfg)
    : GeoLinearBase(std::move(cfg)) {}

double GeoLinearClassifier::init_F(const Eigen::VectorXd& y) const {
    double p_bar  = y.mean();
    p_bar         = std::max(1e-6, std::min(1.0 - 1e-6, p_bar));
    double logodds = std::log(p_bar / (1.0 - p_bar));
    return std::max(-4.0, std::min(4.0, logodds));
}

Eigen::VectorXd GeoLinearClassifier::pseudo_residuals(const Eigen::VectorXd& y,
                                                        const Eigen::VectorXd& F) const {
    return y - F.unaryExpr([](double x) { return sigmoid(x); });
}

Eigen::VectorXd GeoLinearClassifier::hessian_weights(const Eigen::VectorXd& F) const {
    return F.unaryExpr([](double x) {
        double p = sigmoid(x);
        return p * (1.0 - p);
    });
}

GeoLinearClassifier& GeoLinearClassifier::fit(const Eigen::MatrixXd& X,
                                               const Eigen::VectorXd& y) {
    fit_boosting(X, y);
    return *this;
}

Eigen::VectorXd GeoLinearClassifier::predict(const Eigen::MatrixXd& X) const {
    return predict_raw(X).unaryExpr([](double x) { return x >= 0.0 ? 1.0 : 0.0; });
}

Eigen::VectorXd GeoLinearClassifier::predict_proba(const Eigen::MatrixXd& X) const {
    return predict_raw(X).unaryExpr([](double x) { return sigmoid(x); });
}

} // namespace geolinear
