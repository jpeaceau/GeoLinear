#pragma once
#include <Eigen/Dense>
#include <vector>
#include "hvrt/types.h"

namespace hvrt {

// StandardScaler-equivalent (Variance mode) or robust median/MAD scaler (MAD mode).
// Handles both continuous and categorical columns identically
// (same algorithm, separate cat_mask tracking).
class Whitener {
public:
    // fit: compute per-feature statistics from X (n x d).
    // cat_mask[j] == true  →  column j is categorical (left un-whitened).
    // mode: Variance = mean±std (default); MAD = median±(1.4826×MAD).
    void fit(const Eigen::MatrixXd& X, const std::vector<bool>& cat_mask,
             WhiteningMode mode = WhiteningMode::Variance);

    // transform: X → X_z  (in-place copy; X unchanged)
    Eigen::MatrixXd transform(const Eigen::MatrixXd& X) const;

    // inverse_transform: X_z → X_orig
    Eigen::MatrixXd inverse_transform(const Eigen::MatrixXd& X_z) const;

    // Accessors
    const Eigen::VectorXd& means()    const { return means_; }
    const Eigen::VectorXd& stds()     const { return stds_; }
    const std::vector<bool>& cat_mask() const { return cat_mask_; }

    bool fitted() const { return fitted_; }

private:
    // Variance-mode (existing)
    Eigen::VectorXd means_;
    Eigen::VectorXd stds_;
    // MAD-mode (new)
    Eigen::VectorXd medians_;
    Eigen::VectorXd mads_;

    WhiteningMode mode_ = WhiteningMode::Variance;
    std::vector<bool> cat_mask_;
    bool fitted_ = false;

    static constexpr double kEps     = 1e-8;
    static constexpr double kMADScale = 1.4826;  // Gaussian consistency factor
};

} // namespace hvrt
