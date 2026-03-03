#include "hvrt/whitener.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <vector>

namespace hvrt {

// Column median via nth_element on a copy of the column values.
static double col_median(std::vector<double> vals) {
    int n = static_cast<int>(vals.size());
    if (n == 0) return 0.0;
    std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
    if (n % 2 == 1) return vals[n / 2];
    double upper = vals[n / 2];
    std::nth_element(vals.begin(), vals.begin() + n / 2 - 1, vals.end());
    return 0.5 * (vals[n / 2 - 1] + upper);
}

void Whitener::fit(const Eigen::MatrixXd& X, const std::vector<bool>& cat_mask,
                   WhiteningMode mode) {
    const int n = static_cast<int>(X.rows());
    const int d = static_cast<int>(X.cols());
    if (static_cast<int>(cat_mask.size()) != d) {
        throw std::invalid_argument("cat_mask size must match X columns");
    }

    cat_mask_ = cat_mask;
    mode_     = mode;

    if (mode == WhiteningMode::MAD) {
        medians_.resize(d);
        mads_.resize(d);
        for (int j = 0; j < d; ++j) {
            if (cat_mask[j]) {
                medians_[j] = 0.0;
                mads_[j]    = 1.0;
                continue;
            }
            std::vector<double> col(n);
            for (int i = 0; i < n; ++i) col[i] = X(i, j);
            medians_[j] = col_median(col);
            // Compute absolute deviations from median
            for (int i = 0; i < n; ++i) col[i] = std::abs(col[i] - medians_[j]);
            double raw_mad = col_median(col);
            mads_[j] = std::max(raw_mad * kMADScale, kEps);
        }
        // Also fill means_/stds_ with sentinel values so accessors don't crash
        means_ = medians_;
        stds_  = mads_;
    } else {
        means_ = X.colwise().mean();

        // std = sqrt(mean((X - mean)^2))  — population std (ddof=0)
        Eigen::MatrixXd centered = X.rowwise() - means_.transpose();
        stds_ = (centered.array().square().colwise().mean()).sqrt();

        // Guard against zero-std (constant features)
        for (int j = 0; j < d; ++j) {
            if (stds_[j] < kEps) stds_[j] = 1.0;
        }
    }

    fitted_ = true;
}

Eigen::MatrixXd Whitener::transform(const Eigen::MatrixXd& X) const {
    if (!fitted_) throw std::runtime_error("Whitener not fitted");
    if (mode_ == WhiteningMode::MAD) {
        Eigen::MatrixXd X_z = X.rowwise() - medians_.transpose();
        X_z = X_z.array().rowwise() / mads_.transpose().array();
        return X_z;
    }
    Eigen::MatrixXd X_z = (X.rowwise() - means_.transpose());
    X_z = X_z.array().rowwise() / stds_.transpose().array();
    return X_z;
}

Eigen::MatrixXd Whitener::inverse_transform(const Eigen::MatrixXd& X_z) const {
    if (!fitted_) throw std::runtime_error("Whitener not fitted");
    if (mode_ == WhiteningMode::MAD) {
        Eigen::MatrixXd X = X_z.array().rowwise() * mads_.transpose().array();
        X = X.rowwise() + medians_.transpose();
        return X;
    }
    Eigen::MatrixXd X = X_z.array().rowwise() * stds_.transpose().array();
    X = X.rowwise() + means_.transpose();
    return X;
}

} // namespace hvrt
