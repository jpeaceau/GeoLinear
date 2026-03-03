#include "hvrt/tree.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <stdexcept>
#ifdef _OPENMP
#  include <omp.h>
#endif

namespace hvrt {

// ── Auto-tune ────────────────────────────────────────────────────────────────

std::pair<int,int> PartitionTree::auto_tune_params(int n, int d, bool for_reduction) {
    int msl, max_leaf;
    if (for_reduction) {
        msl = std::max(5, (d * 40 * 2) / 3);
    } else {
        msl = static_cast<int>(std::max(static_cast<double>(d + 2),
                                        std::sqrt(static_cast<double>(n))));
    }
    max_leaf = std::max(30, std::min(1500, 3 * n / (msl * 2)));
    return {max_leaf, msl};
}

// ── Variance reduction gain ───────────────────────────────────────────────────
// gain = n * var(parent) - n_left * var(left) - n_right * var(right)
// Using Welford / running formula:
//   var_gain = (sum_sq - sum*sum/n) - [(sum_sq_L - sum_L^2/n_L) + (sum_sq_R - sum_R^2/n_R)]
// Equivalent to: (sum_L/n_L - sum_R/n_R)^2 * n_L*n_R / n  (for equal-variance split gain)

static double variance_gain(double sum_p, double sum_sq_p, int n_p,
                             double sum_l, double sum_sq_l, int n_l) {
    if (n_l <= 0 || n_l >= n_p) return 0.0;
    int n_r = n_p - n_l;
    double sum_r = sum_p - sum_l;
    double sum_sq_r = sum_sq_p - sum_sq_l;

    // variance of parent node (unnormalised: sum_sq - sum^2/n)
    double var_p   = sum_sq_p - sum_p  * sum_p  / n_p;
    double var_l   = sum_sq_l - sum_l  * sum_l  / n_l;
    double var_r   = sum_sq_r - sum_r  * sum_r  / n_r;

    double gain = var_p - var_l - var_r;
    return gain;
}

// ── MAE helpers for AbsoluteError criterion ───────────────────────────────────

// Compute total absolute deviation from the median using a target-bin histogram.
// tbin[tb] = count of samples in bin tb. n_total = total samples.
static double mae_from_bins(const std::vector<int>& tbin, int n_total, int n_tbins) {
    if (n_total == 0) return 0.0;
    // Find median bin: smallest tb where cumulative count >= (n_total+1)/2
    int med_bin = n_tbins - 1;
    {
        int cum = 0;
        int half = (n_total + 1) / 2;
        for (int tb = 0; tb < n_tbins; ++tb) {
            cum += tbin[tb];
            if (cum >= half) { med_bin = tb; break; }
        }
    }
    double mae = 0.0;
    for (int tb = 0; tb < n_tbins; ++tb)
        mae += std::abs(tb - med_bin) * tbin[tb];
    return mae;
}

// Approximate MAE gain = parent_mae - mae_left - mae_right (in bin units).
// tbin_left[tb] = count of left-group samples in target bin tb.
// tbin_total[tb] = count of ALL node samples in target bin tb.
static double mae_gain(const std::vector<int>& tbin_left,
                        const std::vector<int>& tbin_total,
                        int n_left, int n_node, int n_tbins,
                        double parent_mae) {
    if (n_left <= 0 || n_left >= n_node) return 0.0;
    int n_right = n_node - n_left;

    double mae_left = mae_from_bins(tbin_left, n_left, n_tbins);

    // Derive right group bins
    std::vector<int> tbin_right(n_tbins);
    for (int tb = 0; tb < n_tbins; ++tb)
        tbin_right[tb] = tbin_total[tb] - tbin_left[tb];

    double mae_right = mae_from_bins(tbin_right, n_right, n_tbins);

    return parent_mae - mae_left - mae_right;
}

// Exact total MAE for a small list of values (sort + median by nth_element).
static double exact_total_mae(std::vector<double> vals) {
    int n = static_cast<int>(vals.size());
    if (n == 0) return 0.0;
    std::nth_element(vals.begin(), vals.begin() + n / 2, vals.end());
    double med = vals[n / 2];
    double mae = 0.0;
    for (double v : vals) mae += std::abs(v - med);
    return mae;
}

// ── Continuous split evaluation ───────────────────────────────────────────────
//
// Two-stage algorithm:
//   A. Transposed scatter (sample-outer, feature-inner):
//      X_binned is RowMajor → row(idx) is contiguous in fi → cache-friendly.
//      If OpenMP is available, threads split the sample range; each keeps
//      thread-local histogram arrays and merges under a critical section.
//   B. Prefix scan per feature: independent → can run after merge.

PartitionTree::SplitResult PartitionTree::evaluate_continuous_splits(
    const std::vector<int>& indices,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_binned,
    const std::vector<int>& cont_cols,
    const Eigen::VectorXd& target,
    const std::vector<Eigen::VectorXd>& bin_edges,
    int n_bins,
    int min_samples_leaf,
    const HVRTConfig& cfg) const
{
    const int n_node = static_cast<int>(indices.size());
    const int d_cont = static_cast<int>(cont_cols.size());
    // nb_max: worst-case bins per feature; actual nb per feature may be less.
    const int nb_max = n_bins + 1;

    SplitResult best;
    best.valid = false;
    best.gain  = -1.0;

    if (d_cont == 0) return best;

    const int stride = static_cast<int>(X_binned.cols()); // == d_cont (RowMajor)

    // ── AbsoluteError criterion: double-histogram ─────────────────────────────
    if (cfg.split_criterion == SplitCriterion::AbsoluteError) {
        static constexpr int n_tbins = 32;

        // Compute target range for this node
        double t_min = target[indices[0]], t_max = t_min;
        for (int si = 1; si < n_node; ++si) {
            double t = target[indices[si]];
            if (t < t_min) t_min = t;
            if (t > t_max) t_max = t;
        }
        double t_range = t_max - t_min;
        if (t_range < 1e-10) return best;  // constant target — no gain possible

        // Precompute tbin_total (independent of feature)
        std::vector<int> tbin_total(n_tbins, 0);
        for (int si = 0; si < n_node; ++si) {
            double t = target[indices[si]];
            int tb = static_cast<int>((t - t_min) / t_range * n_tbins);
            if (tb >= n_tbins) tb = n_tbins - 1;
            tbin_total[tb]++;
        }
        double parent_mae = mae_from_bins(tbin_total, n_node, n_tbins);

        // 3D histogram: bin_tbin_cnt[fi][b][tb] stored flat as
        //   fi * nb_max * n_tbins + b * n_tbins + tb
        std::vector<int> bin_tbin_cnt(d_cont * nb_max * n_tbins, 0);
        std::vector<int> bin_cnt(d_cont * nb_max, 0);

        // ── Stage A: scatter ─────────────────────────────────────────────────
        for (int si = 0; si < n_node; ++si) {
            const int idx = indices[si];
            double t = target[idx];
            int tb = static_cast<int>((t - t_min) / t_range * n_tbins);
            if (tb >= n_tbins) tb = n_tbins - 1;
            const uint8_t* row = X_binned.data() + (ptrdiff_t)idx * stride;
            for (int fi = 0; fi < d_cont; ++fi) {
                const int b = static_cast<int>(row[fi]);
                bin_tbin_cnt[fi * nb_max * n_tbins + b * n_tbins + tb]++;
                bin_cnt[fi * nb_max + b]++;
            }
        }

        // ── Stage B: prefix scan per feature ─────────────────────────────────
        for (int fi = 0; fi < d_cont; ++fi) {
            const int nb = static_cast<int>(bin_edges[fi].size()) - 1;
            if (nb <= 0) continue;

            std::vector<int> tbin_left(n_tbins, 0);
            int cum_cnt = 0;
            for (int b = 0; b < nb - 1; ++b) {
                // Accumulate bin b into left group
                const int base_tb = fi * nb_max * n_tbins + b * n_tbins;
                for (int tb = 0; tb < n_tbins; ++tb)
                    tbin_left[tb] += bin_tbin_cnt[base_tb + tb];
                cum_cnt += bin_cnt[fi * nb_max + b];

                if (cum_cnt < min_samples_leaf || (n_node - cum_cnt) < min_samples_leaf) continue;

                const double g = mae_gain(tbin_left, tbin_total, cum_cnt, n_node,
                                          n_tbins, parent_mae);
                if (g > best.gain) {
                    best.valid     = true;
                    best.gain      = g;
                    best.feature   = cont_cols[fi];
                    best.bin       = b;
                    best.threshold = bin_edges[fi][b + 1];
                    best.is_binary = false;
                }
            }
        }
        return best;
    }

    // ── Variance criterion (default) ──────────────────────────────────────────
    // Flat histogram storage: feature fi, bin b → index fi * nb_max + b.
    std::vector<double> bin_sum(d_cont * nb_max, 0.0);
    std::vector<double> bin_sum_sq(d_cont * nb_max, 0.0);
    std::vector<int>    bin_cnt(d_cont * nb_max, 0);

    double sum_p = 0.0, sum_sq_p = 0.0;

    // ── Stage A: transposed scatter ──────────────────────────────────────────
#ifdef _OPENMP
    // Each thread accumulates its own local histograms and reduces under a
    // critical section.  Allocation is proportional to d_cont * nb_max per
    // thread — typically a few KB, negligible vs. scatter work.
    #pragma omp parallel
    {
        double my_sum_p = 0.0, my_sum_sq_p = 0.0;
        std::vector<double> my_sum(d_cont * nb_max, 0.0);
        std::vector<double> my_sum_sq(d_cont * nb_max, 0.0);
        std::vector<int>    my_cnt(d_cont * nb_max, 0);

        #pragma omp for schedule(static)
        for (int si = 0; si < n_node; ++si) {
            const int    idx = indices[si];
            const double t   = target[idx];
            const double t2  = t * t;
            my_sum_p    += t;
            my_sum_sq_p += t2;
            const uint8_t* row = X_binned.data() + (ptrdiff_t)idx * stride;
            for (int fi = 0; fi < d_cont; ++fi) {
                const int b    = static_cast<int>(row[fi]);
                const int base = fi * nb_max;
                my_sum[base + b]    += t;
                my_sum_sq[base + b] += t2;
                my_cnt[base + b]    += 1;
            }
        }

        #pragma omp critical
        {
            sum_p    += my_sum_p;
            sum_sq_p += my_sum_sq_p;
            const int flat = d_cont * nb_max;
            for (int k = 0; k < flat; ++k) {
                bin_sum[k]    += my_sum[k];
                bin_sum_sq[k] += my_sum_sq[k];
                bin_cnt[k]    += my_cnt[k];
            }
        }
    } // end parallel
#else
    for (int si = 0; si < n_node; ++si) {
        const int    idx = indices[si];
        const double t   = target[idx];
        const double t2  = t * t;
        sum_p    += t;
        sum_sq_p += t2;
        const uint8_t* row = X_binned.data() + (ptrdiff_t)idx * stride;
        for (int fi = 0; fi < d_cont; ++fi) {
            const int b    = static_cast<int>(row[fi]);
            const int base = fi * nb_max;
            bin_sum[base + b]    += t;
            bin_sum_sq[base + b] += t2;
            bin_cnt[base + b]    += 1;
        }
    }
#endif

    // ── Stage B: prefix scan per feature ─────────────────────────────────────
    // Features are independent; serial scan is typically fast (d_cont * n_bins).
    for (int fi = 0; fi < d_cont; ++fi) {
        const int nb   = static_cast<int>(bin_edges[fi].size()) - 1;
        if (nb <= 0) continue;
        const int base = fi * nb_max;

        double cum_sum = 0.0, cum_sum_sq = 0.0;
        int    cum_cnt = 0;
        for (int b = 0; b < nb - 1; ++b) {
            cum_sum    += bin_sum[base + b];
            cum_sum_sq += bin_sum_sq[base + b];
            cum_cnt    += bin_cnt[base + b];

            // Skip splits that violate msl on either side — this ensures the
            // gain formula can only select splits that will actually be committed.
            if (cum_cnt < min_samples_leaf || (n_node - cum_cnt) < min_samples_leaf) continue;

            const double g = variance_gain(sum_p, sum_sq_p, n_node,
                                           cum_sum, cum_sum_sq, cum_cnt);
            if (g > best.gain) {
                best.valid     = true;
                best.gain      = g;
                best.feature   = cont_cols[fi];
                best.bin       = b;
                best.threshold = bin_edges[fi][b + 1];  // right boundary: routes ~bin_cnt[b] left
                best.is_binary = false;
            }
        }
    }
    return best;
}

// ── Binary split evaluation ───────────────────────────────────────────────────

PartitionTree::SplitResult PartitionTree::evaluate_binary_splits(
    const std::vector<int>& indices,
    const Eigen::MatrixXd& X_z,
    const std::vector<int>& binary_cols,
    const Eigen::VectorXd& target,
    const HVRTConfig& cfg) const
{
    const int n_node = static_cast<int>(indices.size());
    SplitResult best;
    best.valid = false;
    best.gain  = -1.0;

    if (binary_cols.empty()) return best;

    // ── AbsoluteError criterion: exact MAE ────────────────────────────────────
    if (cfg.split_criterion == SplitCriterion::AbsoluteError) {
        std::vector<double> parent_vals;
        parent_vals.reserve(n_node);
        for (int idx : indices) parent_vals.push_back(target[idx]);
        double parent_mae = exact_total_mae(parent_vals);

        for (int fc : binary_cols) {
            std::vector<double> left_vals, right_vals;
            left_vals.reserve(n_node);
            right_vals.reserve(n_node);
            for (int idx : indices) {
                if (X_z(idx, fc) <= 0.0) left_vals.push_back(target[idx]);
                else                       right_vals.push_back(target[idx]);
            }
            if (left_vals.empty() || right_vals.empty()) continue;
            double g = parent_mae - exact_total_mae(left_vals) - exact_total_mae(right_vals);
            if (g > best.gain) {
                best.valid     = true;
                best.gain      = g;
                best.feature   = fc;
                best.threshold = 0.0;
                best.is_binary = true;
            }
        }
        return best;
    }

    // ── Variance criterion (default) ──────────────────────────────────────────
    double sum_p = 0.0, sum_sq_p = 0.0;
    for (int idx : indices) {
        double t = target[idx];
        sum_p   += t;
        sum_sq_p+= t * t;
    }

    for (int fc : binary_cols) {
        // Threshold at 0 (features are whitened, binary: ~0 or ~1)
        double sum_l = 0.0, sum_sq_l = 0.0;
        int    cnt_l = 0;
        for (int idx : indices) {
            if (X_z(idx, fc) <= 0.0) {
                sum_l    += target[idx];
                sum_sq_l += target[idx] * target[idx];
                ++cnt_l;
            }
        }
        double g = variance_gain(sum_p, sum_sq_p, n_node,
                                 sum_l, sum_sq_l, cnt_l);
        if (g > best.gain) {
            best.valid     = true;
            best.gain      = g;
            best.feature   = fc;
            best.threshold = 0.0;
            best.is_binary = true;
        }
    }
    return best;
}

// ── Build ─────────────────────────────────────────────────────────────────────

Eigen::VectorXi PartitionTree::build(
    const Eigen::MatrixXd& X_z,
    const Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X_binned,
    const std::vector<int>& cont_cols,
    const std::vector<int>& binary_cols,
    const Eigen::VectorXd& target,
    const HVRTConfig& cfg)
{
    const int n = static_cast<int>(X_z.rows());
    d_full_ = static_cast<int>(X_z.cols());
    const int d_cont = static_cast<int>(cont_cols.size());

    // Determine limits
    int max_leaves  = cfg.n_partitions;
    int msl         = cfg.min_samples_leaf;

    if (cfg.auto_tune) {
        auto [ml, ms] = auto_tune_params(n, d_full_, /*for_reduction=*/true);
        max_leaves = ml;
        msl        = ms;
    }

    // Cache bin edges for apply()
    cont_cols_cached_   = cont_cols;
    binary_cols_cached_ = binary_cols;
    // Extract bin edges from X_binned range (need them for apply threshold lookup)
    // We store threshold in the node directly, so bin_edges only needed during build.
    // Build a local bin-edge structure from X_z / X_binned:
    // The binner was already called; we receive a pre-computed X_binned.
    // For apply, we only need the threshold stored in the node.

    // Initialize feature importances
    feature_importances_.assign(d_full_, 0.0);

    // Root node covers all samples
    nodes_.clear();
    nodes_.reserve(2 * max_leaves);
    nodes_.push_back(TreeNode{});  // root = node 0

    // Build bin_edges for each continuous feature from unique sorted values in X_z
    // (needed for evaluate_continuous_splits)
    std::vector<Eigen::VectorXd> bin_edges(d_cont);
    for (int fi = 0; fi < d_cont; ++fi) {
        int fc = cont_cols[fi];
        // Collect unique sorted edges from X_binned indices → map back through X_z
        // Since we have X_binned and X_z, edges are X_z quantiles.
        // Simple approach: gather unique X_z values for this feature, sort, build edges.
        std::vector<double> vals(n);
        for (int i = 0; i < n; ++i) vals[i] = X_z(i, fc);
        std::sort(vals.begin(), vals.end());
        vals.erase(std::unique(vals.begin(), vals.end()), vals.end());

        // Build edges: min, then n_bins-1 quantiles, then max
        int nb = std::min(cfg.n_bins, static_cast<int>(vals.size()));
        Eigen::VectorXd edges(nb + 1);
        edges[0] = vals.front();
        for (int b = 1; b <= nb; ++b) {
            int pos = static_cast<int>(std::round(
                static_cast<double>(b) / nb * (static_cast<int>(vals.size()) - 1)));
            pos = std::clamp(pos, 0, static_cast<int>(vals.size()) - 1);
            edges[b] = vals[pos];
        }
        bin_edges[fi] = edges;
    }
    bin_edges_ = bin_edges;
    n_bins_cached_ = cfg.n_bins;

    // BFS queue: (node_index, sample_indices)
    struct QueueEntry {
        int node_idx;
        std::vector<int> indices;
        int depth;
    };

    std::vector<int> all_indices(n);
    std::iota(all_indices.begin(), all_indices.end(), 0);
    std::queue<QueueEntry> bfs;
    bfs.push({0, std::move(all_indices), 0});

    int leaf_count = 0;
    Eigen::VectorXi partition_ids(n);
    partition_ids.fill(-1);

    // Tracking for max gain normalization
    double total_gain = 0.0;
    std::vector<std::pair<int,double>> gain_log; // (feature, gain)

    while (!bfs.empty()) {
        auto [node_idx, indices, depth] = std::move(bfs.front());
        bfs.pop();

        int n_node = static_cast<int>(indices.size());
        TreeNode& node = nodes_[node_idx];

        bool can_split = (n_node >= 2 * msl) &&
                         (depth < cfg.max_depth) &&
                         (leaf_count + static_cast<int>(bfs.size()) + 1 < max_leaves);

        if (!can_split) {
            node.is_leaf      = true;
            node.partition_id = leaf_count++;
            for (int idx : indices) partition_ids[idx] = node.partition_id;
            continue;
        }

        // Evaluate both streams
        SplitResult cont_split = evaluate_continuous_splits(
            indices, X_binned, cont_cols, target, bin_edges, cfg.n_bins, msl, cfg);
        SplitResult bin_split = evaluate_binary_splits(
            indices, X_z, binary_cols, target, cfg);

        // Choose best
        SplitResult chosen;
        if (!cont_split.valid && !bin_split.valid) {
            node.is_leaf      = true;
            node.partition_id = leaf_count++;
            for (int idx : indices) partition_ids[idx] = node.partition_id;
            continue;
        } else if (!cont_split.valid) {
            chosen = bin_split;
        } else if (!bin_split.valid) {
            chosen = cont_split;
        } else {
            chosen = (bin_split.gain > cont_split.gain) ? bin_split : cont_split;
        }

        // Check min_samples_leaf on both sides
        std::vector<int> left_idx, right_idx;
        left_idx.reserve(n_node);
        right_idx.reserve(n_node);
        for (int idx : indices) {
            double val = X_z(idx, chosen.feature);
            if (val <= chosen.threshold) left_idx.push_back(idx);
            else                         right_idx.push_back(idx);
        }

        if (static_cast<int>(left_idx.size()) < msl ||
            static_cast<int>(right_idx.size()) < msl) {
            node.is_leaf      = true;
            node.partition_id = leaf_count++;
            for (int idx : indices) partition_ids[idx] = node.partition_id;
            continue;
        }

        // Commit split
        node.feature_idx = chosen.feature;
        node.threshold   = chosen.threshold;
        node.is_binary   = chosen.is_binary;

        feature_importances_[chosen.feature] += chosen.gain;
        total_gain += chosen.gain;

        int left_node  = static_cast<int>(nodes_.size());
        int right_node = left_node + 1;
        node.left  = left_node;
        node.right = right_node;
        nodes_.push_back(TreeNode{});
        nodes_.push_back(TreeNode{});

        bfs.push({left_node,  std::move(left_idx),  depth + 1});
        bfs.push({right_node, std::move(right_idx), depth + 1});
    }

    n_leaves_ = leaf_count;

    // Normalise feature importances
    if (total_gain > 0.0) {
        for (auto& fi : feature_importances_) fi /= total_gain;
    }

    fitted_ = true;
    return partition_ids;
}

// ── Apply ─────────────────────────────────────────────────────────────────────

Eigen::VectorXi PartitionTree::apply(const Eigen::MatrixXd& X_z) const {
    if (!fitted_) throw std::runtime_error("PartitionTree not fitted");
    const int n = static_cast<int>(X_z.rows());
    Eigen::VectorXi ids(n);

    for (int i = 0; i < n; ++i) {
        int node_idx = 0;
        while (!nodes_[node_idx].is_leaf) {
            const TreeNode& nd = nodes_[node_idx];
            double val = X_z(i, nd.feature_idx);
            node_idx = (val <= nd.threshold) ? nd.left : nd.right;
        }
        ids[i] = nodes_[node_idx].partition_id;
    }
    return ids;
}

} // namespace hvrt
