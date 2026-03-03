#pragma once
#include <string>

namespace geolinear {

// ── GeoLinear configuration ───────────────────────────────────────────────────
// Boosted ensemble of partition-local Ridge models fitted within HVRT
// cooperative geometry partitions.

struct GeoLinearConfig {
    // Boosting
    int    n_rounds                = 20;
    double learning_rate           = 0.1;

    // HVRT partitioner
    double y_weight                = 0.5;
    int    hvrt_min_samples_leaf   = -1;  // -1 = HVRT auto-tune
    int    hvrt_n_partitions       = -1;  // -1 = HVRT auto-tune

    // Per-partition linear learner
    std::string base_learner       = "ridge"; // "ridge", "ols", "lasso"
    double alpha                   = 1.0;  // L2 penalty (ridge/ols) or L1 penalty (lasso)
    int    min_samples_partition   = 5;    // partitions below this → predict 0

    // Linear boosting within each cross-product partition.
    // 1 = single model (current behaviour). M > 1 = M rounds of base_learner
    // against successive inner residuals. Predictions are accumulated into a
    // single effective model (coef_eff = Σ coef_m, intercept_eff = Σ intercept_m).
    int    partition_inner_rounds  = 1;

    // HVRT inner T-residual boosting: number of HVRT trees per outer stage.
    // 1 = single tree (current behaviour). K > 1 = K trees built sequentially:
    //   tree_1 fits T, tree_2 fits T - leaf_means(tree_1), etc.
    //   Final partition = cross-product of all K trees' leaf assignments.
    int    hvrt_inner_rounds       = 1;

    // HVRT refit (fast tree rebuild, skips whitening/binning/geometry)
    // 0 = disabled (new HVRT per round). k > 0 = refit every k rounds.
    int    refit_interval          = 0;

    // HVRT model variant: "hvrt" (default) | "hart" | "fast_hvrt" | "fast_hart" | "pyramid_hart"
    // Controls geometry_mode, whitening_mode, and split_criterion of the HVRT partitioner.
    std::string hvrt_model         = "hvrt";

    // Misc
    int    random_state            = 42;
};

} // namespace geolinear
