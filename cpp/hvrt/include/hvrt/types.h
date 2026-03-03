#pragma once
#include <string>
#include <vector>

namespace hvrt {

// ── Enums ────────────────────────────────────────────────────────────────────

enum class SplitStrategy {
    Best,
    Random
};

enum class ReductionMethod {
    CentroidFPS,
    MedoidFPS,
    VarianceOrdered,
    Stratified
};

enum class GenerationStrategy {
    Epanechnikov,
    MultivariateKDE,
    BootstrapNoise,
    UnivariateCopula,
    Auto
};

enum class GeometryMode   { T, A, S };             // pairwise / ℓ₁-cooperation / sum
enum class WhiteningMode  { Variance, MAD };        // mean±std / median±MAD
enum class SplitCriterion { Variance, AbsoluteError }; // split gain formula

// ── Configuration ─────────────────────────────────────────────────────────────

struct HVRTConfig {
    int    n_partitions    = 50;
    int    min_samples_leaf = 20;
    int    max_depth       = 10;
    int    n_bins          = 32;
    int    n_threads       = 4;
    int    random_state    = 42;
    float  y_weight        = 0.0f;
    bool   auto_tune       = true;
    std::string bandwidth  = "auto";  // "auto", "scott", or numeric string
    GeometryMode   geometry_mode   = GeometryMode::T;
    WhiteningMode  whitening_mode  = WhiteningMode::Variance;
    SplitCriterion split_criterion = SplitCriterion::Variance;
};

// ── Output structs ────────────────────────────────────────────────────────────

struct PartitionInfo {
    int    id;
    int    size;
    double mean_abs_z;
    double variance;
};

struct ParamRecommendation {
    int n_partitions;
    int min_samples_leaf;
};

} // namespace hvrt
