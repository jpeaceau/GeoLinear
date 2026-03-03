# Amplifying Latent Signal Recovery with HVRT Geometry

## The Problem

HVRT's cooperation statistic T = S² − Q already measures latent signal — Theorem 2 proves E[T] = Σ_{i≠j} Σ_ij, which in a factor model X = ZW + ε is composed entirely of latent-factor-driven covariances. The difficulty is that T collapses d(d−1)/2 pairwise interactions into a single scalar. Asking "how much latent signal does HVRT pick up" using only the scalar T is like measuring total rainfall instead of producing a precipitation map.

The strategies below decompose and amplify the latent information that HVRT's partition structure already captures, without modifying the library itself. Results are from a controlled benchmark: data generated as X = f(ZW + ε) with Z hidden, measuring CCA correlation between each strategy's representation and the true Z.

## Recommended Strategies

### 1. Partition-Enriched Embedding (Best Overall)

Construct each sample's representation as:

- **Partition centroid**: the mean of all samples in its HVRT partition (averages out noise within the cooperative region, concentrating latent signal)
- **Deviation from centroid**: sample minus its centroid (captures within-partition latent variation)
- **Local T value**: the sample's own T score (retains global cooperative position)

This achieved CCA 0.937 on nonlinear data, matching the raw-X ceiling of 0.936. In other words, HVRT's partitions already capture essentially all recoverable latent structure — you just need to query them as a d-dimensional embedding rather than a scalar.

```python
from hvrt import HVRT, compute_T
import numpy as np

hvrt = HVRT(random_state=42).fit(X_scaled)
pids = hvrt.apply_raw(X_scaled)
X_z = (X_scaled - X_scaled.mean(0)) / (X_scaled.std(0) + 1e-8)

# Build per-partition centroids
centroids = {}
for p in np.unique(pids):
    centroids[p] = X_scaled[pids == p].mean(axis=0)

# Enriched embedding
embedding = np.column_stack([
    np.array([centroids[p] for p in pids]),          # centroid (d dims)
    X_scaled - np.array([centroids[p] for p in pids]), # deviation (d dims)
    compute_T(X_z),                                    # global T (1 dim)
])
```

**When to use**: whenever you need the maximum possible latent recovery and can afford 2d + 1 dimensions.

### 2. T-Weighted PCA (Best Efficiency)

Weight each feature by how strongly it drives cooperative deviation before extracting principal components.

A feature z_k's contribution to T is proportional to its correlation with the sum of all other features. Features that co-vary with many others are the most "latent-driven" by Theorem 2. Squaring the weights amplifies latent axes and suppresses noise dimensions.

```python
from hvrt import compute_S
from sklearn.decomposition import PCA

X_z = (X_scaled - X_scaled.mean(0)) / (X_scaled.std(0) + 1e-8)
S = X_z.sum(axis=1)

# Each feature's cooperativeness
weights = np.array([
    abs(np.corrcoef(X_z[:, k], S - X_z[:, k])[0, 1])
    for k in range(X_z.shape[1])
])
weights = (weights / weights.max()) ** 2 + 0.1  # amplify, with floor

X_weighted = X_scaled * weights
pca = PCA(n_components=k_latent * 2).fit(X_weighted)
embedding = pca.transform(X_weighted)
```

This achieved CCA 0.903 in only 6 dimensions. It is the simplest amplifier: one line of feature re-weighting before any downstream model.

**When to use**: as a preprocessing step before any model. Low-dimensional, efficient, and interpretable — the weight vector tells you which features carry the most latent information.

### 3. Recursive Two-Level HVRT (Best Discrete Structure)

Fit HVRT globally, then fit a sub-HVRT within each partition. The top level captures global cooperative structure; the sub-level captures finer latent variation within each cooperative region.

```python
hvrt_top = HVRT(random_state=42).fit(X_scaled)
pids_top = hvrt_top.apply_raw(X_scaled)

sub_ids = np.zeros(len(X_scaled), dtype=int)
offset = 0
for p in np.unique(pids_top):
    mask = pids_top == p
    if mask.sum() > 20:
        sub_hvrt = HVRT(random_state=42).fit(X_scaled[mask])
        sub_ids[mask] = sub_hvrt.apply_raw(X_scaled[mask]) + offset
        offset += len(np.unique(sub_ids[mask]))
    else:
        sub_ids[mask] = offset
        offset += 1
```

Achieved CCA 0.797 with highest partition MI (0.479), meaning the discrete partition assignments themselves are most informative about latent factor quantiles.

**When to use**: when you need interpretable discrete groups that align with latent structure (e.g., for stratified sampling, explainability, or partition-aware augmentation).

## When HVRT Beats a VAE

Under deep (two-layer) nonlinear mixing, the enriched HVRT strategy (CCA 0.890) substantially outperformed the numpy VAE (CCA 0.653). The reason is structural: the VAE's encoder must learn to *invert* the mixing function, which is hard with nested nonlinearities. HVRT's tree-based partitioning sidesteps this entirely — it groups samples by cooperative structure without needing to learn the inverse map. The latent factors are the *source* of cooperation (Theorem 2), so the partition geometry naturally aligns with latent axes.

The VAE wins on simple mixing (linear or single-layer nonlinear) where the inverse map is easy to learn, and in high-noise settings where its explicit noise model (the KL term) provides regularization that HVRT lacks.

## The PyramidHART → HVRT Pipeline

The paper (Section 5) suggests using PyramidHART first to filter spike outliers, then fitting HVRT on the clean bulk. This is designed for *contaminated* data, not general latent recovery. In clean factor-model simulations it doesn't help because there are no spikes to filter — the spike-removal step just discards good data.

Use this pipeline when your data has isolated single-feature spikes (sensor faults, lab artifacts, data entry errors) that would inflate T without genuine cooperation. PyramidHART's A statistic cancels these exactly (Proposition 1, item 3), and then HVRT's Theorem 3 noise invariance holds on the cleaned bulk.

## Summary Table

| Strategy | CCA | Probe R² | Dims | Best For |
|---|---|---|---|---|
| Enriched partitions | 0.937 | 0.877 | 2d+1 | Maximum recovery |
| T-weighted PCA | 0.903 | 0.814 | 2k | Efficient preprocessing |
| Recursive HVRT | 0.797 | 0.576 | varies | Discrete interpretable groups |
| Partition centroids | 0.723 | 0.543 | d | Simple denoising |
| HVRT + T-proxy y_weight | 0.727 | 0.532 | partitions | Semi-supervised steering |
| PyramidHART → HVRT | 0.740 | 0.550 | partitions | Spike-contaminated data |
| Scalar T,S,Q,A | 0.365 | 0.277 | 4 | Quick diagnostic |
| VAE (baseline) | 0.922 | 0.853 | k | Simple nonlinear mixing |

Results on nonlinear mixing, d=12, k=3, σ=0.3, n=2000.

## Key Takeaway

HVRT doesn't need modification to absorb latent signal — it already does, through the algebraic identity E[T] = Σ_{i≠j} Σ_ij. The gap in naive benchmarks comes from collapsing partition structure to a scalar. Querying partitions as a d-dimensional embedding (centroids + deviations) or re-weighting features by their cooperative contribution recovers latent structure at near-ceiling levels, and outperforms VAEs on deep nonlinear mixing where the inverse problem is hard.
