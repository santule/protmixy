# Methods Documentation

## Overview

This document provides detailed information about the algorithms and methods implemented in ProtMixy for generating evolutionary paths between protein sequences.

## Core Algorithm: Beam Search with Simulated Annealing

### Initialization

1. Load starting sequence `s_start` and target sequence `s_target`
2. Compute initial distance `d_0` between sequences using ESM2 embeddings
3. Set stopping tolerance: `τ = STOP_TOL_FACTOR × d_0`
4. Initialize beam with starting sequence

### Main Loop

For each iteration `t = 1, ..., N_ITER`:

1. **For each beam state** (default: 3 beams):
   
   a. **Mask Generation**:
      - Compute positions to mask based on method (IRS or APC)
      - Number of masked positions varies cyclically
   
   b. **Candidate Sampling** (N_TOSS × N_CANDIDATES):
      - Mask selected positions in current sequence
      - Feed masked sequence + MSA context to MSA-Transformer
      - Sample top-k candidates from output distribution
      - Generate additional argmax candidate
   
   c. **Candidate Evaluation**:
      - Compute cosine distance to target using ESM2
      - Calculate amino acid differences
      - Compute position-wise entropy
      - Check convergence: if `d < τ`, mark as converged
   
   d. **Acceptance (Simulated Annealing)**:
      - For each candidate with score `s_cand`:
      - Compute `Δs = s_cand - s_current`
      - Accept with probability:
        - `P = 1` if `Δs ≤ 0` (improvement)
        - `P = exp(-10Δs / T)` if `Δs > 0` (worse)
      - Temperature: `T = T_0 × (decay)^t`

2. **Beam Selection**:
   - Pool all accepted candidates from all beams
   - Select top `N_BEAM` candidates by log-likelihood
   - These form the beam for next iteration

3. **Convergence Check**:
   - If any candidate converged, continue but track it
   - If no candidates remain, terminate

### Output

- Path history with all candidates and their relationships
- Converged paths traced back to starting sequence
- Intermediate sequences for analysis

## Position Sampling Methods

### IRS (Iterative Refinement Sampling)

**Objective**: Sample positions independently based on distance to target

**Algorithm**:

1. Compute position-wise cosine distances `d_i` to target
2. Filter positions:
   - Only positions differing from target
   - Only positions with entropy > threshold percentile
3. Convert distances to probabilities:
   ```
   p_i ∝ exp(d_i / τ_dist)
   ```
4. Sample positions without replacement

**Advantages**:
- Simple and interpretable
- Focuses on problematic positions
- Fast computation

### APC (Attention-based Positional Coupling)

**Objective**: Sample spatially coupled positions using attention

**Algorithm**:

1. Extract row attention from MSA-Transformer layer 12
2. Average over heads and layers: `A ∈ ℝ^(L×L)`
3. Apply Average Product Correction (APC):
   ```
   A_ij^APC = A_ij - (Σ_k A_ik)(Σ_k A_kj) / (Σ_k,l A_kl)
   ```
4. Initialize with seed position sampled by distance
5. Breadth-first search using attention weights:
   - For current position, find top-10 attention neighbors
   - Add neighbors that differ from target
   - Continue until desired number of positions

**Advantages**:
- Captures co-evolutionary relationships
- Samples spatially proximate positions
- Biologically motivated

**APC Correction**: Removes background correlation, similar to direct coupling analysis in contact prediction

## Evaluation Metrics

### Cosine Distance

For sequences `s_1` and `s_2`:

1. Compute ESM2 embeddings: `e_1, e_2 ∈ ℝ^(L×d)`
2. Position-wise cosine distance:
   ```
   d_i = 1 - (e_1[i] · e_2[i]) / (||e_1[i]|| ||e_2[i]||)
   ```
3. Overall distance: `d = mean(d_i)`

### Entropy

Position-wise entropy from MSA:
```
H_i = -Σ_a p(a|i) log p(a|i)
```

High entropy indicates variable positions in the MSA.

### Log-Likelihood

Position-wise log-likelihood under MSA-Transformer:
```
LL = Σ_i log P(s_i | MSA, s_{<i})
```

Used for ranking candidates within an iteration.

## Hyperparameters

### Critical Parameters

- **N_BEAM** (default: 3): Number of parallel search paths
- **N_TOSS** (default: 3): Sampling attempts per beam
- **N_CANDIDATES** (default: 5): Top-k candidates per toss
- **STOP_TOL_FACTOR** (default: 0.25): Convergence threshold as fraction of initial distance

### Annealing Schedule

- **ANNEAL_TEMP** (default: 1.0): Initial temperature
- **TEMP_DECAY** (default: 0.99): Exponential decay rate
- **ANNEAL_TEMP_MIN** (default: 0.0001): Minimum temperature

### Masking Schedule

- **MAX_P_MASK** (default: 0.05): Maximum masking proportion
- **MIN_P_MASK** (default: 0.05): Minimum masking proportion
- **MASK_CYCLE** (default: 1): Cycle length for masking schedule

### Filtering

- **ENTROPY_THRESHOLD_FILTER** (default: 30): Percentile threshold for entropy
- **DISTANCE_TEMP** (default: 0.1): Temperature for distance-based sampling

## Computational Complexity

### Per Iteration

- **MSA-Transformer forward pass**: O(L² × MSA_size)
  - L: sequence length
  - Typically ~1-2 seconds on GPU for L=200, MSA_size=100

- **ESM2 embedding**: O(L²)
  - For evaluation of each candidate
  - ~0.5 seconds per sequence

### Total Runtime

For typical parameters:
- N_ITER = 100
- N_BEAM = 3, N_TOSS = 3, N_CANDIDATES = 5
- Total candidates per iteration: ~45

Expected runtime: 2-4 hours on GPU

## Model Details

### MSA-Transformer

- **Model**: esm_msa1_t12_100M_UR50S
- **Parameters**: 100M
- **Input**: MSA with query sequence + context
- **Output**: Per-position amino acid distributions

### ESM2

- **Model**: esm2_t33_650M_UR50D (default)
- **Parameters**: 650M
- **Input**: Single sequence
- **Output**: Per-position embeddings (dim=1280)

## References

1. Rao et al. (2021). "MSA Transformer." ICML.
2. Lin et al. (2022). "Language models of protein sequences at the scale of evolution enable accurate structure prediction." bioRxiv.
3. Dunn et al. (2008). "Mutual information without the influence of phylogeny or entropy dramatically improves residue contact prediction." Bioinformatics.
