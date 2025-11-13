# ProtMixy: Mutational Pathway via MSA-Transformer to Generate Hybrid Protein Sequences

A computational framework for generating biologically plausible mutational pathways between protein sequences, enabling the design of hybrid proteins through MSA-Transformer-guided sequence space exploration.

## Overview

ProtMixy generates step-by-step mutational pathways that connect two protein sequences, producing hybrid intermediates that maintain biological plausibility. The framework leverages MSA-Transformer's understanding of sequence context and co-evolutionary patterns to guide mutations.

### Two Pathway Generation Methods

1. **IRS (Iterative Refinement Sampling)**: Position-independent sampling based on embedding distance
   - Samples positions with high cosine distance to target
   - Prioritizes high-entropy positions for mutation
   - Efficient for sequences with distributed differences

2. **APC (Attention-based Positional Coupling)**: Co-evolutionary position sampling
   - Uses MSA-Transformer attention weights to identify coupled positions
   - Samples spatially related positions together
   - Captures epistatic interactions and structural constraints
   - Applies Average Product Correction (APC) to attention matrices

Both methods employ beam search with simulated annealing to maintain multiple pathway candidates and ensure smooth transitions through sequence space.

## Key Features

- **Hybrid Protein Design**: Generate intermediate sequences between any two proteins
- **Biologically Plausible Pathways**: MSA context ensures mutations respect natural variation
- **Beam Search**: Explores multiple pathway candidates simultaneously (default: 3 beams)
- **Simulated Annealing**: Temperature-based acceptance for smooth sequence transitions
- **Co-evolutionary Awareness**: APC method captures epistatic interactions
- **Convergence Tracking**: Monitors embedding distance and stops when target is reached
- **Path Validation**: Comprehensive consistency checks on generated pathways

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 16GB+ RAM recommended

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd protmixy

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
protmixy/
├── config/              # Configuration files
│   └── settings.py      # Hyperparameters and paths
├── src/                 # Core source code
│   ├── msat_beam_evolver.py  # Main pathway generation algorithm
│   └── utils/                # Utility modules
│       ├── model_loader.py   # MSA-Transformer model loader
│       ├── evaluator.py      # Sequence evaluation
│       ├── helpers.py        # Pathway generation utilities
│       └── msa_output.py     # MSA processing
├── scripts/             # Executable scripts
│   └── generate_pathway.py  # Main entry point
├── data/                # Data directory (MSA files, sequences)
├── docs/                # Documentation
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Usage

### Basic Usage

```bash
python scripts/generate_pathway.py --start dummy1 --end dummy2 --seed 42
```

### Advanced Options

```bash
python scripts/generate_pathway.py \
    --start START_SEQ_NAME \
    --end END_SEQ_NAME \
    --seed 42 \
    --n-iter 100 \
    --max-mask 0.05 \
    --min-mask 0.05 \
    --context-msa /path/to/msa.aln \
    --output-dir /path/to/output/
```

### Parameters

- `--start`: Starting sequence name (must exist in MSA file)
- `--end`: Target sequence name (must exist in MSA file)
- `--seed`: Random seed for reproducibility
- `--n-iter`: Maximum number of iterations (default: 100)
- `--max-mask`: Maximum proportion of sequence to mask (default: 0.05)
- `--min-mask`: Minimum proportion of sequence to mask (default: 0.05)
- `--context-msa`: Path to MSA alignment file
- `--output-dir`: Directory for output files

## Configuration

Edit `config/settings.py` to customize:

- **Protein family**: Set `PROTEIN_FAMILY` (e.g., 'lac', 'kari', 'pla2g2')
- **Pathway method**: Choose `GENERATOR_METHOD` ('irs' or 'apc')
- **Beam parameters**: Adjust `N_BEAM`, `N_TOSS`, `N_CANDIDATES`
- **Annealing schedule**: Modify `ANNEAL_TEMP`, `TEMP_DECAY`
- **Model selection**: Change `GENERATOR_MODEL_NAME`

## Output Files

The pathway generation produces several output files:

1. **`beam_evol_msat_history_{seed}.pkl`**: Complete pathway history with all candidates and metadata
2. **`beam_evol_msat_intermediate_seqs_{seed}.fasta`**: All accepted hybrid intermediate sequences
3. **`beam_evol_path_{idx}_{seed}.fasta`**: Individual mutational pathways (one per converged beam)

## Methods

### IRS (Iterative Refinement Sampling)

- Samples positions based on cosine distance to target
- Focuses on positions with high entropy and large distance
- Independent position sampling

### APC (Attention-based Positional Coupling)

- Uses attention weights from MSA-Transformer
- Samples spatially coupled positions
- Captures co-evolutionary relationships
- Applies Average Product Correction (APC) to attention matrices

## Algorithm Overview

1. **Initialization**: Start with source sequence in MSA context
2. **Iterative Pathway Generation**:
   - For each beam candidate:
     - Generate position mask (IRS: distance-based, APC: attention-based)
     - Sample multiple hybrid candidates from MSA-Transformer
     - Evaluate candidates using embedding distance to target
     - Apply simulated annealing acceptance criterion
   - Select top candidates based on log-likelihood
   - Check convergence (embedding distance to target < threshold)
3. **Pathway Assembly**: Trace back converged beams to construct complete mutational pathways

## Dependencies

### Core Libraries

- **PyTorch**: Deep learning framework
- **fair-esm**: MSA-Transformer model
- **NumPy/SciPy**: Numerical computations and distance metrics
- **BioPython**: Sequence manipulation
- **pysam**: FASTA/MSA file handling
- **h5py**: Embedding storage

## Troubleshooting

### Common Issues

**Out of Memory (OOM)**
- Reduce `N_BEAM`, `N_TOSS`, or `N_CANDIDATES`
- Use smaller ESM2 model (e.g., 'esm2_t6_8M_UR50D')
- Reduce MSA size

**No Convergence**
- Increase `N_ITER`
- Adjust `STOP_TOL_FACTOR`
- Check that start and end sequences are in MSA
