# ProtMixy: Generating Hybrid Proteins with the MSA-Transformer

A computational framework for generating mutational pathways between protein sequences, enabling the design of hybrid proteins through MSA-Transformer-guided sequence space exploration.

## Overview

ProtMixy generates step-by-step mutational pathways that connect two protein sequences, producing hybrid intermediates that is informed my the model. The framework leverages MSA-Transformer's understanding of sequence context and co-evolutionary patterns to guide mutations.

### Two Pathway Generation Methods

1. **IRS (Independent Residue Sampling)**: Position-independent sampling based on embedding based cosine distance
   - Samples positions with high cosine distance to target
   - Prioritizes high-entropy positions for mutation

2. **APC (Attention-based Positional Coupling)**: Co-evolutionary position sampling
   - Uses MSA-Transformer row attention to identify coupled positions
   - Samples spatially related positions together
   - Applies Average Product Correction (APC) to attention matrices

Both methods employ beam search with simulated annealing to maintain multiple pathway candidates and ensure smooth transitions through sequence space.

## Key Features

- **Hybrid Protein Design**: Generate intermediate sequences between two homologous proteins
- **Beam Search**: Explores multiple pathway candidates simultaneously (default: 3 beams)
- **Simulated Annealing**: Temperature-based acceptance for smooth sequence transitions
- **Co-evolutionary Awareness**: APC based masking leverages MSA-Transformer row attention
- **Convergence Tracking**: Monitors cosine distance and stops when 75% of total distance between source and target is reached
- **Path Validation**: Comprehensive consistency checks on generated pathways

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 24GB+ RAM recommended

### Setup

```bash
# Clone the repository
git clone https://github.com/santule/protmixy.git
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
├── data/                # Data directory (MSA files, sequences)
├── docs/                # Documentation
├── generate_pathway.py  # Main entry point script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Usage

### Basic Usage

```bash
python generate_pathway.py
```

All configuration is controlled via `config/settings.py`. Before running:

- Set `START_SEQ_NAME` and `END_SEQ_NAME` to sequence IDs present in `FULL_CONTEXT_FILE`.
- Ensure your input files exist:
  - `MSA_CONTEXT_FILE`  (default: `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}/full_context.aln`)
  - `FULL_CONTEXT_FILE` (default: `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}/cond_context.aln`)
- Adjust paths if needed:
  - `ROOT_PATH`, `MAIN_DATA_PATH`, `INPUT_FILE_PATH`, `OUTPUT_FILE_PATH`
- Adjust pathway generation hyperparameters, if needed:
  - `GENERATOR_METHOD`, `N_ITER`, `P_MASK`, `N_BEAM`, `N_TOSS`, `N_CANDIDATES`, etc.

## Configuration

Edit `config/settings.py` to customize parameters:

- **Pathway method**: Choose `GENERATOR_METHOD` ('irs' or 'apc')
- **Beam parameters**: Adjust `N_BEAM`, `N_TOSS`, `N_CANDIDATES` to control how many pathway hypotheses are tracked, how aggressively low-scoring beams are pruned, and how many new candidates are sampled at each step. Higher values explore more but increase compute.
- **Annealing schedule**: Modify `ANNEAL_TEMP`, `TEMP_DECAY` to tune how accepting the search is to worse moves early on and how quickly it becomes greedy. Slower decay (higher `TEMP_DECAY`) allows more exploration but may require more iterations to converge.

Edit `config/settings.py` to run for different source and target protein sequences:
- Set `START_SEQ_NAME` and `END_SEQ_NAME` to sequence IDs present in `FULL_CONTEXT_FILE`.
- Create Folder `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}`
- Ensure your input files exist:
  - `MSA_CONTEXT_FILE`  (default: `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}/full_context.aln`)
  - `FULL_CONTEXT_FILE` (default: `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}/cond_context.aln`)

## Output Files

The pathway generation produces several output files:

1. **`beam_evol_msat_history_{seed}.pkl`**: Complete pathway history with all candidates and metadata
2. **`beam_evol_msat_intermediate_seqs_{seed}.fasta`**: All accepted hybrid intermediate sequences
3. **`beam_evol_path_{idx}_{seed}.fasta`**: Individual mutational pathways (one per converged beam)


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
- Reduce MSA size
- APC can be memory-intensive, consider using IRS instead

**No Convergence**
- Check the MSA conditioning context used for generating pathway
