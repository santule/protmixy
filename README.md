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
- **Hybrid Scoring**: Calculate hybrid score for each intermediate sequence

## Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU
- 24GB+ RAM

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

## Usage

### Basic Usage

#### PART 1 - GENERATING PATHWAYS

##### Running the script
```bash
python generate_pathway.py
```

##### Output Files
The pathway generation produces several output files:

1. **`beam_evol_msat_history_{seed}.pkl`**: Complete pathway history with all candidates and metadata
2. **`beam_evol_msat_intermediate_seqs_{seed}.fasta`**: All accepted hybrid intermediate sequences
3. **`beam_evol_path_{idx}_{seed}.fasta`**: Individual mutational pathways (one per converged beam)


#### PART 2 - SCORING PATHWAYS

##### Running the script
```bash
python score_hybrids.py
```

##### Output Files
The pathway scoring produces several output files:

1. **`hybrid_scores_{seed}.csv`**: Hybrid scores for all intermediate sequences
2. **`hybrid_scores_{seed}.png`**: Hybrid score scatter plot

##### Configuration
All configuration is controlled via `config/settings.py`.

- **GENERATOR_METHOD**: 'irs' or 'apc'
- **N_ITER**: Maximum number of iterations
- **P_MASK**: Proportion of sequence to mask
- **DISTANCE_TEMP**: Temperature for probabilistic sampling (sharper distribution)
- **ENTROPY_THRESHOLD_FILTER**: Percentile threshold for entropy filtering
- **N_BEAM**: Number of beams to maintain
- **N_TOSS**: Number of sampling attempts per beam
- **N_CANDIDATES**: Number of top-k candidates to generate per toss
- **ANNEAL_TEMP**: Initial temperature for simulated annealing
- **TEMP_DECAY**: Temperature decay factor
- **ANNEAL_TEMP_MIN**: Minimum temperature for simulated annealing
- **STOP_TOL_FACTOR**: Stopping tolerance as fraction of initial distance
- **START_SEQ_NAME**: Sequence ID of the source sequence
- **END_SEQ_NAME**: Sequence ID of the target sequence
- **FULL_CONTEXT_FILE**: Path to the full MSA for the protein family
- **MSA_CONTEXT_FILE**: Path to the conditioning context file to generate mutational pathway
- **ROOT_PATH**: Root path for data and output files
- **MAIN_DATA_PATH**: Main data directory
- **INPUT_FILE_PATH**: Input file directory
- **OUTPUT_FILE_PATH**: Output file directory

#### To run for different source and target protein sequences

Edit `config/settings.py`:
- Set `START_SEQ_NAME` and `END_SEQ_NAME` to sequence IDs present in `FULL_CONTEXT_FILE`.
- Create Folder `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}`
- Ensure your input files exist:
  - `MSA_CONTEXT_FILE`  (default: `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}/full_context.aln`)
  - `FULL_CONTEXT_FILE` (default: `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}/cond_context.aln`)

## Algorithm Overview

### PART 1 - GENERATING PATHWAYS

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

### PART 2 - SCORING INTERMEDIATE GENERATED SEQUENCES

1. **Sequence Similarity**: Sequence Identity between each intermediate sequence to the start and end sequences
2. **Structure Similarity**: TM-score between each intermediate structure to the start and end structures
3. **Hybrid Score Calculation**: Calculate hybrid score for each intermediate sequence


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
