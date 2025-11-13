# ProtMixy: Protein Evolution via MSA-Transformer

A computational framework for generating evolutionary paths between protein sequences using MSA-Transformer with beam search and simulated annealing.

## Overview

ProtMixy implements two novel methods for guided protein sequence evolution:

1. **IRS (Iterative Refinement Sampling)**: Uses cosine distance-based position sampling to iteratively refine sequences toward a target
2. **APC (Attention-based Positional Coupling)**: Leverages attention weights from MSA-Transformer to sample coupled positions, capturing co-evolutionary relationships

Both methods use beam search with simulated annealing to explore the sequence space between starting and ending protein sequences while maintaining biological plausibility through MSA context.

## Key Features

- **Beam Search**: Maintains multiple candidate paths simultaneously (default: 3 beams)
- **Simulated Annealing**: Temperature-based acceptance criterion for exploring sequence space
- **Context-Aware**: Uses MSA context to ensure biologically plausible mutations
- **Convergence Tracking**: Monitors distance to target and stops when threshold is reached
- **Path Validation**: Comprehensive consistency checks on generated evolutionary paths

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
│   ├── protein_evolver.py    # Main evolution algorithm
│   ├── evolution_utils.py    # Helper functions
│   └── utils/                # Utility modules
│       ├── model_loader.py   # MSA-Transformer model loader
│       ├── evaluator.py      # Sequence evaluation
│       ├── helpers.py        # General utilities
│       └── msa_output.py     # MSA processing
├── scripts/             # Executable scripts
│   └── generate_pathway.py # Main entry point
├── data/                # Data directory (create subdirectories as needed)
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
- **Evolution method**: Choose `GENERATOR_METHOD` ('irs' or 'apc')
- **Beam parameters**: Adjust `N_BEAM`, `N_TOSS`, `N_CANDIDATES`
- **Annealing schedule**: Modify `ANNEAL_TEMP`, `TEMP_DECAY`
- **Model selection**: Change `GENERATOR_MODEL_NAME` or `ESM2_MODEL_NAME`

## Output Files

The algorithm generates several output files in the specified output directory:

1. **`beam_evol_msat_history_{seed}.pkl`**: Complete path history with all candidates
2. **`beam_evol_msat_intermediate_seqs_{seed}.fasta`**: All accepted intermediate sequences
3. **`beam_evol_path_{idx}_{seed}.fasta`**: Individual evolutionary paths (one per converged candidate)

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
2. **Iteration Loop**:
   - For each beam:
     - Generate mask based on distance/attention
     - Sample multiple candidates from MSA-Transformer
     - Evaluate candidates using ESM2 embeddings
     - Apply simulated annealing acceptance criterion
   - Select top candidates based on log-likelihood
   - Check convergence (distance to target < threshold)
3. **Path Assembly**: Trace back converged candidates to starting sequence

## Dependencies

### Core Libraries

- **PyTorch**: Deep learning framework
- **fair-esm**: ESM models (MSA-Transformer, ESM2)
- **NumPy/SciPy**: Numerical computations
- **BioPython**: Sequence manipulation
- **pysam**: FASTA file handling

### Optional

- **wandb**: Experiment tracking
- **matplotlib/seaborn**: Visualization

## Citation

If you use this code in your research, please cite:

```bibtex
@article{protmixy2024,
  title={Guided Protein Evolution using MSA-Transformer with Attention-based Positional Coupling},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes with clear commit messages
4. Submit a pull request

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

**Import Errors**
- Ensure all dependencies are installed
- Check Python path includes project root
- Verify ESM installation: `python -c "import esm"`

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].

## Acknowledgments

- ESM models from Meta AI Research
- MSA-Transformer architecture
- [Add other acknowledgments]
