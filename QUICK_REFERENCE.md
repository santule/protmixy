# Quick Reference Guide

## 📁 Project Structure

```
protmixy/
├── config/
│   ├── settings.py          # All hyperparameters and paths
│   └── log_config.py        # Logging configuration
├── src/
│   ├── model_loader.py      # MSA-Transformer model loader (singleton)
│   ├── protein_evolver.py   # Main evolution algorithm (iterative_sampling)
│   ├── evolution_utils.py   # Evolution helper functions
│   └── utils/
│       ├── evaluator.py     # EmbeddingEvaluator for scoring
│       ├── helpers.py       # General utilities (MSA, FASTA, etc.)
│       └── msa_output.py    # MSA-Transformer output processing
├── scripts/
│   └── run_evolution.py     # Main CLI entry point
├── docs/
│   └── methods.md           # Detailed algorithm documentation
└── data/                    # Your data goes here
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Settings
Edit `config/settings.py`:
```python
PROTEIN_FAMILY = 'lac'           # Your protein family
GENERATOR_METHOD = 'apc'         # 'apc' or 'irs'
ROOT_PATH = "/path/to/protmixy/" # Your project path
```

### 3. Run Evolution
```bash
python scripts/run_evolution.py \
    --start YOUR_START_SEQ \
    --end YOUR_END_SEQ \
    --seed 42
```

## 🔑 Key Functions

### Main Algorithm
```python
from src.protein_evolver import iterative_sampling

converged = iterative_sampling(
    starting_seq_name='dummy1',
    ending_seq_name='dummy2',
    context_msa_file='path/to/msa.aln',
    random_seed=42,
    n_iter=100,
    max_mask=0.05,
    min_mask=0.05,
    output_file_path='path/to/output/'
)
```

### Model Loading
```python
from src.model_loader import ModelLoader

model, alphabet = ModelLoader.get_model()
```

### Sequence Evaluation
```python
from src.utils.evaluator import EmbeddingEvaluator

score, aa_dist, aa_src, pos_dist, pos_diff, entropy, ll = \
    EmbeddingEvaluator.scorer_manifold(candidate_seq, msa_file)
```

### MSA Processing
```python
from src.utils import msa_output

query_emb, target_emb, query_pos, target_pos, entropy, ll = \
    msa_output.get_current_state_with_ll(candidate_seq, msa_file)
```

### Helper Functions
```python
from src.utils import helpers

# Check MSA statistics
total_seqs, seq_length = helpers.check_msa_stats('file.aln')

# Read MSA
msa_data = helpers.read_msa('file.aln', n_sequences)

# Prepend sequence to FASTA
helpers.prepend_sequence_to_fasta(seq, name, input_file, output_file)
```

## 📊 Output Files

Evolution generates:
- `beam_evol_msat_history_{seed}.pkl` - Complete path history
- `beam_evol_msat_intermediate_seqs_{seed}.fasta` - All accepted sequences
- `beam_evol_path_{idx}_{seed}.fasta` - Individual converged paths

## ⚙️ Key Parameters

### Beam Search
- `N_BEAM = 3` - Number of parallel beams
- `N_TOSS = 3` - Sampling attempts per beam
- `N_CANDIDATES = 5` - Top-k candidates per toss

### Masking
- `MAX_P_MASK = 0.05` - Maximum masking proportion
- `MIN_P_MASK = 0.05` - Minimum masking proportion
- `MASK_CYCLE = 1` - Masking cycle length

### Annealing
- `ANNEAL_TEMP = 1.0` - Initial temperature
- `TEMP_DECAY = 0.99` - Temperature decay rate
- `ANNEAL_TEMP_MIN = 0.0001` - Minimum temperature

### Convergence
- `N_ITER = 100` - Maximum iterations
- `STOP_TOL_FACTOR = 0.25` - Stopping tolerance (fraction of initial distance)

## 🔧 Common Tasks

### Change Evolution Method
```python
# In config/settings.py
GENERATOR_METHOD = 'apc'  # or 'irs'
```

### Adjust Beam Size
```python
# In config/settings.py
N_BEAM = 5  # Increase for more exploration
```

### Change Model
```python
# In config/settings.py
GENERATOR_MODEL_NAME = "esm_msa1_t12_100M_UR50S"
ESM2_MODEL_NAME = 'esm2_t33_650M_UR50D'
```

### Update Data Paths
```python
# In config/settings.py
ROOT_PATH = "/your/path/"
PROTEIN_FAMILY = 'your_family'
```

## 🐛 Troubleshooting

### Import Errors
```bash
# Make sure you're in project root
cd /Users/Shared/projects/protmixy
python scripts/run_evolution.py --help
```

### CUDA Out of Memory
```python
# Reduce beam parameters in config/settings.py
N_BEAM = 2
N_TOSS = 2
N_CANDIDATES = 3
```

### Path Issues
```python
# Use absolute paths in config/settings.py
ROOT_PATH = os.path.abspath("/Users/Shared/projects/protmixy/")
```

## 📚 Documentation

- **README.md** - Full project documentation
- **docs/methods.md** - Algorithm details
- **MIGRATION_GUIDE.md** - Understanding the restructuring
- **SETUP_COMPLETE.md** - Setup instructions

## 🔗 Import Cheatsheet

```python
# Configuration
from config.settings import *
from config.log_config import logger

# Core modules
from src.model_loader import ModelLoader
from src.protein_evolver import iterative_sampling
from src.evolution_utils import *

# Utilities
from src.utils import helpers
from src.utils import msa_output
from src.utils.evaluator import EmbeddingEvaluator
```

## 📦 Git Commands

```bash
# Check status
git status

# View history
git log --oneline

# Push to remote (when ready)
git remote add origin <url>
git push -u origin main
```

---

**For detailed information, see README.md and docs/methods.md**
