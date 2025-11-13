"""
Configuration file for stochastic protein evolution using MSA-Transformer.

This module contains all hyperparameters and settings for running protein
evolution experiments using iterative refinement sampling (IRS) and 
attention-based positional coupling (APC) methods.
"""
import torch
import types
import os

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

# Protein family for evolution experiments
PROTEIN_FAMILY = 'lac'  # Options: kari, lac, pla2g2, klk, 3ftx

# Evolution method selection
GENERATOR_METHOD = 'irs'  # Options: 'irs' (Iterative Refinement Sampling), 
                          #          'apc' (Attention-based Positional Coupling)
CONTEXT_METHOD   = 'beam'   # Context method: 'beam', '1s_100t', '1s_int_100t', '100s_1t'

# Sequence identifiers
START_SEQ_NAME   = 'dummy1'
END_SEQ_NAME     = 'dummy2'

# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Root path - adjust based on your environment
ROOT_PATH = f"/Users/Shared/projects/protmixi/"

# Data paths
MAIN_DATA_PATH = f"{ROOT_PATH}/data/{PROTEIN_FAMILY}/"
INPUT_FILE_PATH = f"{MAIN_DATA_PATH}input_data/"
OUTPUT_FILE_PATH = f"{MAIN_DATA_PATH}output_data/"
PAIR_OUTPUT_FILE_PATH = f"{OUTPUT_FILE_PATH}{START_SEQ_NAME}_{END_SEQ_NAME}/{CONTEXT_METHOD}/"
GENERATOR_OUTPUT_PATH = f"{PAIR_OUTPUT_FILE_PATH}{GENERATOR_METHOD}/"
RESULT_OUTPUT_PATH = f"{ROOT_PATH}results/"
PAPER_PLOTS_PATH = f"{ROOT_PATH}plots_for_paper/"

# Input files
MSA_CONTEXT_FILE = f'{INPUT_FILE_PATH}{PROTEIN_FAMILY}_extants.aln'
FULL_CONTEXT_FILE = f'{INPUT_FILE_PATH}{PROTEIN_FAMILY}_extants.fasta'

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_FILE_PATH):
    os.makedirs(OUTPUT_FILE_PATH, exist_ok=True)

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# MSA-Transformer model for sequence generation
GENERATOR_MODEL_NAME = "esm_msa1_t12_100M_UR50S"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# PATHWAY GENERATION HYPERPARAMETERS
# =============================================================================

# Simulated annealing parameters
ANNEAL_TEMP = 1.0
TEMP_DECAY = 0.99
ANNEAL_TEMP_MIN = 0.0001 * ANNEAL_TEMP

# Masking parameters
MASK_ID = 32  # Token ID for masked positions
MAX_P_MASK = 0.05  # Maximum proportion of sequence to mask
MIN_P_MASK = 0.05  # Minimum proportion of sequence to mask
MASK_CYCLE = 1  # Cyclic linear schedule for masking
DISTANCE_TEMP = 0.1  # Temperature for probabilistic sampling (sharper distribution)
ENTROPY_THRESHOLD_FILTER = 30  # Percentile threshold for entropy filtering

# Beam search parameters
N_ITER = 100  # Maximum number of iterations
STOP_TOL_FACTOR = 0.25  # Stopping tolerance as fraction of initial distance
N_BEAM = 3  # Number of beams to maintain
N_TOSS = 3  # Number of sampling attempts per beam
N_CANDIDATES = 5  # Number of top-k candidates to generate per toss

# =============================================================================
# CONFIGURATION SUMMARY
# =============================================================================

def print_config():
    """Print all configuration parameters."""
    config_vars = {
        key: value
        for key, value in globals().items()
        if not key.startswith("__") and not callable(value) and not isinstance(value, types.ModuleType)
    }
    
    exclude_keys = {"CONFIGS", "config_vars"}
    
    print("\n" + "="*80)
    print("PATHWAY GENERATION CONFIGURATION")
    print("="*80)
    for key, value in config_vars.items():
        if key not in exclude_keys:
            print(f"{key}: {value}")
    print("="*80 + "\n")

# Auto-print configuration when module is imported
if __name__ != "__main__":
    print_config()
