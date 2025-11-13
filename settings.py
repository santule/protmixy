''' config file for stochastic protein evolution using MSA-TRANSFORMER '''
import torch, types
import os

# SEQUENCE for which to generate evolution
PROTEIN_FAMILY = 'lac' # kari, lac, pla2g2, klk, 3ftx

# gpu machine , server, local
#ROOT_PATH  = f"/media/WorkingSpace/Share/protmixi/"
#ROOT_PATH  = f"/protmixi/"
ROOT_PATH = f"/Users/Shared/projects/protmixi/"

# Levers of generation - context experiments
#GENERATOR_METHOD = 'irs'     # 'irs' , 'apc'
#CONTEXT_METHOD   = '1s_100t' # '1s_100t','1s_int_100t','100s_1t'

# Levers of generation - beam experiments
GENERATOR_METHOD = 'apc'  # 'irs' , 'apc'
CONTEXT_METHOD   = 'beam'    # 'beam'

# Sequence names
START_SEQ_NAME   = 'dummy1'
END_SEQ_NAME     = 'dummy2'

# File paths for context experiments
#MAIN_DATA_PATH         = f"{ROOT_PATH}/context_exps_data/{PROTEIN_FAMILY}/"

# Other File paths
MAIN_DATA_PATH         = f"{ROOT_PATH}/data/{PROTEIN_FAMILY}/"
INPUT_FILE_PATH        = f"{MAIN_DATA_PATH}input_data/"
OUTPUT_FILE_PATH       = f"{MAIN_DATA_PATH}output_data/"
PAIR_OUTPUT_FILE_PATH  = f"{OUTPUT_FILE_PATH}{START_SEQ_NAME}_{END_SEQ_NAME}/{CONTEXT_METHOD}/"
GENERATOR_OUTPUT_PATH  = f"{PAIR_OUTPUT_FILE_PATH}{GENERATOR_METHOD}/"
RESULT_OUTPUT_PATH     = f"{ROOT_PATH}results/"
PAPER_PLOTS_PATH       = f"{ROOT_PATH}plots_for_paper/"

# GITHUB INSTALLATION PATH
#SYSTEM_PATH = "/Users/Shared/projects/protogenix/"
SYSTEM_PATH = "/"
GITHUB_DOWNLOAD_PATH = f"{SYSTEM_PATH}protmixi/git_downloads/"

# PROTEIN FAMILY FILES
FULL_CONTEXT_FILE  = f'{INPUT_FILE_PATH}{PROTEIN_FAMILY}_extants.aln'  # only extant sequences

# Annealing
ANNEAL_TEMP = 1
TEMP_DECAY  = 0.99
ANNEAL_TEMP_MIN = 0.0001 * ANNEAL_TEMP

# MASKING
MASK_ID  = 32
MAX_P_MASK, MIN_P_MASK   = 0.05, 0.05
MASK_CYCLE    = 1  # cyclic linear schedule for masking
DISTANCE_TEMP = 0.1 # for probabilistic sampling and makes the distribution sharper
ENTROPY_THRESHOLD_FILTER = 30

# ITERATIONS - MAIN BEAM ITERATIONS AND REFINEMENT ITERATIONS
N_ITER          = 100 
STOP_TOL_FACTOR = 0.25
N_BEAM = 3
N_TOSS = 3
N_CANDIDATES = 5

# wandb
RUN_NAME = f'{PROTEIN_FAMILY}_{GENERATOR_METHOD}_{CONTEXT_METHOD}'

# Make the output folder if does not exist
if not os.path.exists(OUTPUT_FILE_PATH):
    os.makedirs(OUTPUT_FILE_PATH, exist_ok=True)

# GENERATOR MODEL
GENERATOR_MODEL_NAME = "esm_msa1_t12_100M_UR50S"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ANALYSIS AND PLOT MODEL
esm2_tokens_per_batch   = 4096
esm2_model_name = 'esm2_t33_650M_UR50D' #'esm2_t6_8M_UR50D' # 'esm2_t33_650M_UR50D'
esm2_truncation_seq_length = 1022
esm2_output_layer = 33 #6 # 6 # 6, 33 # last layer

# Method colors
METHOD_COLORS = {
    'asr': '#1f77b4',     # Blue
    'irs': '#800080',     # Purple
    'apc': '#FFD700',     # Yellow
    'random': '#ff7f0e',  # Orange
    'context': '#2ca02c'  # Green
}

# Color palette as a list for seaborn/matplotlib
METHOD_COLOR_PALETTE = [METHOD_COLORS['asr'], METHOD_COLORS['irs'], 
                        METHOD_COLORS['apc'], METHOD_COLORS['random']]

# Print all parameters, excluding modules
config_vars = {
    key: value
    for key, value in globals().items()
    if not key.startswith("__") and not callable(value) and not isinstance(value, types.ModuleType)
}

# List of variables you want to exclude from printing (e.g., full config dicts)
EXCLUDE_KEYS = {"CONFIGS", "config_vars"}

# Print only relevant parameters
for key, value in config_vars.items():
    if key not in EXCLUDE_KEYS:
        print(f"{key}: {value}")