# ProtMixy: Generating Hybrid Proteins with the MSA-Transformer

<p align="center">
  <img src="hybridproteins1.png" alt="ProtMixy diagram" width="700">
</p>


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
The pathway generation would use all the parameters set in the config/settings.py file. 

##### Running the script
```bash
python generate_pathway.py
```

##### Output Files
The pathway generation produces several output files:

1. **`beam_evol_msat_history_{seed}.pkl`**: Complete pathway history with all candidates and metadata
2. **`beam_evol_msat_intermediate_seqs_{seed}.fasta`**: All accepted hybrid intermediate sequences
3. **`beam_evol_path_{idx}_{seed}.fasta`**: Individual mutational pathways (one per converged beam)


#### PART 2 - SCORING INTERMEDIATE GENERATED SEQUENCES
The second part is to score the intermediate sequences generated in the first part. This predicts structure using ESMFold and then uses TM-score to calculate the structure similarity. It also uses sequence identity to calculate the sequence similarity. Both information is used to calculate the weighted hybrid score. 

##### Running the script
```bash
python score_hybrids.py
```

##### Output Files
The pathway scoring produces several output files:

1. **`hybrid_scores_{seed}.csv`**: Hybrid scores for all intermediate sequences
2. **`hybrid_scores_{seed}.png`**: Hybrid score scatter plot

#### PART 3 - PROFILE INTERMEDIATE GENERATED SEQUENCES
The third part is to profile the intermediate sequences generated in the first part. This uses Sparse AutoEncoder (SAE) InterProt to calculate the profile of the intermediate sequences.

##### Running the script
```bash
python sae_profile_hybrids.py
```

##### Output Files
The pathway profiling produces several output files in the folder `GENERATOR_OUTPUT_PATH`:

1. **`hybrid_sae_profile_boxplot_{seed}.png`**: Hybrid SAE profile boxplot
2. **`hybrid_sae_profile_scatterplot_{seed}.png`**: Hybrid SAE profile scatterplot


## Configuration
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
- **FULL_MSA_FILE**: Path to the full MSA for the protein family
- **MSA_CONTEXT_FILE**: Path to the conditioning context file to generate mutational pathway
- **ROOT_PATH**: Root path for data and output files
- **MAIN_DATA_PATH**: Main data directory
- **INPUT_FILE_PATH**: Input file directory
- **OUTPUT_FILE_PATH**: Output file directory
- **GENERATOR_OUTPUT_PATH**: Output file directory for pathway generation
- **DENSE_SEED_THRESHOLD**: Threshold for average cosine distance to filter dense clusters of sequences for creating conditioning context

## Running for your own source and target protein sequences
For running on your protein family with different source and target protein, we recommend the following steps:

STEP 1 : Create conditioning context which provides a MSA of sequences that are similar to the target sequence. You need to provide a full MSA of the protein family in the `FULL_MSA_FILE`.

Edit `config/settings.py`:
- Set `FULL_MSA_FILE` to path of the full MSA for the protein family.

Run the below script. The script uses KNN algorithm on MSA-Transformer embeddings to find dense clusters of sequences and select potential sequences chosen from the FULL_MSA_FILE for conditioning context. The cluster aggregated cosine distance is filtered based on the settings parameter `DENSE_SEED_THRESHOLD` and anything above this threshold is filtered out. Different sizes of K can be tried out to find the best conditioning context. If the MSA is large, it is recommended to use 48 / 80 GB of GPU memory.

```bash
python create_conditioning_context.py
```

##### Output Files

All files are created in the `INPUT_FILE_PATH`
1. **`conditioning_context_{nearest_label}.aln`**: Five different samples of conditioning context MSA (you can choose to create more if needed)
2. **`full_context_embeddings.npz`**: Full MSA-Transformer embeddings for the `FULL_MSA_FILE` data
3. **`potential_cond_context.json`**: Potential conditioning contexts with random source sequences at different identity to target sequence

You can review the potention conditioning context in the `conditioning_context_{nearest_label}.aln` and the diffferent source sequences in the `potential_cond_context.json` file. The conditioning context can be manually curated or adjusted based on domain knowledge or any other sequence annotations available.

STEP 2: Setup the configuration and folder structure so we can now run the pathway generation.

Edit `config/settings.py`:
- Set `START_SEQ_NAME` and `END_SEQ_NAME` from any of the suggested pairs in the `potential_cond_context.json` file.
- Create Folder `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}`
- Copy the `conditioning_context_{nearest_label}.aln` to the `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}` folder. Rename it to `cond_context.aln`
- Copy the `full_context.aln` to the `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}` folder.
- Ensure following files exist before running the code:
  - `MSA_CONTEXT_FILE`  (default: `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}/full_context.aln`)
  - `FULL_CONTEXT_FILE` (default: `data/output_data/{START_SEQ_NAME}_{END_SEQ_NAME}/cond_context.aln`)

Now you are ready to run the pathway generation.

```bash
python generate_pathway.py
```

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

### PART 3 - PROFILE INTERMEDIATE GENERATED SEQUENCES

1. **SAE Profile**: SAE profile of each intermediate sequence
Out of top 500 activating latent for source and target sequence, find the top 5 common feature (present in both source and target), top 5 unique feature -source_only (present in source but not in target) and top 5 unique feature -target_only (present in target but not in source) and calculate the percentage change to the source sequence.
2. **SAE Profile Boxplot**: Boxplot of SAE profile of each intermediate sequence with respect to source sequence for all 15 features (5 common, 5 source_only, 5 target_only)
3. **SAE Profile Scatterplot**: Scatterplot of SAE profile of each intermediate sequence with respect to source sequence for mean source_only and target_only features


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
