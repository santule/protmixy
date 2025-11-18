#!/usr/bin/env python3
"""
Main script for running mutational pathway generation experiments.

This script serves as the entry point for generating mutational pathways
between protein sequences using MSA-Transformer.

Usage:
    python scripts/generate_pathway.py --start START_SEQ --end END_SEQ --seed SEED

Example:
    python scripts/generate_pathway.py --start dummy1 --end dummy2 --seed 42
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.msat_beam_evolver import iterative_sampling
from config.settings import (
    FULL_CONTEXT_FILE,
    MSA_CONTEXT_FILE, 
    GENERATOR_OUTPUT_PATH,
    N_ITER,
    P_MASK,
    START_SEQ_NAME,
    END_SEQ_NAME
)


def main():
    """Main entry point for mutational pathway generation experiments."""

    # All configuration is taken from config/settings.py
    full_context_file = FULL_CONTEXT_FILE
    starting_seq_name = START_SEQ_NAME
    ending_seq_name = END_SEQ_NAME
    context_msa_file = MSA_CONTEXT_FILE
    output_dir = GENERATOR_OUTPUT_PATH
    random_seed = 10
    n_iter = N_ITER
    p_mask = P_MASK

    # Print experiment configuration
    print("\n" + "="*80)
    print("MUTATIONAL PATHWAY GENERATION.")
    print("="*80)
    print(f"Starting sequence: {starting_seq_name}")
    print(f"Ending sequence: {ending_seq_name}")
    print(f"Random seed: {random_seed}")     
    print(f"Max iterations: {n_iter}")
    print(f"Masking proportion: {p_mask}")
    print(f"Context MSA: {context_msa_file}")
    print(f"Output directory: {output_dir}")
    print("="*80 + "\n")

    # Run evolution
    try:
        converged = iterative_sampling(
            full_context_file=full_context_file,
            starting_seq_name=starting_seq_name,
            ending_seq_name=ending_seq_name,
            context_msa_file=context_msa_file,
            random_seed=random_seed,
            n_iter=n_iter,
            p_mask=p_mask,
            output_file_path=output_dir
        )
        
        if converged:
            print("\n✅ Pathway generation completed successfully! Paths converged to target.")
        else:
            print("\n⚠️ Pathway generation completed but did not converge within iteration limit.")
            
    except Exception as e:
        print(f"\n❌ Error during pathway generation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
