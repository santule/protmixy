#!/usr/bin/env python3
"""
Main script for running protein evolution experiments.

This script serves as the entry point for generating evolutionary paths
between protein sequences using MSA-Transformer.

Usage:
    python scripts/run_evolution.py --start START_SEQ --end END_SEQ --seed SEED

Example:
    python scripts/run_evolution.py --start dummy1 --end dummy2 --seed 42
"""
import argparse
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.protein_evolver import iterative_sampling
from config.settings import (
    FULL_CONTEXT_FILE, 
    GENERATOR_OUTPUT_PATH,
    N_ITER,
    MAX_P_MASK,
    MIN_P_MASK,
    START_SEQ_NAME,
    END_SEQ_NAME
)


def main():
    """Main entry point for protein evolution experiments."""
    parser = argparse.ArgumentParser(
        description='Generate evolutionary paths between protein sequences using MSA-Transformer'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        default=START_SEQ_NAME,
        help=f'Starting sequence name (default: {START_SEQ_NAME})'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        default=END_SEQ_NAME,
        help=f'Ending sequence name (default: {END_SEQ_NAME})'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--n-iter',
        type=int,
        default=N_ITER,
        help=f'Maximum number of iterations (default: {N_ITER})'
    )
    
    parser.add_argument(
        '--max-mask',
        type=float,
        default=MAX_P_MASK,
        help=f'Maximum masking proportion (default: {MAX_P_MASK})'
    )
    
    parser.add_argument(
        '--min-mask',
        type=float,
        default=MIN_P_MASK,
        help=f'Minimum masking proportion (default: {MIN_P_MASK})'
    )
    
    parser.add_argument(
        '--context-msa',
        type=str,
        default=FULL_CONTEXT_FILE,
        help='Path to context MSA file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=GENERATOR_OUTPUT_PATH,
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Print experiment configuration
    print("\n" + "="*80)
    print("PROTEIN EVOLUTION EXPERIMENT")
    print("="*80)
    print(f"Starting sequence: {args.start}")
    print(f"Ending sequence: {args.end}")
    print(f"Random seed: {args.seed}")
    print(f"Max iterations: {args.n_iter}")
    print(f"Masking range: [{args.min_mask}, {args.max_mask}]")
    print(f"Context MSA: {args.context_msa}")
    print(f"Output directory: {args.output_dir}")
    print("="*80 + "\n")
    
    # Run evolution
    try:
        converged = iterative_sampling(
            starting_seq_name=args.start,
            ending_seq_name=args.end,
            context_msa_file=args.context_msa,
            random_seed=args.seed,
            n_iter=args.n_iter,
            max_mask=args.max_mask,
            min_mask=args.min_mask,
            output_file_path=args.output_dir
        )
        
        if converged:
            print("\n✅ Evolution completed successfully! Paths converged to target.")
        else:
            print("\n⚠️ Evolution completed but did not converge within iteration limit.")
            
    except Exception as e:
        print(f"\n❌ Error during evolution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
