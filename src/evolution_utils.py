import torch
import logging
import heapq

logger = logging.getLogger(__name__)
import numpy as np
from numpy.random import rand
from src.utils import helpers
from src.utils.evaluator import EmbeddingEvaluator
import tempfile
import math
from src.model_loader import ModelLoader
from config.settings import DEVICE, MASK_ID, DISTANCE_TEMP,\
     MASK_CYCLE, ENTROPY_THRESHOLD_FILTER, GENERATOR_METHOD, N_CANDIDATES
import random
import os

# load the generator + alphabet
msa_transformer, msa_alphabet = ModelLoader.get_model()
msa_batch_converter = msa_alphabet.get_batch_converter()
idx_list = msa_alphabet.tok_to_idx # reference dict for model token
aa_list  = {v: k for k,v in idx_list.items()}

valid_aa_vals = torch.tensor([ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
                             18, 19, 20, 21, 22, 23, 30],dtype=torch.int64)

class MSAContextMismatchError(Exception):
    pass

def get_row_col_attention(msa_batch_tokens):
    with torch.no_grad():
        results = msa_transformer(msa_batch_tokens.to(DEVICE),repr_layers=[12],return_contacts=False, need_head_weights=True)
        row_attn = results['row_attentions'][0].to('cpu').numpy() # 12,12,734,734
        avg_row_attn = np.mean(row_attn, axis=(0,1)) # 734,734
    return avg_row_attn 

def apply_apc(attn_matrix):
    """
    Apply Average Product Correction (APC) to a square attention matrix.
    
    Parameters:
        attn_matrix (np.ndarray): Attention matrix of shape (L, L)
        
    Returns:
        apc_corrected (np.ndarray): APC-corrected matrix of shape (L, L)
    """
    row_means = np.mean(attn_matrix, axis=1, keepdims=True)
    col_means = np.mean(attn_matrix, axis=0, keepdims=True)
    global_mean = np.mean(attn_matrix)

    expected = row_means @ col_means / global_mean
    apc_corrected = attn_matrix - expected
    return apc_corrected

def top_k_samples(prob_matrix, k):

    # prob_matrix: shape (n_positions, vocab_size)
    beam = [([], 1.0)]  # list of (sequence_so_far, prob)
  
    for pos in range(1, prob_matrix.shape[0]):
        new_beam = []

        for seq, seq_prob in beam:
            for val in range(prob_matrix.shape[1]):
                new_seq = seq + [val]
                new_prob = seq_prob * prob_matrix[pos, val]
                new_beam.append((new_seq, new_prob))
        
        # Keep top-k sequences with highest joint prob
        beam = heapq.nlargest(k, new_beam, key=lambda x: x[1])

    # only want sequence tokens
    beam = [seq for seq, _ in beam]

    # Add cls token
    beam = [np.array([0] + seq) for seq in beam]
    
    return beam

def msa_query_sample_manifold(msa_batch_tokens):
    """
    Sample a sequence from the MSA-Transformer model using the provided batch tokens.
    
    Args:
        msa_batch_tokens (torch.Tensor): Batch of tokenized MSA sequences
        
    Returns:
        list of torch.Tensor: Sampled sequence tokens from the model
    """
    #print(f"🧬 Sampling candidates from MSA-Transformer...")
    
    with torch.no_grad():
        results_logits  = msa_transformer(msa_batch_tokens.to(DEVICE),repr_layers=[12],return_contacts=False)['logits'] # 1, total_seqs, 734, 33
        #print(f"🧬 Model logits shape: {results_logits.shape}")
        
        hybrid_logits   = torch.softmax(results_logits[:,0,:,:],axis=2) # query sequence only
        argmax_token    = torch.argmax(hybrid_logits, dim=2).reshape(-1).to('cpu').numpy()     
        argmax_token[0] = 0 # cls token
        #print(f"🧬 Argmax token sequence length: {len(argmax_token)}")

        hybrid_probs = hybrid_logits.squeeze(0).cpu().numpy()
        top_k_seq_tokens = top_k_samples(hybrid_probs, N_CANDIDATES)
        #print(f"🧬 Generated {len(top_k_seq_tokens)} top-k candidates")
        
        # debug - Verify top-k by calculating log probabilities for each candidate
        # print(f"🧬 Verifying top-k candidates:")
        # candidate_log_probs = []
        # for i, seq in enumerate(top_k_seq_tokens):
        #     log_prob = 0.0
        #     for pos, token_id in enumerate(seq):
        #         if pos < len(hybrid_probs):
        #             log_prob += np.log(hybrid_probs[pos, token_id] + 1e-10)
        #     candidate_log_probs.append((i, log_prob))
        #     print(f"  Candidate {i}: log_prob = {log_prob:.4f}")
        
        # # Sort by log probability to verify ranking
        # sorted_candidates = sorted(candidate_log_probs, key=lambda x: x[1], reverse=True)
        # print(f"🧬 Candidates ranked by log probability: {[f'C{idx}({prob:.4f})' for idx, prob in sorted_candidates]}")

        # add hybrid token to the top K token
        all_candidates = top_k_seq_tokens + [argmax_token]
        #print(f"🧬 Total candidates before deduplication: {len(all_candidates)}")
        
        unique_sequences_set = {tuple(seq) for seq in all_candidates}
        final_candidates_np = [np.array(seq) for seq in unique_sequences_set]
        #print(f"🧬 Final unique candidates after deduplication: {len(final_candidates_np)}")

    del results_logits

    return [torch.from_numpy(seq) for seq in final_candidates_np]
    
def decode_token_to_aa(seq_tokens):
    """
    Convert token IDs to amino acid sequences.
    
    Args:
        seq_tokens (torch.Tensor): Sequence tokens
        
    Returns:
        str: Decoded amino acid sequence
    """
    aa_generated_lst = []
    for tk in seq_tokens.reshape(-1)[1:]: # remove cls token
        aa_generated_lst.append(aa_list[int(tk)])
    aa_generated_seq = "".join(aa_generated_lst)
    return aa_generated_seq

def create_msa_for_iterative_sampling(context_msa_file, starting_seq):
    """
    Create an MSA file for iterative sampling by prepending a starting sequence.
    
    Args:
        context_msa_file (str): Path to the context MSA file
        starting_seq (str): Starting sequence to prepend
        
    Returns:
        tuple: (msa_data, total_seqs, size_align_seq) where:
            - msa_data (list): List of MSA sequences
            - total_seqs (int): Total number of sequences in the MSA
            - size_align_seq (int): Alignment sequence length
    """
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as temp_file:
        start_msa_file_name = temp_file.name
        helpers.prepend_sequence_to_fasta(starting_seq, 'candidate_sequence', context_msa_file, start_msa_file_name)
    total_seqs, size_align_seq = helpers.check_msa_stats(start_msa_file_name)
    print(f"MSA Stats - total sequences is {total_seqs} and each sequence length is {size_align_seq}")
    msa_data = [helpers.read_msa(start_msa_file_name, total_seqs)]
    print(f"READ MSA file {start_msa_file_name} with {total_seqs} sequences")
    return msa_data, total_seqs, size_align_seq

def safe_exp(x):
    """
    Safely compute exponential function, handling potential overflow errors.
    
    Args:
        x (float): Input value for exponential function
        
    Returns:
        float: Result of math.exp(x) or 0 if overflow occurs
    """
    try:
        return math.exp(x)
    except OverflowError:
        print(f"Overflow in exponential calculation with input {x}")
        return 0

def mask_sequence(curr_sequence_token_temp, mask):
    """
    Apply a mask to the current sequence tokens, replacing masked positions with MASK_ID.
    
    Args:
        curr_sequence_token_temp (torch.Tensor): Current sequence tokens
        mask (torch.Tensor): Binary mask where 0 indicates positions to mask
        
    Returns:
        tuple: (masked_seq_token, masked_positions) where:
            - masked_seq_token (torch.Tensor): Sequence with masked positions
            - masked_positions (torch.Tensor): Indices of masked positions
    """
    masked_seq_token = curr_sequence_token_temp * mask + (1 - mask) * MASK_ID
    masked_positions = masked_seq_token == MASK_ID
    masked_positions = masked_positions.nonzero(as_tuple=True)[0]
    #print(f"Masked positions: {masked_positions}")
    return masked_seq_token, masked_positions

def check_msa_context(initial_msa_batch_tokens, msa_batch_tokens):
    """
    Check if the MSA context has changed between iterations.
    
    Args:
        initial_msa_batch_tokens (torch.Tensor): Initial MSA batch tokens
        msa_batch_tokens (torch.Tensor): Current MSA batch tokens
        
    Raises:
        MSAContextMismatchError: If the MSA context has changed
    """
    diff_context_msa_batch_tokens = initial_msa_batch_tokens[0, 1:, :] - msa_batch_tokens[0, 1:, :]
    if torch.any(diff_context_msa_batch_tokens != 0):
        raise MSAContextMismatchError("Mismatch in the MSA context detected!")

def eval_candidate_manifold(candidate, src, tgt, context_msa_file):
    """
    Evaluate a candidate sequence by comparing it to source and target sequences.
    
    Args:
        candidate (str): The candidate sequence to evaluate
        src (str): The source sequence
        tgt (str): The target sequence
        context_msa_file (str): Path to the MSA context file
        
    Returns:
        tuple: A tuple containing:
            - current_score (float): Overall score of the candidate
            - d_aa_tgt (int): Number of amino acid differences from target
            - d_aa_src (int): Number of amino acid differences from source
            - pos_d_cos_tgt (numpy.ndarray): Position-wise cosine distances to target
            - tgt_pos_diff_aa (list): Positions where candidate differs from target
            - src_pos_entropy (numpy.ndarray): Position-wise entropy values
            - src_ll (numpy.ndarray): Position-wise log likelihood values
    """
    
    # AA changes wrt source and target
    src_diff_aa  = []
    tgt_diff_aa  = []
    tgt_pos_diff_aa = []
    for pos, (ele_cand, ele_src, ele_tgt) in enumerate(zip(candidate, src, tgt)):
        if ele_cand != ele_src:
            src_diff_aa.append(pos)
        if ele_cand != ele_tgt:
            tgt_diff_aa.append(pos)
            tgt_pos_diff_aa.append(pos+1) # 0 is cls token
    d_aa_tgt = len(tgt_diff_aa)
    d_aa_src = len(src_diff_aa)

    # plm distance wrt source and target
    d_cos_tgt, pos_d_cos_tgt, src_pos_entropy, src_ll = EmbeddingEvaluator.scorer_manifold(candidate, context_msa_file)
    return d_cos_tgt, d_aa_tgt, d_aa_src, pos_d_cos_tgt, tgt_pos_diff_aa, src_pos_entropy, src_ll

def tokens_changed(curr_sequence_token, cand_sequence_token):
    """
    Analyze changes between current and candidate sequence tokens.
    
    Args:
        curr_sequence_token (torch.Tensor): Current sequence tokens
        cand_sequence_token (torch.Tensor): Candidate sequence tokens
        pos_masked (torch.Tensor or list): Positions that were masked during sampling
        curr_probs (numpy.ndarray): Current position-wise probability distributions
        cand_probs (numpy.ndarray): Candidate position-wise probability distributions
        
    Returns:
        tuple: A tuple containing:
            - actual_pos_changed_list (list): Positions where tokens changed
            - num_pos_changed (int): Number of positions where tokens changed
    """
    # Find positions where tokens changed
    actual_pos_changed = curr_sequence_token != cand_sequence_token
    actual_pos_changed = actual_pos_changed.nonzero(as_tuple=True)[0]    
    # Convert the tensor to a list for easier handling and serialization
    actual_pos_changed_list = actual_pos_changed.cpu().numpy().tolist()
    
    return actual_pos_changed_list, len(actual_pos_changed)

def distance_to_probabilities(distances, entropy, tgt_pos_diff):
    """
    Convert position-wise distances to sampling probabilities for mask generation.
    
    This function transforms cosine distances into a probability distribution for sampling
    positions to mask. It focuses on positions that differ from the target sequence and
    have high entropy (variability).
    
    Args:
        distances (numpy.ndarray): Position-wise cosine distances to target
        entropy (numpy.ndarray): Position-wise entropy values
        tgt_pos_diff (list): Positions where sequence differs from target
        
    Returns:
        numpy.ndarray: Probability distribution for sampling positions to mask
    """
    # Create a mask where positions different from target are True
    diff_mask = np.zeros_like(distances, dtype=bool)
    diff_mask[tgt_pos_diff] = True
        
    # Zero out positions that are not different from target
    distances = distances.copy()  # Create a copy to avoid modifying the original
    distances[~diff_mask] = 0
    
    # mask for not considering positions with low entropy and have the same AA as the target
    nonzero_mask = distances != 0
    
    entropy_threshold = np.percentile(entropy, ENTROPY_THRESHOLD_FILTER) 
    #print(f"Entropy threshold: {entropy_threshold}")
    entropy_mask = entropy > entropy_threshold
    combined_mask = nonzero_mask & entropy_mask

    # cosine distances
    masked_distances = distances[combined_mask]
    masked_distances = masked_distances / DISTANCE_TEMP
    masked_distances = masked_distances - np.max(masked_distances)
    exp_distances = np.exp(masked_distances)
    norm_dist_probs = exp_distances / np.sum(exp_distances)

    # Create a full-size probability array and insert the computed values
    probs = np.zeros_like(distances, dtype=float)
    probs[combined_mask] = norm_dist_probs
    if probs.sum() > 0:
        probs /= probs.sum()

    return probs

def calc_mask_size(size_align_seq, iteration_no, max_mask, min_mask):
    """
    Calculate the number of positions to mask based on the current iteration.
    
    This function implements a cyclic masking strategy where the number of masked positions
    varies throughout the evolution process according to a cycle defined by MASK_CYCLE.
    
    Args:
        size_align_seq (int): Size of the alignment sequence
        iteration_no (int): Current iteration number
        
    Returns:
        int: Number of positions to mask in this iteration
    """
    cycle_progress = (iteration_no % MASK_CYCLE) / MASK_CYCLE
        
    max_positions = int(max_mask * size_align_seq)
    min_positions = int(min_mask * size_align_seq)
    num_pos_mask  = max(min_positions, int(max_positions - (cycle_progress * (max_positions - min_positions))))

    return num_pos_mask

def get_sampled_positions_cpos(current_pos_wise_dist_to_tgt, num_pos_mask, current_pos_entropy, current_tgt_pos_diff):
    """
    Sample positions to mask based on their distance to target and entropy.
    
    Args:
        current_pos_wise_dist_to_tgt (numpy.ndarray): Position-wise cosine distances to target
        num_pos_mask (int): Number of positions to mask
        current_pos_entropy (numpy.ndarray): Position-wise entropy values
        current_tgt_pos_diff (list): Positions where sequence differs from target
        
    Returns:
        numpy.ndarray: Sampled positions to mask
    """
    logger.info(f"Number of positions to mask: {num_pos_mask}")
    # Convert distances to probabilities for sampling
    pos_sample_probabilities = distance_to_probabilities(current_pos_wise_dist_to_tgt, current_pos_entropy, current_tgt_pos_diff)
        
    # Get indices of non-zero probabilities
    nonzero_indices = np.flatnonzero(pos_sample_probabilities)
    logger.info(f"Number of positions with non-zero probability: {len(nonzero_indices)}")
        
    # Adjust number of positions to mask if needed
    num_pos_mask = min(num_pos_mask, len(nonzero_indices))
    logger.info(f"Adjusted number of positions to mask: {num_pos_mask}")
        
    # Sample positions based on probabilities
    all_positions = np.random.choice(
            len(current_pos_wise_dist_to_tgt), 
            size=num_pos_mask, 
            replace=False, 
            p=pos_sample_probabilities
    )
    
    # remove positions
    return all_positions

def get_sampled_positions_ra(current_pos_wise_dist_to_tgt, num_pos_mask, current_pos_entropy, current_tgt_pos_diff, avg_row_attn):
    """
    Sample positions to mask based on their distance to target and entropy.
    
    Args:
        current_pos_wise_dist_to_tgt (numpy.ndarray): Position-wise cosine distances to target
        num_pos_mask (int): Number of positions to mask
        current_pos_entropy (numpy.ndarray): Position-wise entropy values
        current_tgt_pos_diff (list): Positions where sequence differs from target
        
    Returns:
        numpy.ndarray: Sampled positions to mask
    """
    #print(f"Number of positions to mask: {num_pos_mask}")
    # Convert distances to probabilities for sampling
    pos_sample_probabilities = distance_to_probabilities(current_pos_wise_dist_to_tgt, current_pos_entropy, current_tgt_pos_diff)
        
    # Get indices of non-zero probabilities
    nonzero_indices = np.flatnonzero(pos_sample_probabilities)
    #print(f"Number of positions with non-zero probability: {len(nonzero_indices)}")
        
    # Adjust number of positions to mask if needed
    num_pos_mask = min(num_pos_mask, len(nonzero_indices))
    #print(f"Adjusted number of positions to mask: {num_pos_mask}")
        
    masked_positions = set()
    queue = []
    current_tgt_pos_diff = set(current_tgt_pos_diff)
    filled = False
    
    while len(masked_positions) < num_pos_mask:
        while not queue:
            # Sample new seed if queue is empty
            seed_pos = np.random.choice(
                len(current_pos_wise_dist_to_tgt),
                size=1,
                replace=False,
                p=pos_sample_probabilities
            )[0]

            if seed_pos not in masked_positions:
                masked_positions.add(seed_pos)
                queue.append(seed_pos)

        current = queue.pop(0)

        # Get top-k attention neighbors
        top_attn_indices = np.argsort(-avg_row_attn[current])[:10]

        for idx in top_attn_indices:
            if idx in current_tgt_pos_diff and idx not in masked_positions and idx != 0:
                masked_positions.add(idx)
                if len(masked_positions) >= num_pos_mask:
                    filled = True
                    break
                queue.append(idx)

        if filled:
            break
        # if len(masked_positions) >= num_pos_mask:
        #     break

    all_positions = sorted(masked_positions)
    #print(f"Masked positions: {all_positions}")
    return all_positions

def generate_manifold_mask(iteration_no, size_align_seq, current_pos_wise_dist_to_tgt, current_pos_entropy, current_tgt_pos_diff, row_attention, max_mask, min_mask):
    """
    Generate a mask for positions to be sampled based on their distance to target and entropy.
    
    Args:
        iteration_no (int): Current iteration number
        size_align_seq (int): Alignment sequence length
        current_pos_wise_dist_to_tgt (numpy.ndarray): Position-wise cosine distances to target
        current_pos_entropy (numpy.ndarray): Position-wise entropy values
        current_tgt_pos_diff (list): Positions where sequence differs from target
        
    Returns:
        numpy.ndarray: Sampled positions to mask
    """
    num_pos_mask = calc_mask_size(size_align_seq, iteration_no, max_mask, min_mask)
    if iteration_no == 1:
        num_pos_mask = 0
        all_positions = np.array([])
    else:
        if GENERATOR_METHOD == 'apc':
            all_positions = get_sampled_positions_ra(current_pos_wise_dist_to_tgt, num_pos_mask, current_pos_entropy, current_tgt_pos_diff,row_attention)
        elif GENERATOR_METHOD == 'irs':
            all_positions = get_sampled_positions_cpos(current_pos_wise_dist_to_tgt, num_pos_mask, current_pos_entropy, current_tgt_pos_diff)

    mask = torch.ones(size_align_seq + 1, dtype=torch.uint8)
    mask[all_positions] = 0
    mask[0] = 1 # dont mask bos token
    return mask, len(all_positions)

def accept_or_reject_iteration_candidates(this_iteration_candidates, N_BEAM):

    # Sort candidates by log-likelihood (higher is better)
    sorted_candidates = sorted(this_iteration_candidates, key=lambda x: x['llm_ll'], reverse=True)
    
    # debugging purposes
    # print(f"📊 Sorting {len(sorted_candidates)} candidates by log-likelihood:")
    # for i, cand in enumerate(sorted_candidates):
    #     status = "✅ SELECTED" if i < N_BEAM else "❌ REJECTED"
    #     print(f"  {status} Rank {i+1}: {cand['id']} (llm_ll={cand['llm_ll']:.4f})")
    
    
    return sorted_candidates[:N_BEAM]

def accept_or_reject_beam_candidates(beam_candidates, current_state, t, it):
    """
    Selects the top candidates for the next beam based on their scores and a temperature-based criterion.
    There would be total of N_TOSS * N_CANDIDATES candidates.

    Args:
        beam_candidates (list): A list of candidate state dictionaries.
        current_state (dict): The current state dictionary.
        t (float): The current temperature for the Metropolis criterion.
        it (int): The current iteration number.

    Returns:
        list: A list of the selected candidate states for the next beam.
    """
    if not beam_candidates:
        return []

    # if iteration is 1 we will accept all candidates as the optimiser is settling in
    if it == 1:
        return beam_candidates

    # Sort candidates by score (lower is better)
    sorted_candidates = sorted(beam_candidates, key=lambda x: x['score'])

    accepted_candidates = []
    random_toss = random.random()
    
    for cand in sorted_candidates:
        logger.info(f" 📊 Candidate {cand['id']} has score {cand['score']} and delta score {cand['score'] - current_state['score']}")
        delta_score = cand['score'] - current_state['score']
        if delta_score <= 0:
            acceptance_probability = 1.0
        else:
            # Scale the delta to prevent overflow and have a reasonable acceptance range
            scaled_delta = 10.0 * delta_score
            acceptance_probability = safe_exp((-1 * scaled_delta) / t)

        if random_toss < acceptance_probability:
            accepted_candidates.append(cand)
            logger.info(f"Accepted candidate {cand['id']} with score {cand['score']} (p={acceptance_probability:.4f})")
        else:
            logger.info(f"Rejected candidate {cand['id']} with score {cand['score']} (p={acceptance_probability:.4f})")
            pass

    # If no candidates were accepted, return the current state to prevent the beam from dying.
    if not accepted_candidates:
        logger.info(f"No new candidates accepted for this beam. Repeating current state.")
        return [current_state]

    return accepted_candidates

def write_evolution_fasta(output_file_path, path_history, starting_seq_name, starting_seq, ending_seq_name, ending_seq, random_seed):
    """
    Write the evolutionary path to a FASTA file, including start sequence, accepted sequences, and end sequence.
    
    Args:
        path_history (dict): Dictionary containing the evolutionary path history
        starting_seq_name (str): Name of the starting sequence
        starting_seq (str): Starting sequence
        ending_seq_name (str): Name of the ending sequence
        ending_seq (str): Ending sequence
        random_seed (int): Random seed used for the evolutionary process
        
    Returns:
        str: Path to the created FASTA file
    """
    # Create evolution FASTA file
    evol_fasta_file = os.path.join(output_file_path, f"evol_msat_{random_seed}.fasta")
    
    # Write all sequences at once
    with open(evol_fasta_file, "w") as f:
        # Write the starting sequence
        f.write(f">{starting_seq_name}\n{starting_seq}\n")
        
        # Write all accepted sequences from the path history
        for iter_num, path in path_history.items():
            if path['status'] == 'ACCEPT':
                f.write(f">{path['name']}\n{path['sequence']}\n")
        
        # Write the ending sequence
        f.write(f">{ending_seq_name}\n{ending_seq}\n")

    return evol_fasta_file

def assemble_paths(path_history, output_file_path, random_seed):
    """ Assemble the evolutionary paths into a single FASTA file."""

    # Create a lookup dictionary for quick access by id
    path_lookup = {path['id']: path for path in path_history}

    # Find all converged candidates (these are the endpoints of successful paths)
    converged_candidates = [path for path in path_history if path['converged'] == True and path['status'] != 'END']
    print(f"🔍 Found {len(converged_candidates)} converged candidates to trace back")

    # Trace back each converged candidate to its starting point
    complete_paths = []
    
    for idx, endpoint in enumerate(converged_candidates):
        # Trace back from this endpoint to the start
        current_path = []
        current_node = endpoint
        
        # Trace backwards using parent_id
        while current_node is not None:
            current_path.append(current_node)
            
            # Find parent
            parent_id = current_node.get('parent_id')
            if parent_id == 0:
                # Reached the start sequence, add it and break
                start_node = path_lookup.get(0)
                if start_node:
                    current_path.append(start_node)
                break
            elif parent_id is None:
                # No parent found
                break
            
            # Look for parent in path_lookup
            current_node = path_lookup.get(parent_id)
            if current_node is None:
                print(f"⚠️ Warning: Could not find parent with id {parent_id}")
                break
        
        # Reverse to get path from start to end
        current_path.reverse()
        
        if len(current_path) > 1:  # Valid path should have at least start and end
            complete_paths.append((idx, current_path))
            #print(f"✅ Path {idx}: {len(current_path)} steps from {current_path[0]['name']} to {current_path[-1]['name']}")
    
    # Find the END sequence
    end_sequence = next((path for path in path_history if path['status'] == 'END'), None)
    
    # Create FASTA files for each complete path
    for path_idx, path_steps in complete_paths:
        fasta_filename = f"{output_file_path}beam_evol_path_{path_idx}_{random_seed}.fasta"
        
        with open(fasta_filename, "w") as f:
            for step_idx, step in enumerate(path_steps):
                if step_idx == 0:
                    sequence_name = f"{step['name']}_START"
                else:
                    sequence_name = f"{step['name']}_step_{step_idx}"
                f.write(f">{sequence_name}\n{step['sequence']}\n")
            
            # Add the END sequence to complete the path
            if end_sequence:
                f.write(f">{end_sequence['name']}_END\n{end_sequence['sequence']}\n")
        
        logger.info(f"✅ Path {path_idx} saved to {fasta_filename}")
    
    logger.info(f"🎯 Total of {len(complete_paths)} complete evolutionary paths assembled")

def validate_path_consistency(path_history, output_file_path, random_seed):
    """
    Perform consistency checks on the path_history and generated files.
    
    Args:
        path_history (list): List of path dictionaries from evolution
        output_file_path (str): Path to output directory
        random_seed (int): Random seed used for file naming
        
    Returns:
        dict: Validation results
    """
    print(f"\n🔍 CONSISTENCY VALIDATION")
    
    # Check path_history structure
    total_entries = len(path_history)
    converged_count = len([p for p in path_history if p.get('converged', False) and p.get('status') != 'END'])
    accept_count = len([p for p in path_history if p.get('status') == 'ACCEPT'])
    end_count = len([p for p in path_history if p.get('status') == 'END'])
    start_count = len([p for p in path_history if p.get('id') == 0])
    
    print(f"  📊 Path History Statistics:")
    print(f"    - Total entries: {total_entries}")
    print(f"    - Start sequences (id=0): {start_count}")
    print(f"    - Converged candidates: {converged_count}")
    print(f"    - ACCEPT status: {accept_count}")
    print(f"    - END status: {end_count}")
    
    # Check for missing parent_ids
    all_ids = {p.get('id') for p in path_history}
    all_parent_ids = {p.get('parent_id') for p in path_history if p.get('parent_id') is not None}
    
    # Filter out special parent_id -9999999 (used for END sequence with multiple possible parents)
    missing_parents = all_parent_ids - all_ids - {-9999999}
    
    if missing_parents:
        print(f"  ⚠️ WARNING: Missing parent IDs: {missing_parents}")
    else:
        print(f"  ✅ All parent IDs are present in path_history")
    
    # Check for END sequence with special parent_id
    end_sequences_with_special_parent = [p for p in path_history if p.get('parent_id') == -9999999]
    if end_sequences_with_special_parent:
        print(f"  📝 END sequences with special parent_id (-9999999): {len(end_sequences_with_special_parent)}")
    
    # Check sequence name patterns
    sequence_names = [p.get('name', 'NO_NAME') for p in path_history if p.get('name')]
    unique_names = set(sequence_names)
    
    print(f"  📝 Sequence Names:")
    print(f"    - Total sequence entries: {len(sequence_names)}")
    print(f"    - Unique sequence names: {len(unique_names)}")
    
    if len(sequence_names) != len(unique_names):
        print(f"  ⚠️ WARNING: Duplicate sequence names detected!")
    
    # Sample sequence names
    sample_names = sequence_names[:5]
    print(f"    - Sample names: {sample_names}")
    
    # Check if intermediate file exists and validate path sequences against it
    intermediate_file = f"{output_file_path}beam_evol_msat_intermediate_seqs_{random_seed}.fasta"
    path_validation_passed = True
    
    if os.path.exists(intermediate_file):
        print(f"\n  🔍 Validating path sequences against intermediate file...")
        
        # Read intermediate file sequences
        try:
            from Bio import SeqIO
            intermediate_sequences = set()
            with open(intermediate_file, 'r') as f:
                for record in SeqIO.parse(f, "fasta"):
                    # Remove _START suffix if present
                    seq_name = record.id.replace('_START', '') if record.id.endswith('_START') else record.id
                    intermediate_sequences.add(seq_name)
            
            print(f"    - Found {len(intermediate_sequences)} sequences in intermediate file")
            
            # Check path sequences against intermediate sequences
            # All sequences in path files should have ACCEPT status (except END sequence)
            accept_sequences = set()
            start_sequences = set()
            end_sequences = set()
            
            for path in path_history:
                seq_name = path.get('name', '')
                if path.get('status') == 'ACCEPT':
                    if path.get('id') == 0:  # Start sequence with ACCEPT status
                        start_sequences.add(seq_name)
                    else:  # All other ACCEPT sequences (these go in path files)
                        accept_sequences.add(seq_name)
                elif path.get('status') == 'END':  # End sequence
                    end_sequences.add(seq_name)
            
            # Find ACCEPT sequences that should be in intermediate but are missing
            missing_from_intermediate = accept_sequences - intermediate_sequences
            
            print(f"    - Start sequences (ACCEPT status, excluded from check): {len(start_sequences)}")
            print(f"    - End sequences (END status, excluded from check): {len(end_sequences)}")
            print(f"    - ACCEPT sequences to validate: {len(accept_sequences)}")
            print(f"    - Missing from intermediate: {len(missing_from_intermediate)}")
            
            if missing_from_intermediate:
                print(f"  ⚠️ WARNING: ACCEPT sequences missing from intermediate file:")
                for seq in list(missing_from_intermediate)[:5]:  # Show first 5
                    print(f"    - {seq}")
                if len(missing_from_intermediate) > 5:
                    print(f"    - ... and {len(missing_from_intermediate) - 5} more")
                path_validation_passed = False
            else:
                print(f"  ✅ All ACCEPT sequences are present in intermediate file")
                
        except Exception as e:
            print(f"  ❌ Error reading intermediate file: {e}")
            path_validation_passed = False
    else:
        print(f"  ⚠️ WARNING: Intermediate file not found: {intermediate_file}")
        path_validation_passed = False
    
    validation_results = {
        'total_entries': total_entries,
        'converged_count': converged_count,
        'accept_count': accept_count,
        'end_count': end_count,
        'missing_parents': len(missing_parents),
        'unique_names': len(unique_names),
        'path_validation_passed': path_validation_passed
    }
    
    if path_validation_passed and len(missing_parents) == 0:
        print(f"  ✅ All consistency checks passed!")
    else:
        print(f"  ⚠️ Some consistency issues detected - see warnings above")
    
    return validation_results
    
    