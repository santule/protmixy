"""
MSA output utilities for extracting embeddings and statistics from MSA-Transformer.

This module provides functions to compute embeddings, entropy, and log-likelihood
from MSA-Transformer for candidate sequences in the context of an MSA.
"""
import torch
import os
import numpy as np
import tempfile
from src.utils import helpers
from src.utils.model_loader import ModelLoader
from config.settings import DEVICE

def get_embedding(context_msa_file):
    """
    Compute the MSA embedding for a candidate sequence using a pretrained MSA Transformer model.

    Args:
        context_msa_file (str): Path to the context MSA file.

    Returns:
        np.ndarray: Embedding of the candidate sequence.
    """
    model, alphabet = ModelLoader.get_model()
    msa_batch_converter = alphabet.get_batch_converter()

    if not os.path.exists(context_msa_file):
        raise FileNotFoundError(f"Context MSA file '{context_msa_file}' not found.")
    
    total_seqs,size_align_seq = helpers.check_msa_stats(context_msa_file)
    msa_data = [helpers.read_msa(context_msa_file, total_seqs)]
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    
    with torch.no_grad():
        out = model(msa_batch_tokens.to(DEVICE), repr_layers=[12], need_head_weights=False)
        token_representations_wo_cls = out["representations"][12][0,:,1:,:] # 1st position is CLS # 1,3,733,768
        msa_sequence_embedding    = token_representations_wo_cls.mean(dim = 1).to('cpu').squeeze(0).numpy() # 3,768
        
    return msa_batch_labels, msa_sequence_embedding

def get_query_embedding(candidate_sequence, context_msa_file):
  
    """
    Compute the MSA embedding for a candidate sequence using a pretrained MSA Transformer model.

    Args:
        candidate_sequence (str): Protein sequence of the candidate.
        context_msa_file (str): Path to the context MSA file.

    Returns:
        np.ndarray: Embedding of the candidate sequence.
    """
  
    model, alphabet = ModelLoader.get_model()
    msa_batch_converter = alphabet.get_batch_converter()

    # add candidate sequence to the context  
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as temp_file:
        temp_msa_file_name = temp_file.name
        helpers.prepend_sequence_to_fasta(candidate_sequence, 'candidate_sequence', context_msa_file, temp_msa_file_name)

    total_seqs,size_align_seq = helpers.check_msa_stats(temp_msa_file_name)
    msa_data = [helpers.read_msa(temp_msa_file_name, total_seqs)]
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    
    with torch.no_grad():
        out = model(msa_batch_tokens.to(DEVICE), repr_layers=[12], need_head_weights=False)
        token_representations_wo_cls = out["representations"][12][0,0,1:,:] # 1st position is CLS # 1,3,733,768
        candidate_sequence_embedding   = token_representations_wo_cls.mean(dim = 0).to('cpu').squeeze(0).numpy() # 3,768
        candidate_pos_representation = out["representations"][12][0,0,:,:].to('cpu') # keep as tensor
    
    return candidate_sequence_embedding, candidate_pos_representation

def get_current_state(candidate_sequence, context_msa_file):
  
    """
    Compute embeddings for the current MSA. Order of sequences in the MSA file should be:
    1: Query/Candidate, 2: Source, 3: Target, followed by context sequences

    Returns:
        tuple: (query_embedding, source_embedding, target_embedding, 
                query_pos_embedding, source_pos_embedding, target_pos_embedding)
    """
  
    model, alphabet = ModelLoader.get_model()
    msa_batch_converter = alphabet.get_batch_converter()

    # add candidate sequence to the context  
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as temp_file:
        temp_msa_file_name = temp_file.name
        helpers.prepend_sequence_to_fasta(candidate_sequence, 'candidate_sequence', context_msa_file, temp_msa_file_name)


    total_seqs, size_align_seq = helpers.check_msa_stats(temp_msa_file_name)
    msa_data = [helpers.read_msa(temp_msa_file_name, total_seqs)]
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    
    with torch.no_grad():
        out = model(msa_batch_tokens.to(DEVICE), repr_layers=[12], return_contacts=False, need_head_weights=False)
        
        all_representations = out["representations"][12][0]  # batch size is 1
        
        # Extract sequence embeddings (mean across positions, excluding CLS token)
        # For the first 2 sequences: source, target
        query_embedding  = all_representations[0, 1:, :].mean(dim=0).cpu().numpy()

        # mean of all target embedding
        #mean_target_embedding = all_representations[1:, 1:, :].mean(dim=(0,1)).cpu().numpy()
        mean_target_embedding = all_representations[1, 1:, :].mean(dim=0).cpu().numpy()
        
        # covariance of all target embedding
        target_sequence_embeddings = all_representations[1:, 1:, :].mean(dim=1).cpu().numpy()
        target_covariance = np.cov(target_sequence_embeddings, rowvar=False)
        # Regularize covariance (shrinkage) to avoid singular matrix
        target_covariance_reg = (1 - 0.1) * target_covariance + 0.1 * np.eye(target_covariance.shape[0])
        # Inverse covariance
        inv_cov_B = np.linalg.inv(target_covariance_reg)

        # check shape of query and target position embeddings else error
        if query_embedding.shape != mean_target_embedding.shape:
            raise ValueError("Query and Target embeddings have different shapes")

        # Extract position-wise embeddings (keep as tensors)
        query_pos_embedding  = all_representations[0, :, :].cpu()

        # mean of positions-wise embedings
        #mean_target_pos_embedding = all_representations[1:, :, :].mean(dim=0).cpu()
        mean_target_pos_embedding = all_representations[1, :, :].cpu()

        # check shape of query and target position embeddings else error
        if query_pos_embedding.shape != mean_target_pos_embedding.shape:
            raise ValueError("Query and Target position embeddings have different shapes")

        # entropy
        model_logits = out["logits"]
        # Convert logits to probabilities using softmax
        query_probs = torch.softmax(model_logits[:,0,:,:], dim=2) # 1, 734, 33
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        log_probs   = torch.log(query_probs + epsilon)
        query_pos_entropy = -torch.sum(query_probs * log_probs, dim=2)
        query_pos_entropy = query_pos_entropy.reshape(-1).cpu().numpy()
        
    return query_embedding, mean_target_embedding, query_pos_embedding, mean_target_pos_embedding, query_pos_entropy, query_probs.cpu().numpy(), inv_cov_B

def get_logits(candidate_sequence, context_msa_file):
    """
    Compute the MSA logits for a candidate sequence using a pretrained MSA Transformer model.

    Args:
        candidate_sequence_name (str): Name of the candidate sequence.
        candidate_sequence (str): Protein sequence of the candidate.
        context_msa_file (str): Path to the context MSA file.
        model: Pretrained MSA Transformer model.
        alphabet: Model's alphabet object for batch conversion.

    Returns:
        np.ndarray: Logits of the candidate sequence.
    """

    model, alphabet = ModelLoader.get_model()
    msa_batch_converter = alphabet.get_batch_converter()

    # add candidate sequence to the context  
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as temp_file:
        temp_msa_file_name = temp_file.name
        helpers.prepend_sequence_to_fasta(candidate_sequence, 'candidate_sequence', context_msa_file, temp_msa_file_name)

    total_seqs,size_align_seq = helpers.check_msa_stats(temp_msa_file_name)
    msa_data = [helpers.read_msa(temp_msa_file_name, total_seqs)]
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    
    with torch.no_grad():
        out = model(msa_batch_tokens.to(DEVICE), repr_layers=[12], need_head_weights=False) # [1, 3, 734, 33]
        model_logits = out["logits"].to('cpu').squeeze(0)  # 1,734,33
        candidate_sequence_logits = torch.softmax(model_logits[0].squeeze(0),dim=1) # 734,33
    
    return candidate_sequence_logits


def get_current_state_with_ll(candidate_sequence, context_msa_file):
  
    """
    Compute embeddings for the current MSA. Order of sequences in the MSA file should be:
    1: Query/Candidate, 2: Source, 3: Target, followed by context sequences

    Returns:
        tuple: (query_embedding, source_embedding, target_embedding, 
                query_pos_embedding, source_pos_embedding, target_pos_embedding)
    """
  
    model, alphabet = ModelLoader.get_model()
    msa_batch_converter = alphabet.get_batch_converter()

    # add candidate sequence to the context  
    with tempfile.NamedTemporaryFile(suffix=".fasta", delete=False) as temp_file:
        temp_msa_file_name = temp_file.name
        helpers.prepend_sequence_to_fasta(candidate_sequence, 'candidate_sequence', context_msa_file, temp_msa_file_name)


    total_seqs, size_align_seq = helpers.check_msa_stats(temp_msa_file_name)
    msa_data = [helpers.read_msa(temp_msa_file_name, total_seqs)]
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    
    with torch.no_grad():
        out = model(msa_batch_tokens.to(DEVICE), repr_layers=[12], return_contacts=False, need_head_weights=False)
        
        all_representations = out["representations"][12][0]  # batch size is 1
        model_logits = out["logits"]
        
        # Extract sequence embeddings (mean across positions, excluding CLS token) - For the first 2 sequences: source, target
        query_embedding  = all_representations[0, 1:, :].mean(dim=0).cpu().numpy()
        target_embedding = all_representations[1, 1:, :].mean(dim=0).cpu().numpy()
        
        # Extract position-wise embeddings (keep as tensors)
        query_pos_embedding  = all_representations[0, :, :].cpu()
        target_pos_embedding = all_representations[1, :, :].cpu()

        # check shape of query and target position embeddings else error
        if query_embedding.shape != target_embedding.shape:
            raise ValueError("Query and Target embeddings have different shapes")

        # check shape of query and target position embeddings else error
        if query_pos_embedding.shape != target_pos_embedding.shape:
            raise ValueError("Query and Target position embeddings have different shapes")
    
        # query sequence  - log likelihood calculation
        #idx_list = alphabet.tok_to_idx # reference dict from aa to model token
        #print(idx_list.items())
        #aa_list  = {v: k for k,v in idx_list.items()}

        query_token_probs = torch.log_softmax(model_logits[:,0,:,:], dim=-1).squeeze(0) # 734, 33
        query_token_probs = query_token_probs[1:] # avoid cls token
        query_str = msa_batch_strs[0][0] # msa_batch_strs is list of list
        #print(query_str)
        idx = [alphabet.tok_to_idx[q] for q in query_str]
        #print(idx)
        query_ll_sum = np.sum(np.diag(query_token_probs[:,idx].cpu().numpy()))
        query_avg_ll = query_ll_sum  #/ len(query_token_probs)

        # DEBUG: Uncomment below for testing log-likelihood calculation
        # print(f"Query string length: {len(query_str)}")
        # print(f"Token indices: {idx}")
        # print(f"query_token_probs shape: {query_token_probs.shape}")
        # # Check if dimensions match
        # if len(idx) != query_token_probs.shape[0]:
        #     print(f"⚠️ Dimension mismatch: query_str has {len(query_str)} chars, but query_token_probs has {query_token_probs.shape[0]} positions")
        # # Verify the calculation step by step
        # selected_probs = query_token_probs[:,idx].cpu().numpy()
        # print(f"Selected probabilities shape: {selected_probs.shape}")
        # diagonal_values = np.diag(selected_probs)
        # print(f"Diagonal values (first 5): {diagonal_values[:5]}")
        # print(f"Log-likelihood sum: {np.sum(diagonal_values):.4f}, average: {np.sum(diagonal_values)/len(diagonal_values):.4f}")
        
        # query sequence - entropy calculation
        query_pos_probs = torch.softmax(model_logits[:,0,:,:], dim=2) # 1, 734, 33
        epsilon = 1e-10
        log_probs   = torch.log(query_pos_probs + epsilon)
        query_pos_entropy = -torch.sum(query_pos_probs * log_probs, dim=2)
        query_pos_entropy = query_pos_entropy.reshape(-1).cpu().numpy()

        return query_embedding, target_embedding, query_pos_embedding, target_pos_embedding, query_pos_entropy, query_avg_ll