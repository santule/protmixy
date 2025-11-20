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
    
    total_seqs, size_align_seq = helpers.check_msa_stats(context_msa_file)
    msa_data = [helpers.read_msa(context_msa_file, total_seqs)]
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    
    with torch.no_grad():
        out = model(msa_batch_tokens.to(DEVICE), repr_layers=[12], need_head_weights=False)
        token_representations_wo_cls = out["representations"][12][0,:,1:,:] # 1st position is CLS # 1,3,733,768
        msa_sequence_embedding    = token_representations_wo_cls.mean(dim = 1).to('cpu').squeeze(0).numpy() # 3,768
        
    return msa_batch_labels, msa_sequence_embedding

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

        query_token_probs = torch.log_softmax(model_logits[:,0,:,:], dim=-1).squeeze(0) # 734, 33
        query_token_probs = query_token_probs[1:] # avoid cls token
        query_str = msa_batch_strs[0][0] # msa_batch_strs is list of list
        idx = [alphabet.tok_to_idx[q] for q in query_str]
        query_ll_sum = np.sum(np.diag(query_token_probs[:,idx].cpu().numpy()))
        query_avg_ll = query_ll_sum  #/ len(query_token_probs)
     
        # query sequence - entropy calculation
        query_pos_probs = torch.softmax(model_logits[:,0,:,:], dim=2) # 1, 734, 33
        epsilon = 1e-10
        log_probs   = torch.log(query_pos_probs + epsilon)
        query_pos_entropy = -torch.sum(query_pos_probs * log_probs, dim=2)
        query_pos_entropy = query_pos_entropy.reshape(-1).cpu().numpy()

        return query_embedding, target_embedding, query_pos_embedding, target_pos_embedding, query_pos_entropy, query_avg_ll