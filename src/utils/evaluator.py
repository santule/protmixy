"""
Evaluation functions for scoring candidate protein sequences.

This module provides the PathwayEvaluator class which computes various
distance metrics between candidate sequences and target sequences using
MSA-Transformer embeddings.
"""
import torch.nn.functional as F
from scipy.spatial.distance import cosine
from src.utils import msa_output


class PathwayEvaluator:
    _target_embedding = None
    _target_pos_embedding = None
    _source_embedding = None
    _source_pos_embedding =  None

    @staticmethod
    def scorer(candidate_sequence, context_msa_file):

        '''returns
        cos_dist_to_tgt: overall cosine distance between source and target
        pos_cos_dist_to_tgt: position wise cosine distance between source and target
        query_pos_entropy: query position entropy
        query_ll: query likelihood

         '''

        # get query, target embedding overall and position wise. Also get query position entropy and query likelihood
        query_embedding, target_embedding, query_pos_embedding, target_pos_embedding, query_pos_entropy, query_ll \
        = msa_output.get_current_state_with_ll(candidate_sequence, context_msa_file)

        # overall distance between source and target using cosine distance
        cos_dist_to_tgt = cosine(target_embedding, query_embedding)

        # position wise distance between source and target
        pos_cos_dist_to_tgt = F.cosine_similarity(query_pos_embedding, target_pos_embedding, dim=1)
        pos_cos_dist_to_tgt = 1 - pos_cos_dist_to_tgt
        pos_cos_dist_to_tgt = pos_cos_dist_to_tgt.cpu().numpy()


        return cos_dist_to_tgt, pos_cos_dist_to_tgt, query_pos_entropy, query_ll