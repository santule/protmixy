''' given an MSA, create a conditioning context using MSA-Transformer embeddings and cosine distance and KNN '''

import torch
import numpy as np
import os, json
from pysam import FastaFile
from src.utils.model_loader import ModelLoader
from config.settings import INPUT_FILE_PATH, FULL_MSA_FILE, DENSE_SEED_THRESHOLD
from sklearn.neighbors import NearestNeighbors
from src.utils import msat_output

# load the generator + alphabet
msa_transformer, msa_alphabet = ModelLoader.get_model()
msa_batch_converter = msa_alphabet.get_batch_converter()
idx_list = msa_alphabet.tok_to_idx # reference dict for model token
aa_list  = {v: k for k,v in idx_list.items()}
valid_aa_vals = torch.tensor([ 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 
                             18, 19, 20, 21, 22, 23, 30],dtype=torch.int64)

def l2_normalize(X, eps=1e-12):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return X / n

# perform KNN 
def compute_knn_cosine_stats(X, k=50, normalize=True, agg="mean", algorithm="auto"):
    """
    Compute, for each point i in X (N,d), the k nearest neighbours by cosine
    (including itself if k+1 requested), and return aggregated distance (mean/median)
    and the neighbor indices.
    Returns:
      agg_dists: array (N,) aggregated distances (lower -> denser)
      knn_idxs: array (N, k) neighbor indices (excluding self if possible)
      knn_dists: array (N, k) neighbor distances (cosine distances)
    """
    Xn = l2_normalize(X) if normalize else X
    n  = Xn.shape[0]
    # we ask for k+1 neighbors to include self (distance 0) then drop it
    neigh = min(k + 1, n)
    nbrs  = NearestNeighbors(n_neighbors=neigh, metric="cosine", algorithm=algorithm).fit(Xn)
    dists, idxs = nbrs.kneighbors(Xn, return_distance=True)  # shapes (N, neigh)
    # remove self (first column should be 0 distance to itself)
    if neigh > 1:
        dists = dists[:, 1:]   # shape (N, k') where k' = min(k, n-1)
        idxs  = idxs[:, 1:]
    else:
        dists = np.zeros((n, 0))
        idxs  = np.zeros((n, 0), dtype=int)

    if agg == "mean":
        agg_dists = dists.mean(axis=1)
    elif agg == "median":
        agg_dists = np.median(dists, axis=1)
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    return agg_dists, idxs, dists

def greedy_nonoverlapping_seeds(X, max_k=50, top_n=20, normalize=True, agg="mean"):
    agg_dists, knn_idxs, _ = compute_knn_cosine_stats(X, k=max_k, normalize=normalize, agg=agg)
    remaining = set(range(X.shape[0]))
    seeds = []
    
    while len(seeds) < top_n and remaining:
        # consider only remaining points
        cand = np.array(list(remaining))
        # choose remaining point with smallest agg_dists
        best_rel = np.argmin(agg_dists[cand])
        best = int(cand[best_rel])
        seeds.append({
            "seed_idx": best,
            "agg_dist": float(agg_dists[best]),
            "knn_indices": knn_idxs[best].tolist()
        })
        # remove best and its neighbours from remaining (so next seed is far away)
        neighs = set(int(x) for x in knn_idxs[best])
        neighs.add(best)
        remaining -= neighs
    return seeds

def multi_k_dense_seeds(
    X,
    k_values = (40, 30, 20),
    top_n_per_k = 20,
    normalize = True,
    agg = "mean"):
    """Run greedy_nonoverlapping_seeds for several k values and rank all seeds.

    For each k in k_values, we:
      - compute dense, non-overlapping seeds via greedy_nonoverlapping_seeds
      - tag each seed with the k used

    All seeds from all k are then combined and sorted by agg_dist (ascending),
    so the first entries correspond to the densest local neighborhoods across
    all tested scales.
    """

    all_seeds = []
    for k in k_values:
        k_seeds = greedy_nonoverlapping_seeds(
            X,
            max_k=k,
            top_n=top_n_per_k,
            normalize=normalize,
            agg=agg,
        )
        for seed in k_seeds:
            seed_with_k = dict(seed)
            seed_with_k["k"] = k
            all_seeds.append(seed_with_k)

    # Order seeds globally by aggregated distance (densest first)
    all_seeds.sort(key=lambda s: s["agg_dist"])

    # Heuristic filter: keep only very dense seeds
    filtered_seeds = [s for s in all_seeds if s["agg_dist"] <= DENSE_SEED_THRESHOLD]
    return filtered_seeds

def get_full_msa_embeddings(msa_file):
    msa_batch_labels, msa_sequence_embedding_np = msat_output.get_embedding(msa_file)
    msa_batch_labels     = msa_batch_labels[0]
    all_embeddings_np    = msa_sequence_embedding_np[:,1:] # (n, 768)
    print(f"seq_labels {len(msa_batch_labels)} and embeddings {all_embeddings_np.shape}")

    embedding_cache_file = f"{INPUT_FILE_PATH}/full_context_embeddings.npz"
    np.savez(embedding_cache_file, labels=np.array(msa_batch_labels, dtype=object), embeddings= all_embeddings_np)
    print(f"Saved embeddings to {embedding_cache_file}")

    return msa_batch_labels, all_embeddings_np

def find_cluster_center(cluster_indices, all_embeddings_np):
    # compute mean embedding for this dense cluster (seed + its neighbours)
    cluster_emb = np.mean(all_embeddings_np[cluster_indices], axis=0)
    # Normalize embeddings
    cluster_norm = np.linalg.norm(cluster_emb) + 1e-8
    cluster_unit = cluster_emb / cluster_norm
    all_norms = np.linalg.norm(all_embeddings_np, axis=1, keepdims=True) + 1e-8
    all_unit = all_embeddings_np / all_norms
    # Cosine distance = 1 - cosine similarity
    cos_sims = all_unit @ cluster_unit
    cos_dists = 1.0 - cos_sims
    nearest_idx = int(np.argmin(cos_dists))
    return nearest_idx

def create_conditioning_context_aln_file(cluster_indices, msa_batch_labels, nearest_idx):
    fasta = FastaFile(FULL_MSA_FILE)
    nearest_label = msa_batch_labels[nearest_idx]

    out_path = f"{INPUT_FILE_PATH}conditioning_context_{nearest_label}.aln"
    with open(out_path, "w") as out_f:
        seq = fasta.fetch(nearest_label)
        out_f.write(f">{nearest_label}\n{seq}\n")

        for idx in cluster_indices:
            label = msa_batch_labels[idx]
            if label == nearest_label:
                continue
            seq = fasta.fetch(label)
            out_f.write(f">{label}\n{seq}\n")

    print(f"Saved conditioning context FASTA to {out_path}")    

def seq_identity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    # Use min length and compare prefix
    L = min(len(a), len(b))
    if L == 0:
        return 0.0
    matches = sum(c1 == c2 for c1, c2 in zip(a[:L], b[:L]))
    return matches / float(L)   

def find_potential_source_sequence(cluster_indices, msa_batch_labels, target_idx):
    """Select up to 5 candidate source sequences at varying identity to target.

    - Excludes sequences that are in the given cluster (cluster_indices)
      and the target itself.
    - Uses simple ungapped sequence identity on the FULL_MSA_FILE MSA.
    - Picks sequences spread across the identity range by sampling
      roughly evenly from the sorted list.
    """

    fasta = FastaFile(FULL_MSA_FILE)

    # Map indices in cluster for quick exclusion
    cluster_set = set(cluster_indices)

    target_label = msa_batch_labels[target_idx]
    target_seq = fasta.fetch(target_label).replace("\n", "")
    candidates = []  # (idx, label, identity)
    for idx, label in enumerate(msa_batch_labels):
        if idx == target_idx or idx in cluster_set:
            continue
        seq = fasta.fetch(label).replace("\n", "")
        ident = seq_identity(target_seq, seq)
        candidates.append((idx, label, ident))

    if not candidates:
        return []

    # Sort by identity and pick up to 5 spread across the range
    candidates.sort(key=lambda x: x[2])  # ascending identity
    n_sources = min(5, len(candidates))
    if n_sources == 0:
        return [], []

    step = max(1, len(candidates) // n_sources)
    chosen_indices = []
    chosen_idents = []
    for i in range(0, len(candidates), step):
        idx_i, label_i, ident_i = candidates[i]
        chosen_indices.append(idx_i)
        chosen_idents.append(ident_i)
        if len(chosen_indices) >= n_sources:
            break

    return chosen_indices, chosen_idents

def main():
    # check if input directory exists
    if not os.path.exists(FULL_MSA_FILE):
        print(f'Please provide {FULL_MSA_FILE} file in the {INPUT_FILE_PATH} in order to find potential sequences for conditioning context')
        raise FileNotFoundError(f"Input file {FULL_MSA_FILE} not found")
        

    # get msa-transformer embeddings
    print(f"Processing and getting MSA-Transformer embeddings")
    msa_batch_labels, all_embeddings_np = get_full_msa_embeddings(FULL_MSA_FILE)
    
    # get dense clusters of sequences
    print(f"Getting dense clusters of sequences")
    dense_seeds = multi_k_dense_seeds(all_embeddings_np)
    print(f"Found {len(dense_seeds)} dense seeds")

    # lets choose top 5 clusters, can be changed as per requirement
    print(f"Choosing top 5 dense clusters")
    dense_seeds = dense_seeds[:5]

    # Attach labels and dump to JSON/TSV for downstream use
    print(f"Attaching labels and dumping to JSON/TSV for downstream use")
    labelled_seeds = []
    for seed in dense_seeds:
        seed_idx = seed["seed_idx"]
        knn_indices = seed["knn_indices"]

        # Compute mean embedding for this dense cluster (seed + its neighbours)
        cluster_indices = [seed_idx] + knn_indices

        # Compute mean embedding and find the nearest sequence to the mean embedding (cosine distance)
        nearest_idx = find_cluster_center(cluster_indices, all_embeddings_np)

        # potential source sequences for conditioning context (indices outside the cluster)
        source_indices, source_idents = find_potential_source_sequence(cluster_indices, msa_batch_labels, nearest_idx)

        # print cluster stats
        print(f"Cluster size: {len(cluster_indices)} with the target sequence {msa_batch_labels[nearest_idx]} and total cluster tightness {seed['agg_dist']}")

        labelled_seeds.append(
            {
                "k": seed["k"],
                "seed_idx": seed_idx,
                "seed_label": msa_batch_labels[seed_idx],
                "agg_dist": seed["agg_dist"],
                "knn_indices": knn_indices,
                "knn_labels": [msa_batch_labels[i] for i in knn_indices],
                "target_idx": nearest_idx,
                "target_label": msa_batch_labels[nearest_idx],
                "source_indices": source_indices,
                "source_labels_list": [msa_batch_labels[i] for i in source_indices],
                "source_idents": source_idents,
            }
        )

        # create conditioning context alignment file
        create_conditioning_context_aln_file(cluster_indices, msa_batch_labels,nearest_idx)
    
    json_out = f"{INPUT_FILE_PATH}potential_cond_context.json"
    with open(json_out, "w") as f:
        json.dump(labelled_seeds, f, indent=2)
    print(f"Saved potential conditioning context JSON to {json_out}")



if __name__ == "__main__":
    main()
