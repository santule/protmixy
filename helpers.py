import string, itertools
import tempfile
from typing import List, Tuple
from Bio import SeqIO
from pysam import FastaFile,FastxFile
import h5py
from ete3 import Tree
import os
import csv
import config.settings as settings
import matplotlib.pyplot as plt


deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def remove_gaps_from_fasta(fasta_file, fasta_file_wo_gaps):
    with open(fasta_file_wo_gaps, 'w') as out_fasta:
        with FastxFile(fasta_file, 'r') as fh:
            for entry in fh:
                out_fasta.write(f">{entry.name}\n{entry.sequence.replace('-','')}\n")


def clean_plt(ax):
    ax.tick_params(direction='out', length=2, width=1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['top'].set_linewidth(0)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['right'].set_linewidth(0)
    ax.tick_params(labelsize=10.0)
    ax.tick_params(axis='x', which='major', pad=2.0)
    plt.xticks(rotation=45)
    ax.tick_params(axis='y', which='major', pad=2.0)
    return ax
    
def check_msa_stats(msa_file):
    total_seqs = 0
    with FastxFile(msa_file, 'r') as fh:
        for entry in fh:
            total_seqs += 1
            size_align_seq = len(entry.sequence)
    return total_seqs,size_align_seq

def read_fasta(fasta_path):
    sequences = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.replace('>', '').strip()
                sequences[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                sequences[ uniprot_id ] += ''.join( line.split() ).upper().replace("-","") # drop gaps and cast to upper-case          
    return sequences

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""    
    
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]

def get_similarity_group(similarity_score):
    """
    Assigns a similarity score to a predefined group.

    Args:
        similarity_score (float or str): The sequence similarity score.

    Returns:
        str: The similarity group label.
    """
    try:
        similarity_score = float(similarity_score)
    except (ValueError, TypeError):
        return "N/A"

    if 0.10 <= similarity_score <= 0.5:
        return "10-50%"
    elif 0.5 < similarity_score <= 0.6:
        return "50-60%"
    elif 0.6 < similarity_score <= 0.7:
        return "60-70%"
    elif 0.7 < similarity_score <= 0.8:
        return "70-80%"
    elif 0.8 < similarity_score <= 0.9:
        return "80-90%"
    elif 0.9 < similarity_score <= 1.0:
        return "90-100%"
    else:
        return "N/A"

def prepend_sequence_to_fasta(candidate_sequence, candidate_sequence_name, context_msa_file, new_msa_file):
    
    # Read the original content of the file
    with open(context_msa_file, 'r') as file:
        original_content = file.read()

    # Create the new sequence in FASTA format
    new_sequence = f">{candidate_sequence_name}\n{candidate_sequence}\n"
    
    # Prepend the new sequence to the original content
    updated_content = new_sequence + original_content
    

    # Write the updated content back to the file
    # check if directory exists
    os.makedirs(os.path.dirname(new_msa_file), exist_ok=True)
    with open(new_msa_file, 'w') as file:
        file.write(updated_content)

def remove_duplicate_sequences(input_fasta, output_fasta):
    # remove duplicate sequences
    unique_sequences = {}
    # Read the input FASTA file
    for record in SeqIO.parse(input_fasta, "fasta"):
        seq_str = str(record.seq)
        if seq_str not in unique_sequences:
            unique_sequences[seq_str] = record

    # Write the unique sequences to the output FASTA file
    with open(output_fasta, "w") as output_handle:
        SeqIO.write(unique_sequences.values(), output_handle, "fasta")

def save_embeddings_to_hdf5(embeddings, file_name):
    with h5py.File(file_name, "w") as h5file:
        for seq_name, emb in embeddings.items():
            h5file.create_dataset(seq_name, data=emb)

# Function to load embeddings from an HDF5 file
def load_embeddings_from_hdf5(file_name):
    embeddings = {}
    with h5py.File(file_name, "r") as h5file:
        for seq_name in h5file.keys():
            embeddings[seq_name] = h5file[seq_name][:]
    return embeddings

# code to get only extant sequences from the msa and tree file
def get_extant_sequences(nwk_tree_file,global_msa_fasta,extant_msa_fasta):

    extant_seqs = []
    
    tree = Tree(nwk_tree_file,format=1)
    for node in tree.traverse():
        if node.is_leaf():
            extant_seqs.append(node.name)

    all_seq_lkp = FastaFile(global_msa_fasta)

    with open(extant_msa_fasta, 'w') as out_f:
        for ext_seq in extant_seqs:
            out_f.write(f">{ext_seq}\n{all_seq_lkp.fetch(ext_seq)}\n")
           
# create aln file from fasta file. fasta file has no gaps
def create_aln_file(cluster_rep_fasta_file, global_aln_file, output_aln_file):

    try:
        # Get sequence names from cluster representatives
        wanted_names = set()
        for record in SeqIO.parse(cluster_rep_fasta_file, "fasta"):
            wanted_names.add(record.id)

        if not wanted_names:
            raise ValueError(f"No sequences found in {cluster_rep_fasta_file}")


        # Create a dictionary of sequences from global alignment file
        lookup_dict = SeqIO.to_dict(SeqIO.parse(global_aln_file, "fasta"))

        if not lookup_dict:
            raise ValueError(f"No sequences found in {global_aln_file}")

        # Keep track of found and not found sequences
        found_sequences = []
        not_found = set()

        # Write sequences to output file
        with open(output_aln_file, 'w') as output_handle:
            for name in wanted_names:
                try:
                    record = lookup_dict[name]
                    output_handle.write(f">{record.id}\n{str(record.seq)}\n")
                    found_sequences.append(name)
                except KeyError:
                    not_found.add(name)

        # Print summary
        print(f"Successfully wrote {len(found_sequences)} sequences to {output_aln_file}")
        if not_found:
            print(f"Warning: Could not find {len(not_found)} sequences:")
            print("\n".join(sorted(not_found)))

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
    except Exception as e:
        print(f"Error: {e}")
        

def load_protein_pairs_details_from_csv(csv_file_path):
    """
    Load protein pairs from a CSV file.
    
    Parameters:
    -----------
    csv_file_path : str
        Path to the CSV file containing protein pairs.
        Expected format: CSV with headers including 'protein1' and 'protein2' columns.
        
    Returns:
    --------
    list of tuples
        List of (protein1, protein2) pairs.
    """
    protein_pairs = []
    
    try:
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                family = row['family']
                protein1 = row['sequence1']
                cluster1 = row['cluster1']
                protein2 = row['sequence2']
                cluster2 = row['cluster2']
                group1 = row['group1']
                group2 = row['group2']
                max_mask = row['max_mask']
                min_mask = row['min_mask']
                n_iterations = row['iterations']
                sequence_similarity = row['sequence_similarity']
                no_diff_gap_positions = row['no_diff_gap_positions']
                is_within_cluster = row['is_within_cluster']
                to_run = row['to_run']
                protein_pairs.append((family, protein1, cluster1, protein2, cluster2, group1, group2, max_mask, min_mask, n_iterations, sequence_similarity, no_diff_gap_positions, is_within_cluster, to_run))
                
        return protein_pairs
        
    except Exception as e:
        print(f"Error loading protein pairs from CSV: {e}")
        return []

def update_msat_run_settings(start_seq, end_seq, max_mask, min_mask, n_iter, mask_cycle, generator_method):
    """
    Update the settings for the current protein pair.
    
    Parameters:
    -----------
    start_seq : str
        Name of the start sequence.
    end_seq : str
        Name of the end sequence.
    """
    settings.START_SEQ_NAME   = start_seq
    settings.END_SEQ_NAME     = end_seq
    settings.MAX_P_MASK       = float(max_mask)
    settings.MIN_P_MASK       = float(min_mask)
    settings.N_ITER           = int(n_iter)
    settings.MASK_CYCLE       = int(mask_cycle)
    settings.PAIR_OUTPUT_FILE_PATH = f"{settings.OUTPUT_FILE_PATH}{start_seq}_{end_seq}/{settings.CONTEXT_METHOD}/"
    settings.GENERATOR_METHOD = generator_method

    if settings.GENERATOR_METHOD == 'irs':
        settings.GENERATOR_OUTPUT_PATH   = f"{settings.PAIR_OUTPUT_FILE_PATH}irs/"
        
    if settings.GENERATOR_METHOD == 'apc':
        settings.GENERATOR_OUTPUT_PATH   = f"{settings.PAIR_OUTPUT_FILE_PATH}apc/"
       
    if settings.GENERATOR_METHOD == 'random':
        settings.GENERATOR_OUTPUT_PATH   = f"{settings.PAIR_OUTPUT_FILE_PATH}random/"
    
    if settings.GENERATOR_METHOD == 'asr':
        settings.GENERATOR_OUTPUT_PATH   = f"{settings.PAIR_OUTPUT_FILE_PATH}asr/"

    # Make the output folders if they don't exist
    if not os.path.exists(settings.OUTPUT_FILE_PATH):
        os.makedirs(settings.OUTPUT_FILE_PATH, exist_ok=True)

    if not os.path.exists(settings.PAIR_OUTPUT_FILE_PATH):
        os.makedirs(settings.OUTPUT_FILE_PATH, exist_ok=True)

    if not os.path.exists(settings.GENERATOR_OUTPUT_PATH):
        os.makedirs(settings.GENERATOR_OUTPUT_PATH, exist_ok=True)


def update_msat_run_settings_all(start_seq, end_seq, max_mask, min_mask, n_iter, mask_cycle, generator_method,family):
    """
    Update the settings for the current protein pair.
    
    Parameters:
    -----------
    start_seq : str
        Name of the start sequence.
    end_seq : str
        Name of the end sequence.
    """
    settings.PROTEIN_FAMILY = family
    settings.START_SEQ_NAME   = start_seq
    settings.END_SEQ_NAME     = end_seq
    settings.MAX_P_MASK       = float(max_mask)
    settings.MIN_P_MASK       = float(min_mask)
    settings.N_ITER           = int(n_iter)
    settings.MASK_CYCLE       = int(mask_cycle)
    settings.MAIN_DATA_PATH   = f"/protmixi/data/{family}/"
    settings.OUTPUT_FILE_PATH = f"/protmixi/data/{family}/output_data/"
    settings.PAIR_OUTPUT_FILE_PATH = f"/protmixi/data/{family}/output_data/{start_seq}_{end_seq}/{settings.CONTEXT_METHOD}/"
    settings.GENERATOR_METHOD = generator_method
    settings.INPUT_FILE_PATH  = f"/protmixi/data/{family}/input_data/"
    settings.FULL_CONTEXT_FILE= f"/protmixi/data/{family}/input_data/{family}_extants.aln"


    if settings.GENERATOR_METHOD == 'irs':
        settings.GENERATOR_OUTPUT_PATH   = f"/protmixi/data/{family}/output_data/{start_seq}_{end_seq}/{settings.CONTEXT_METHOD}/irs/"
        
    if settings.GENERATOR_METHOD == 'apc':
        settings.GENERATOR_OUTPUT_PATH   = f"/protmixi/data/{family}/output_data/{start_seq}_{end_seq}/{settings.CONTEXT_METHOD}/apc/"
       
    if settings.GENERATOR_METHOD == 'random':
        settings.GENERATOR_OUTPUT_PATH   = f"/protmixi/data/{family}/output_data/{start_seq}_{end_seq}/{settings.CONTEXT_METHOD}/random/"
    
    # Make the output folders if they don't exist
    if not os.path.exists(settings.OUTPUT_FILE_PATH):
        os.makedirs(settings.OUTPUT_FILE_PATH, exist_ok=True)

    if not os.path.exists(settings.PAIR_OUTPUT_FILE_PATH):
        os.makedirs(settings.OUTPUT_FILE_PATH, exist_ok=True)

    if not os.path.exists(settings.GENERATOR_OUTPUT_PATH):
        os.makedirs(settings.GENERATOR_OUTPUT_PATH, exist_ok=True)

def remove_start_end_sequence_fasta(fasta_file, protein1, protein2):
    """
    Remove sequences that match protein1 and protein2 from a FASTA file
    and write the filtered sequences to a temporary file.
    
    Args:
        fasta_file: Path to the input FASTA file
        protein1: First protein sequence to remove
        protein2: Second protein sequence to remove
        
    Returns:
        Path to the temporary file with filtered sequences
    """

    
    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.fasta')
    temp_file_path = temp_file.name
    
    # Read sequences from the input FASTA file
    sequences = list(SeqIO.parse(fasta_file, "fasta"))

    # Filter out sequences that match protein1 or protein2
    filtered_sequences = []
    for seq in sequences:
        seq_str = str(seq.seq)
        seq_name = seq.id
        # Skip sequences that match protein1 or protein2
        if seq_name == protein1 or seq_name == protein2 :
            continue
        filtered_sequences.append(seq)
    
    # Write filtered sequences to the temporary file
    SeqIO.write(filtered_sequences, temp_file_path, "fasta")
    
    return temp_file_path
   
def add_starting_sequence(fasta_file: str, protein: str, start_seq_name: str = "START") -> str:
    """
    Add the starting sequence to the fasta file.
    
    Args:
        fasta_file (str): Path to the fasta file
        protein (str): Name of the protein
        start_seq_name (str): Name for the starting sequence
        
    Returns:
        str: Path to the fasta file with the starting sequence added
    """
    # Read the fasta file
    with open(fasta_file, 'r') as f:
        lines = f.readlines()
    
    # Add the starting sequence
    lines.insert(0, f">{start_seq_name}\n")
    lines.insert(1, f"{protein}\n")
    
    # Validation check to ensure the sequence was added correctly
    if len(lines) >= 2:
        if not lines[0].startswith(f">{start_seq_name}") or not lines[1].strip() == protein:
            print(f"Warning: Starting sequence may not have been added correctly to {fasta_file}")
            print(f"First two lines: {lines[0].strip()}, {lines[1].strip()}")
    else:
        print(f"Warning: File {fasta_file} has fewer lines than expected after insertion")

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.fasta')
    temp_file_path = temp_file.name

    # Write the fasta file
    with open(temp_file_path, 'w') as f:
        f.writelines(lines)
    
    return temp_file_path