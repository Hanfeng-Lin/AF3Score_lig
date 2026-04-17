from Bio import PDB
from Bio.PDB import Structure, Model
from Bio.PDB import PDBParser, MMCIFIO
import os
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
from pathlib import Path
import math
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

# Dictionary for converting three-letter amino acid codes to single-letter codes
protein_letters_3to1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
    "MSE": "M",
}


def split_by_total_length(df, num_jobs):
    """
    Split the dataframe into num_jobs batches based on the sum of 'total_length':
    1. Ensures each batch has a similar total sequence length sum for load balancing.
    2. Groups samples with similar lengths together.
    """
    # Sort by total_length in descending order (Greedy approach for load balancing)
    df = df.sort_values("total_length", ascending=False).reset_index(
        drop=True
    )

    total_sum = df["total_length"].sum()
    target_sum = total_sum / num_jobs

    groups = []
    current_group = []
    current_sum = 0

    for _, row in df.iterrows():
        val = int(row["total_length"])

        # If adding the current row exceeds the target sum, start a new group
        # Constraint: ensures we don't create too many small groups early on
        if current_sum + val > target_sum and len(groups) < num_jobs - 2:
            groups.append(pd.DataFrame(current_group))
            current_group = [row]
            current_sum = val
        else:
            current_group.append(row)
            current_sum += val

    # Handle the final batch splitting to reach the exact num_jobs count
    len_last_group = len(current_group)
    index = int(len_last_group / 2.2)
    groups.append(pd.DataFrame(current_group[:index]))
    groups.append(pd.DataFrame(current_group[index:]))

    return groups


def format_msa_sequence(sequence):
    """Format a raw sequence into a basic MSA query string."""
    return f">query\n{sequence}\n"


def get_chain_sequences_from_row(row):
    """Extract all non-empty chain sequences from a dataframe row."""
    chain_sequences = []
    chain_columns = [
        col
        for col in row.index
        if col.startswith("chain_") and col.endswith("_seq")
    ]
    for col in chain_columns:
        if pd.notna(row[col]) and row[col] != "":
            chain_id = col.split("_")[1]
            chain_sequences.append((chain_id, row[col]))
    return chain_sequences


def get_sequence_from_chain(chain):
    """Convert Biopython chain object to a single-letter amino acid sequence."""
    sequence = ""
    for residue in chain:
        # ' ' indicates a standard residue (not heteroatoms)
        if residue.id[0] == " ":
            resname = residue.get_resname().upper()
            sequence += protein_letters_3to1.get(resname, "X")
    return sequence


def is_ligand_chain(chain):
    """Check if a chain is a ligand (no standard amino acid residues)."""
    for residue in chain:
        if residue.id[0] == " ":
            resname = residue.get_resname().upper()
            if resname in protein_letters_3to1:
                return False
    return True


def count_ligand_atoms(chain):
    """Count the number of atoms in a ligand chain (each atom = 1 AF3 token)."""
    count = 0
    for residue in chain:
        for atom in residue:
            count += 1
    return count


def process_single_pdb(args):
    """
    Process a single PDB file:
    1. Extract sequences for each chain (protein or ligand).
    2. Save each chain as a separate .cif file for AlphaFold3 template compatibility.
    """
    input_pdb, output_dir_cif = args
    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure("structure", input_pdb)
        base_name = os.path.splitext(os.path.basename(input_pdb))[0]

        chain_sequences = {}
        ligand_chains = {}
        total_tokens = 0

        # Iterate through the first model (index 0) of the PDB
        for chain in structure[0]:
            chain_id = chain.id

            if is_ligand_chain(chain):
                # Ligand chain: count atoms (1 atom = 1 token in AF3)
                n_atoms = count_ligand_atoms(chain)
                ligand_chains[chain_id] = n_atoms
                total_tokens += n_atoms
            else:
                # Protein chain
                sequence = get_sequence_from_chain(chain)
                chain_sequences[chain_id] = sequence
                total_tokens += len(sequence)

                # Create a new structure object containing only this specific chain
                new_structure = Structure.Structure("new_structure")
                new_model = Model.Model(0)
                new_structure.add(new_model)
                new_model.add(chain.copy())

                # Save as MMCIF format (only for protein chains used as templates)
                cif_io = MMCIFIO()
                cif_io.set_structure(new_structure)
                cif_output = os.path.join(
                    output_dir_cif, f"{base_name}_chain_{chain_id}.cif"
                )
                cif_io.save(cif_output)

        return base_name, chain_sequences, total_tokens, ligand_chains

    except Exception as e:
        print(f"Error processing {input_pdb}: {str(e)}")
        return None, None, None, None


def generate_json_files(tasks):
    """Generate AlphaFold3 formatted JSON input files from sequence data."""
    row, cif_dir, output_dir, smiles = tasks
    complex_name = row["complex"]
    chain_sequences = get_chain_sequences_from_row(row)

    # Get ligand chain IDs from the row
    ligand_chain_ids = []
    ligand_cols = [
        col
        for col in row.index
        if col.startswith("ligand_chain_")
    ]
    for col in ligand_cols:
        if pd.notna(row[col]) and int(row[col]) > 0:
            chain_id = col.split("_")[2]
            ligand_chain_ids.append(chain_id)

    if not chain_sequences and not ligand_chain_ids:
        print(f"⚠️ Warning: No valid chain sequences for {complex_name}")
        return None

    sequences = []
    for chain_id, sequence in chain_sequences:
        cif_filename = f"{complex_name}_chain_{chain_id}.cif"
        cif_path = os.path.join(cif_dir, cif_filename)

        if not os.path.exists(cif_path):
            print(f"⚠️ Warning: {cif_filename} not found")
            continue

        # Build the AlphaFold3 JSON structure for the protein component
        sequences.append(
            {
                "protein": {
                    "id": chain_id,
                    "sequence": sequence,
                    "modifications": [],
                    "unpairedMsa": format_msa_sequence(sequence),
                    "pairedMsa": format_msa_sequence(sequence),
                    "templates": [
                        {
                            "mmcifPath": cif_path,
                            "queryIndices": list(range(len(sequence))),
                            "templateIndices": list(range(len(sequence))),
                        }
                    ],
                }
            }
        )

    # Add ligand chains with SMILES
    for chain_id in ligand_chain_ids:
        if not smiles:
            print(
                f"⚠️ Warning: Ligand chain {chain_id} found but no --smiles provided, skipping"
            )
            continue
        sequences.append(
            {
                "ligand": {
                    "id": chain_id,
                    "smiles": smiles,
                }
            }
        )

    if not sequences:
        print(f"⚠️ Warning: No valid sequence data for {complex_name}")
        return None

    json_data = {
        "dialect": "alphafold3",
        "version": 1,
        "name": complex_name,
        "sequences": sequences,
        "modelSeeds": [10],
        "bondedAtomPairs": None,
        "userCCD": None,
    }

    output_filename = f"{complex_name}.json"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w") as f:
        json.dump(json_data, f, indent=2)

    return output_filename


def get_seq_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="input_pdbs")
    parser.add_argument(
        "--output_dir_cif", type=str, default="single_chain_cif"
    )
    parser.add_argument("--save_csv", type=str, default="seq.csv")
    parser.add_argument("--output_dir_json", type=str, default="json")
    parser.add_argument("--batch_dir", type=str, default="batch")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Process count (default CPU-4)",
    )
    parser.add_argument(
        "--num_jobs",
        type=int,
        default=None,
        help="Number of batch groups to create",
    )
    parser.add_argument(
        "--smiles",
        type=str,
        default="",
        help="SMILES string for ligand chains",
    )
    args = parser.parse_args()

    # Environment Setup
    num_workers = (
        args.num_workers if args.num_workers else mp.cpu_count() - 4
    )
    num_jobs = args.num_jobs
    os.makedirs(args.output_dir_cif, exist_ok=True)
    os.makedirs(args.output_dir_json, exist_ok=True)

    print(f"Input Directory: {args.input_dir}")
    print(f"CIF Output Directory: {args.output_dir_cif}")
    print(f"JSON Output Directory: {args.output_dir_json}")

    # Phase 1: Parallel PDB Processing (Extract sequences and split chains)
    pdb_files = list(Path(args.input_dir).glob("*.pdb"))
    process_args = [(str(f), args.output_dir_cif) for f in pdb_files]

    sequences_dict = {}
    with mp.Pool(processes=num_workers) as pool:
        results = list(
            tqdm(
                pool.imap(process_single_pdb, process_args),
                total=len(pdb_files),
                desc="Processing PDBs",
            )
        )

    # Consolidate results into sequence metadata
    for base_name, chain_sequences, total_tokens, ligand_chains in results:
        if base_name is not None:
            sequences_dict[base_name] = {
                "sequences": chain_sequences,
                "length": total_tokens,
                "ligand_chains": ligand_chains,
            }

    # Aggregate all unique protein chain IDs and ligand chain IDs
    all_protein_chain_ids = set()
    all_ligand_chain_ids = set()
    for entry in sequences_dict.values():
        all_protein_chain_ids.update(entry["sequences"].keys())
        all_ligand_chain_ids.update(entry["ligand_chains"].keys())

    all_protein_chain_ids = sorted(all_protein_chain_ids, key=str)
    all_ligand_chain_ids = sorted(all_ligand_chain_ids, key=str)

    # Prepare DataFrame rows
    rows = []
    for complex_name, entry in sequences_dict.items():
        chain_data = entry["sequences"]
        ligand_data = entry["ligand_chains"]
        row = {"complex": complex_name, "total_length": entry["length"]}
        for chain_id in all_protein_chain_ids:
            row[f"chain_{chain_id}_seq"] = chain_data.get(chain_id, "")
        for chain_id in all_ligand_chain_ids:
            row[f"ligand_chain_{chain_id}"] = ligand_data.get(chain_id, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Reorder columns: ID, Length, then specific chain sequences, then ligand info
    cols = ["complex", "total_length"] + [
        c for c in df.columns if c not in ["complex", "total_length"]
    ]
    df = df[cols]
    df.to_csv(args.save_csv, index=False)
    print(f"\n✅ Sequence info saved to {args.save_csv}")

    # Phase 2: Parallel JSON Generation
    json_tasks = [
        (r, args.output_dir_cif, args.output_dir_json, args.smiles)
        for _, r in df.iterrows()
    ]
    with mp.Pool(processes=num_workers) as pool:
        json_results = list(
            tqdm(
                pool.imap(generate_json_files, json_tasks),
                total=len(json_tasks),
                desc="Generating JSONs",
            )
        )

    success_count = sum(1 for r in json_results if r)
    print(
        f"\n✅ Parallel JSON generation complete: {success_count} files created"
    )

    # Phase 3: Batch Partitioning and Symlinking
    batch_json_root = os.path.join(args.batch_dir, "json")
    batch_pdb_root = os.path.join(args.batch_dir, "pdb")

    # Use the length-based splitting logic to balance workloads across jobs
    df_shuffled = df.sample(frac=1).reset_index(drop=True)
    subs = split_by_total_length(df_shuffled, num_jobs)

    print(
        f"📦 Total samples: {len(df_shuffled)}, splitting into {num_jobs} batches."
    )

    for i, sub in enumerate(subs):
        if sub.empty:
            print(f"🔹 Batch {i} is empty, skipping.")
            continue

        max_len = sub["total_length"].max()
        batch_name = f"batch_{i}_{max_len}"
        bd_json = os.path.join(batch_json_root, batch_name)
        bd_pdb = os.path.join(batch_pdb_root, batch_name)

        os.makedirs(bd_json, exist_ok=True)
        os.makedirs(bd_pdb, exist_ok=True)

        count = 0
        for _, r in sub.iterrows():
            cid = r["complex"]
            # Create symbolic links to save space while organizing files into batch folders
            for ext, src_dir, dest_dir in [
                (".pdb", args.input_dir, bd_pdb),
                (".json", args.output_dir_json, bd_json),
            ]:
                src = os.path.abspath(os.path.join(src_dir, f"{cid}{ext}"))
                if os.path.exists(src):
                    dest_path = os.path.join(
                        dest_dir, os.path.basename(src)
                    )
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.symlink(src, dest_path)
            count += 1

        print(f"✅ {batch_name}: contains {count} complexes")

    print(
        f"\n📊 Summary: Processed {len(df)} complexes across {len(os.listdir(batch_json_root))} batches."
    )


if __name__ == "__main__":
    get_seq_main()
