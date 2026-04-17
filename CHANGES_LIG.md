# AF3Score_lig — Changes vs. upstream AF3Score

This fork ([Hanfeng-Lin/AF3Score_lig](https://github.com/Hanfeng-Lin/AF3Score_lig))
extends [Mingchenchen/AF3Score](https://github.com/Mingchenchen/AF3Score) to score
**protein–ligand complexes** (not just protein-only complexes). All modifications
are in the pre-processing and post-processing pipeline scripts; the AF3 model code
is unchanged.

## Summary

| Script | Change |
|---|---|
| `01_prepare_get_json.py` | Detects ligand chains in input PDBs, emits AF3 JSON with `ligand` + SMILES entries. |
| `02_prepare_pdb2jax.py` | Converts ligand chains to token arrays (1 atom = 1 token). |
| `04_get_metrics.py` | Derives chains from AF3's token output; maps AF3-sanitised names back to PDBs. |
| `.gitignore` | Ignores `__pycache__`, `*_af3score/` run dirs, `input/`, and `ccd.pickle`. |

## What's new in detail

### `01_prepare_get_json.py`
- New CLI flag `--smiles <SMILES>` — attached to every detected ligand chain.
- `is_ligand_chain()` / `count_ligand_atoms()`: a chain with no standard amino
  acid residues is treated as a ligand; each atom counts as one AF3 token.
- AF3 JSON now includes `{"ligand": {"id": <chain_id>, "smiles": <SMILES>}}`
  entries alongside protein sequences.
- Output CSV gains `ligand_chain_<ID>` columns (atom counts per ligand chain)
  in addition to the existing `chain_<ID>_seq` columns.
- Batch grouping (`--num_batches`) uses absolute source paths when moving files.

### `02_prepare_pdb2jax.py`
- `structure_to_array(..., include_ligands=True)` adds ligand handling:
  ligand atoms get 1 token each, with the coordinate stored in slot 0 of the
  standard 24-slot atom array (slots 1–23 are zero-padded).
- Protein chains still use the ATOM14 mapping as before.

### `04_get_metrics.py`
- Chain set is taken from AF3's `token_chain_ids` in `confidences.json`, not
  from the input PDB. Required because ligand chains tokenize to atom counts
  rather than residue counts, so the PDB's chain list can disagree with AF3's.
- `sanitise_name()` + `build_pdb_lookup()` map AF3 output subdirectory names
  (lowercased, non-alphanumerics → `_`) back to the original PDB filenames.

## Repo hygiene

- `ccd.pickle` (≈ 504 MB) and `chemical_component_sets.pickle` (≈ 8 KB, derived
  from `ccd.pickle`) are **not** tracked. Rebuild both whenever the CCD is
  refreshed:
  ```bash
  # 1. Rebuild ccd.pickle from RCSB Chemical Component Dictionary
  #    (https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz)
  python -m alphafold3.constants.converters.ccd_pickle_gen \
      /path/to/components.cif.gz \
      src/alphafold3/constants/converters/ccd.pickle

  # 2. Regenerate the derived chemical_component_sets.pickle (reads ccd.pickle)
  python -m alphafold3.constants.converters.chemical_component_sets_gen \
      src/alphafold3/constants/converters/chemical_component_sets.pickle
  ```
- Per-complex AF3Score run outputs live in `*_af3score/` and are gitignored.

## Usage (ligand mode)

```bash
# 1. Prepare JSON + per-chain CIFs, passing the ligand SMILES
python 01_prepare_get_json.py \
    --input_dir input/ \
    --output_dir_cif cif/ \
    --output_dir_json json/ \
    --smiles "CC(=O)Oc1ccccc1C(=O)O"

# 2. Convert PDBs to JAX coord arrays (ligands handled automatically)
python 02_prepare_pdb2jax.py --input_dir input/ --output_dir jax/

# 3. Run AF3Score (unchanged from upstream)
bash AF3score_pipeline.sh ...

# 4. Collect metrics
python 04_get_metrics.py --input_pdb_dir input/ --base_dir <af3score_run_dir>
```
