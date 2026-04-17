#!/usr/bin/env bash
set -euo pipefail

# Local AF3Score pipeline runner.
# Usage: ./run_af3score_local.sh <input_pdb_file> <num_jobs> [output_dir] [model_dir] [db_dir] [--enable-data-pipeline]
# Example: ./run_af3score_local.sh input.pdb 4 ~/models ~/public_databases --enable-data-pipeline

if [[ $# -lt 2 ]]; then
  cat <<EOF
Usage: $0 <input_pdb_file> <num_jobs> [output_dir] [model_dir] [db_dir] [--enable-data-pipeline] [--smiles SMILES]

  input_pdb_file: Path to input .pdb file
  num_jobs: Number of batches to create and process
  output_dir: Directory to store AF3Score outputs and metrics (default: same name as input_pdb_file with '_af3score' suffix)
  model_dir: Path to AlphaFold3 weight/model directory (default: /programs/x86_64-linux/alphafold/3.0.1/alphafold3/models)
  db_dir: Path to AF3Score database directory (default: ~/public_databases) - required if --enable-data-pipeline is used
  --enable-data-pipeline: Enable full AF3 data pipeline (MSA/template search) instead of score-only mode
  --smiles SMILES: SMILES string for ligand chains (non-protein residues will be encoded as ligands)
EOF
  exit 1
fi

INPUT_PDB_FILE=$(realpath "$1")
NUM_JOBS="$2"

# Validate input file exists and is a PDB file
if [[ ! -f "$INPUT_PDB_FILE" ]]; then
  echo "ERROR: Input file '$INPUT_PDB_FILE' does not exist"
  exit 1
fi

if [[ "$INPUT_PDB_FILE" != *.pdb ]]; then
  echo "ERROR: Input file must have .pdb extension"
  exit 1
fi

# Parse remaining arguments
shift 2  # Remove first two args
ENABLE_DATA_PIPELINE=false
MODEL_DIR="/programs/x86_64-linux/alphafold/3.0.1/alphafold3/models"
DB_DIR="$HOME/public_databases"
OUTPUT_DIR_PROVIDED=false
LIGAND_SMILES=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --enable-data-pipeline)
      ENABLE_DATA_PIPELINE=true
      echo "Warning: Enabling full data pipeline mode. This requires MSA databases at DB_DIR and will be much slower."
      echo "Make sure DB_DIR contains the required AF3 databases (bfd, uniref90, pdb_mmcif, etc.)"
      shift
      ;;
    --smiles)
      LIGAND_SMILES="$2"
      shift 2
      ;;
    *)
      if [[ "$OUTPUT_DIR_PROVIDED" == false ]]; then
        OUTPUT_DIR=$(realpath "$1")
        OUTPUT_DIR_PROVIDED=true
      elif [[ "$MODEL_DIR" == "/programs/x86_64-linux/alphafold/3.0.1/alphafold3/models" ]]; then
        MODEL_DIR="$1"
      else
        DB_DIR="$1"
      fi
      shift
      ;;
  esac
done

# Set default output directory if not provided
if [[ "$OUTPUT_DIR_PROVIDED" == false ]]; then
  INPUT_BASENAME=$(basename "$INPUT_PDB_FILE" .pdb)
  OUTPUT_DIR=$(realpath "${INPUT_BASENAME}_af3score")
fi

# Create temporary directory with the single PDB file for processing
TEMP_INPUT_DIR="$OUTPUT_DIR/temp_input"
mkdir -p "$TEMP_INPUT_DIR"
cp "$INPUT_PDB_FILE" "$TEMP_INPUT_DIR/"

REPO_ROOT=$(dirname "$(realpath "$0")")
PYTHON_EXEC=${PYTHON_EXEC:-$(command -v python)}

echo "=== AF3Score local runner ==="
echo "Input PDB: $INPUT_PDB_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Num jobs: $NUM_JOBS"
echo "Model dir: $MODEL_DIR"
echo "DB dir: $DB_DIR"
echo "Data pipeline enabled: $ENABLE_DATA_PIPELINE"
echo "Ligand SMILES: ${LIGAND_SMILES:-none}"
echo "Python: $PYTHON_EXEC"

echo "Activating conda environment if available..."
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi

mkdir -p "$OUTPUT_DIR"

AF3_INPUT_BATCH="$OUTPUT_DIR/af3_input_batch"
OUTPUT_DIR_CIF="$OUTPUT_DIR/single_chain_cif"
SAVE_CSV="$OUTPUT_DIR/single_seq.csv"
OUTPUT_DIR_JSON="$OUTPUT_DIR/json"
OUTPUT_DIR_JAX="$AF3_INPUT_BATCH/jax"
OUTPUT_DIR_AF3SCORE="$OUTPUT_DIR/af3score_outputs"
METRIC_CSV="$OUTPUT_DIR/af3score_metrics.csv"

mkdir -p "$AF3_INPUT_BATCH" "$OUTPUT_DIR_CIF" "$OUTPUT_DIR_JSON" "$OUTPUT_DIR_JAX" "$OUTPUT_DIR_AF3SCORE"

# Step 1: prepare inputs
echo "[1/4] Preparing AF3Score inputs..."
SMILES_ARG=""
if [[ -n "$LIGAND_SMILES" ]]; then
  SMILES_ARG="--smiles $LIGAND_SMILES"
fi
$PYTHON_EXEC "$REPO_ROOT/01_prepare_get_json.py" \
  --input_dir "$TEMP_INPUT_DIR" \
  --output_dir_cif "$OUTPUT_DIR_CIF" \
  --save_csv "$SAVE_CSV" \
  --output_dir_json "$OUTPUT_DIR_JSON" \
  --batch_dir "$AF3_INPUT_BATCH" \
  --num_jobs "$NUM_JOBS" \
  $SMILES_ARG

# Step 2: prepare JAX H5 files
echo "[2/4] Generating H5 files for each batch..."
for batch_dir in "$AF3_INPUT_BATCH/pdb"/*; do
  if [[ -d "$batch_dir" ]]; then
    batch_name=$(basename "$batch_dir")
    output_h5_dir="$OUTPUT_DIR_JAX/$batch_name"
    mkdir -p "$output_h5_dir"
    echo "  -> Processing batch: $batch_name"
    $PYTHON_EXEC "$REPO_ROOT/02_prepare_pdb2jax.py" \
      --pdb_folder "$batch_dir" \
      --output_folder "$output_h5_dir" \
      --num_workers 12 \
      $SMILES_ARG
  fi
done

# Step 3: run AF3Score inference
echo "[3/4] Running AF3Score inference for each batch..."
for batch_dir in "$AF3_INPUT_BATCH/json"/*; do
  if [[ -d "$batch_dir" ]]; then
    batch_name=$(basename "$batch_dir")
    echo "  -> Inference batch: $batch_name"
    $PYTHON_EXEC "$REPO_ROOT/run_af3score.py" \
      --db_dir="$DB_DIR" \
      --model_dir="$MODEL_DIR" \
      --batch_json_dir="$batch_dir" \
      --batch_h5_dir="$OUTPUT_DIR_JAX/$batch_name" \
      --output_dir="$OUTPUT_DIR_AF3SCORE" \
      --run_data_pipeline=$ENABLE_DATA_PIPELINE \
      --run_inference=True \
      --init_guess=True \
      --num_samples=1 \
      --buckets="$(basename "$batch_dir" | grep -oE '[0-9]+$' || echo 1)" \
      --write_cif_model=False \
      --write_summary_confidences=True \
      --write_full_confidences=True \
      --write_best_model_root=False \
      --write_ranking_scores_csv=False \
      --write_terms_of_use_file=False \
      --write_fold_input_json_file=False
  fi
done

# Step 4: extract metrics
echo "[4/4] Extracting AF3Score metrics..."
$PYTHON_EXEC "$REPO_ROOT/04_get_metrics.py" \
  --input_pdb_dir "$TEMP_INPUT_DIR" \
  --af3score_output_dir "$OUTPUT_DIR_AF3SCORE" \
  --save_metric_csv "$METRIC_CSV"

echo "\nAF3Score local run finished. Metrics saved to: $METRIC_CSV"
