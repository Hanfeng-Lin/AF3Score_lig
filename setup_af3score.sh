#!/usr/bin/env bash
set -euo pipefail

# AF3Score local setup helper
# Usage: ./setup_af3score.sh [env_name]
# Example: ./setup_af3score.sh af3score

ENV_NAME=${1:-af3score}
REPO_ROOT=$(dirname "$(realpath "$0")")
CONDA_BASE=$(conda info --base)

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available in PATH. Please install Miniconda/Anaconda first."
  exit 1
fi

source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists. It will be reused."
else
  echo "Creating conda environment: $ENV_NAME"
  conda create -y -n "$ENV_NAME" python=3.11
fi

conda activate "$ENV_NAME"

echo "Installing required compiler toolchain..."
conda install -y -c conda-forge gxx_linux-64 gxx_impl_linux-64 gcc_linux-64 gcc_impl_linux-64

cd "$REPO_ROOT"

echo "Installing Python dependencies from dev-requirements.txt..."
pip install -r dev-requirements.txt

echo "Installing AF3Score package in editable mode..."
pip install --no-deps -e .

echo "Building AF3Score data helpers..."
if command -v build_data >/dev/null 2>&1; then
  build_data
else
  python -m alphafold3.build_data
fi

echo "Installing additional recommended dependencies..."
conda install -y -c conda-forge biopython h5py pandas

echo
cat <<'EOF'
AF3Score setup is complete.

Next steps:
  1) Activate the environment:
       conda activate $ENV_NAME
  2) Configure AF3score_pipeline.sh with your local PYTHON_EXEC, slurm_partition, and slurm_nodelist if you run on HPC.
  3) Run the pipeline:
       ./AF3score_pipeline.sh <input_pdb_dir> <output_dir> <num_jobs>

If you do not use SLURM, edit AF3score_pipeline.sh to run locally or adapt the submission helper in functions.sh.
EOF
