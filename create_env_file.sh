#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

cat > .env <<EOL
export PROJECT_ROOT="${SCRIPT_DIR}"
export HYDRA_JOBS="${SCRIPT_DIR}"
export WANDB_DIR="${SCRIPT_DIR}"
EOL

cp .env "${SCRIPT_DIR}/remote/DiffCSP-official"
cp .env "${SCRIPT_DIR}/remote/cdvae"
