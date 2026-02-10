#!/usr/bin/env bash
# Create a venv and install required packages for QAOA experiments
set -euo pipefail

VENV_DIR=".venv_qiskit"

if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi

echo "Activating virtualenv: $VENV_DIR"
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

pip install --upgrade pip
pip install -r "$(dirname "$0")/requirements_qaoa.txt"

echo "Environment ready. Activate with: source $VENV_DIR/bin/activate"
