#!/usr/bin/env bash
set -euo pipefail

# Creates a virtual environment and installs dependencies from requirements.txt
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Setup completato. Attiva l'ambiente con: source .venv/bin/activate"
echo "Nota: CPLEX richiede installazione/licenza separata; non è installato automaticamente."
