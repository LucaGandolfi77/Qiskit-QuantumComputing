# ═══════════════════════════════════════════════════════════
#  Quantum-Classical Hybrid Neural Network — Makefile
# ═══════════════════════════════════════════════════════════

.PHONY: help install train-classical train-pennylane train-qiskit \
        evaluate analyze test test-circuits test-layers test-hybrid \
        tensorboard clean

PYTHON ?= python3

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────
install: ## Install dependencies
	pip install -r requirements.txt

install-cuda: ## Install PyTorch with CUDA 12.1, then deps
	pip install torch --index-url https://download.pytorch.org/whl/cu121
	pip install -r requirements.txt

check-gpu: ## Verify CUDA & GPU detection
	$(PYTHON) -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# ── Training ─────────────────────────────────────────────
train-classical: ## Train classical baseline (Stage 1)
	$(PYTHON) train.py --config config/classical_baseline.yaml

train-pennylane: ## Train hybrid PennyLane model (Stage 2)
	$(PYTHON) train.py --config config/hybrid_pennylane.yaml

train-pennylane-warmup: ## Train PennyLane with quantum warm-up
	$(PYTHON) train.py --config config/hybrid_pennylane.yaml --quantum-warmup

train-qiskit: ## Train hybrid Qiskit model (Stage 2)
	$(PYTHON) train.py --config config/hybrid_qiskit.yaml

# ── Evaluation ───────────────────────────────────────────
evaluate: ## Evaluate best PennyLane checkpoint
	$(PYTHON) evaluate.py \
		--config config/hybrid_pennylane.yaml \
		--checkpoint outputs/models/hybrid_pennylane/checkpoint_best.pt

# ── Analysis ─────────────────────────────────────────────
analyze: ## Run circuit property analysis (Stage 3-4)
	$(PYTHON) analyze_circuits.py --n-qubits 4 --max-layers 6 --samples 300

analyze-deep: ## Deep analysis with more qubits and samples
	$(PYTHON) analyze_circuits.py --n-qubits 6 --max-layers 8 --samples 500

# ── Testing ──────────────────────────────────────────────
test: ## Run all tests
	$(PYTHON) -m pytest tests/ -v --tb=short

test-circuits: ## Test quantum circuits only
	$(PYTHON) -m pytest tests/test_circuits.py -v --tb=short

test-layers: ## Test quantum layers only
	$(PYTHON) -m pytest tests/test_quantum_layers.py -v --tb=short

test-hybrid: ## Test hybrid models & training
	$(PYTHON) -m pytest tests/test_hybrid_models.py -v --tb=short

# ── Monitoring ───────────────────────────────────────────
tensorboard: ## Launch TensorBoard
	tensorboard --logdir outputs/tensorboard --port 6006

# ── Utilities ────────────────────────────────────────────
clean: ## Remove outputs and caches
	rm -rf outputs/models/*/checkpoint_*.pt
	rm -rf outputs/tensorboard/*
	rm -rf outputs/plots/*.png
	rm -rf outputs/circuits/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
