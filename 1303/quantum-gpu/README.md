# Quantum-Classical Hybrid Neural Network

A research-grade project demonstrating the integration of **Variational Quantum Circuits (VQC)** into classical deep learning pipelines using **PennyLane** and **Qiskit Machine Learning**.

> **Target platform:** Windows 10/11 with NVIDIA GeForce MX130 (2 GB VRAM, CUDA compute 5.0).
> Also runs on Linux, macOS (CPU-only or Apple MPS) and any CUDA-capable GPU.

## Project Overview

| Stage | Description | Status |
|-------|-------------|--------|
| 1 | Classical Baseline (Pure PyTorch MLP) | ✅ |
| 2 | Quantum Circuit Layer Integration (PennyLane + Qiskit) | ✅ |
| 3 | Advanced Quantum Features (encodings, ansatzes, expressibility) | ✅ |
| 4 | Experimental Analysis (advantage study, scalability) | ✅ |

## Architecture

```
Input x ∈ ℝᵈ
    │
    ▼
┌────────────────────┐
│ Classical Pre-Net   │  Linear → ReLU → Dropout
│ (d → n_qubits)     │  Maps features to qubit dimension
└────────────────────┘
    │
    ▼
┌────────────────────┐
│                    │
│ ┌────────────────┐ │
│ │ Data Encoding  │ │  Angle / Amplitude / IQP
│ │ |ψ(x)⟩        │ │
│ └────────────────┘ │
│        │           │
│ ┌────────────────┐ │
│ │ Variational    │ │  StronglyEntangling / HardwareEfficient
│ │ Ansatz U(θ)    │ │  Trainable rotation + CNOT layers
│ └────────────────┘ │
│        │           │
│ ┌────────────────┐ │
│ │ Measurement    │ │  ⟨Z₁⟩, ⟨Z₂⟩, ...
│ │ ⟨O⟩            │ │
│ └────────────────┘ │
│                    │
│  Quantum Layer     │
└────────────────────┘
    │
    ▼
┌────────────────────┐
│ Classical Post-Net  │  Linear → ReLU → Logits
│ (n_outputs → C)     │
└────────────────────┘
    │
    ▼
  ŷ ∈ ℝᶜ (class logits)
```

## Quantum Algorithms

### Data Encoding

**Angle Encoding** — one rotation per qubit:

$$|ψ(x)⟩ = \bigotimes_{i=1}^{n} R_Y(x_i)|0⟩$$

**Amplitude Encoding** — encodes $2^n$ values in amplitudes:

$$|ψ(x)⟩ = \sum_{i=0}^{2^n-1} \frac{x_i}{\|x\|} |i⟩$$

### Variational Ansatz

**Strongly Entangling Layers** (Schuld et al.):

$$U(\theta) = \prod_{l=1}^{L} \left[ \text{ENT}_l \cdot \bigotimes_{i=1}^{n} R_Z(\theta_{l,i,3}) R_Y(\theta_{l,i,2}) R_Z(\theta_{l,i,1}) \right]$$

### Cost Function & Gradients

**Parameter-shift rule** for analytic gradients:

$$\frac{\partial}{\partial \theta_i} f(\theta) = \frac{f(\theta_i + \pi/2) - f(\theta_i - \pi/2)}{2}$$

### Expressibility (Sim et al., 2019)

$$\text{Expr} = D_{KL}\left( \hat{P}_{VQC}(F) \| P_{\text{Haar}}(F) \right)$$

where $P_{\text{Haar}}(F) = (2^n - 1)(1 - F)^{2^n - 2}$

---

## Quick Start (Windows)

### Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10 – 3.12 | [python.org](https://www.python.org/downloads/) — check "Add to PATH" |
| NVIDIA Driver | ≥ 452.39 | [nvidia.com/drivers](https://www.nvidia.com/Download/index.aspx) for MX130 |
| CUDA Toolkit | 12.1 (bundled with PyTorch) | No separate install needed |
| Git | any | [git-scm.com](https://git-scm.com/download/win) |

> **MX130 note:** The MX130 has **2 GB VRAM** and CUDA compute capability **5.0** (Maxwell).
> PyTorch CUDA 12.1 wheels support CC ≥ 5.0 so it works out of the box.
> Keep batch sizes small (≤ 32) to stay within VRAM limits.

### 1 — Create a virtual environment

Open **PowerShell** or **Command Prompt**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1       # PowerShell
# or
.\.venv\Scripts\activate.bat       # Command Prompt
```

### 2 — Install PyTorch with CUDA

```powershell
# Option A: one-step helper (PowerShell)
.\run.ps1 install-cuda

# Option B: manual
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 3 — Verify GPU detection

```powershell
.\run.ps1 check-gpu
# Expected output:
#   PyTorch version : 2.x.x+cu121
#   CUDA available  : True
#   GPU device      : NVIDIA GeForce MX130
#   GPU memory      : 2.0 GB
#   Compute cap.    : (5, 0)
```

### 4 — Train

```powershell
# Classical baseline (runs on CUDA)
.\run.ps1 train-classical

# Hybrid quantum-classical (PennyLane)
.\run.ps1 train-pennylane

# Hybrid quantum-classical (Qiskit)
.\run.ps1 train-qiskit

# Circuit analysis
.\run.ps1 analyze

# Run tests
.\run.ps1 test
```

### Analyze CLI options

The project includes a dedicated script for circuit property analysis: `analyze_circuits.py`.
You can run a full suite or a focused run for a single ansatz.

- Full analysis (default behaviour):

```bash
python analyze_circuits.py --n-qubits 4 --max-layers 6 --samples 300
```

- Single-ansatz analysis (focus on one ansatz / entanglement / depth):

```bash
python analyze_circuits.py --n-qubits 4 --ansatz strongly_entangling --entanglement full --depth 3 --samples 300
```

- Common options:
  - `--n-qubits, -q` : number of qubits (default 4)
  - `--max-layers, -l` : maximum circuit depth for scalability runs
  - `--samples, -s` : random parameter samples per estimate
  - `--ansatz, -a` : run a single ansatz (choices: `strongly_entangling`, `hardware_efficient`, `basic_entangler`)
  - `--entanglement, -e` : entanglement pattern when using `--ansatz` (choices: `full`, `linear`, `circular`)
  - `--depth, -d` : depth to use for the single-ansatz run
  - `--output-dir, -o` : where to save plots (default `outputs/circuits`)

Note: when using the PowerShell helper `run.ps1`, arguments must be forwarded explicitly. Example:

```powershell
.
un.ps1 analyze -- --ansatz strongly_entangling --depth 3 --samples 200
```


### Available commands

Run `.\run.ps1 help` for the full list:

| Command | Description |
|---------|-------------|
| `install` | Install dependencies (CPU fallback) |
| `install-cuda` | Install PyTorch CUDA 12.1 + deps |
| `check-gpu` | Show GPU / CUDA info |
| `train-classical` | Stage 1 baseline |
| `train-pennylane` | Stage 2 PennyLane hybrid |
| `train-qiskit` | Stage 2 Qiskit hybrid |
| `analyze` | Stage 3-4 circuit analysis |
| `test` | Run all pytest tests |
| `clean` | Remove outputs & caches |

---

## Quick Start (Linux / macOS)

```bash
# Install dependencies
make install          # CPU torch
make install-cuda     # or PyTorch + CUDA 12.1
make check-gpu        # verify GPU

# Stage 1: Classical baseline
make train-classical

# Stage 2: Hybrid quantum-classical
make train-pennylane
make train-qiskit

# Stage 3-4: Circuit analysis
make analyze

# Run tests
make test
```

## Configuration

All experiments are configured via YAML files in `config/`:

| Config | Framework | Key Features |
|--------|-----------|--------------|
| `classical_baseline.yaml` | PyTorch | MLP, BatchNorm, Dropout |
| `hybrid_pennylane.yaml` | PennyLane | VQC layer, backprop diff |
| `hybrid_qiskit.yaml` | Qiskit ML | EstimatorQNN, parameter-shift |

### Key Hyperparameters

```yaml
quantum:
  n_qubits: 4          # Number of qubits
  n_layers: 3          # VQC depth
  encoding: angle      # Data encoding strategy
  ansatz: strongly_entangling  # Variational circuit
  entanglement: full   # CNOT pattern
```

### MX130-specific tuning tips

| Parameter | Recommended | Why |
|-----------|-------------|-----|
| `batch_size` | 16 – 32 | Fits within 2 GB VRAM |
| `hidden_dims` | `[64, 32, 16]` or smaller | Keep model footprint light |
| `n_qubits` | 4 – 6 | Quantum sim is CPU-bound, not VRAM-bound |
| `pca_components` | 4 | Matches `n_qubits` |

> The quantum circuit simulation always runs on CPU (PennyLane / Qiskit simulators).
> Only the classical pre-net and post-net layers use the MX130 via CUDA.

## Dataset

**Breast Cancer Wisconsin** (sklearn) — binary classification:
- 569 samples, 30 features
- PCA reduction to 4 dimensions (= n_qubits)
- 80/20 train/test split with stratification

## Device Support

| Device | Classical Layers | Quantum Simulation |
|--------|:----------------:|:------------------:|
| NVIDIA CUDA (MX130, GTX, RTX …) | ✅ | ❌ (CPU fallback) |
| Apple MPS (M1/M2/M3) | ✅ | ❌ (CPU fallback) |
| CPU | ✅ | ✅ |

> Quantum simulators always run on CPU. The classical pre/post layers
> are placed on CUDA (or MPS) automatically when `device: "auto"`.

### Device auto-detection priority

```
CUDA available? ─── yes ──▶ cuda
       │
       no
       │
MPS available? ──── yes ──▶ mps
       │
       no
       │
       ▼
      cpu
```

## Project Structure

```
quantum-gpu/
├── config/                     # YAML configurations
├── src/
│   ├── models/                 # Classical, hybrid, quantum layers
│   ├── quantum/                # Circuits, encodings, entanglement
│   ├── training/               # Trainer + quantum-aware training
│   ├── evaluation/             # Metrics, circuit analysis, plots
│   ├── data/                   # Dataset loading & preprocessing
│   └── utils/                  # Config, logging, quantum helpers
├── notebooks/                  # 4 analysis notebooks
├── tests/                      # Pytest test suite
├── train.py                    # CLI training
├── evaluate.py                 # CLI evaluation
├── analyze_circuits.py         # Circuit property analysis
├── run.ps1                     # Windows PowerShell helper
├── Makefile                    # Linux / macOS commands
└── README.md
```

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `classical_baseline.ipynb` | MLP training, confusion matrix |
| 02 | `quantum_circuit_tutorial.ipynb` | Interactive VQC exploration |
| 03 | `hybrid_model_analysis.ipynb` | Classical vs Hybrid comparison |
| 04 | `quantum_advantage_study.ipynb` | Expressibility, barren plateaus, scalability |

## Troubleshooting (Windows)

| Problem | Fix |
|---------|-----|
| `torch.cuda.is_available()` returns `False` | Re-install PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| CUDA out of memory | Reduce `batch_size` to 16 in the YAML config |
| `.\run.ps1` cannot be loaded (execution policy) | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` in PowerShell |
| `make` not found | Use `.\run.ps1` instead, or install Make via `choco install make` |
| PennyLane import error | `pip install pennylane --upgrade` |

## References

1. **Schuld, M., Bocharov, A., Svore, K.** (2020). *Circuit-centric quantum classifiers.* Physical Review A.
2. **Sim, S., Johnson, P.D., Aspuru-Guzik, A.** (2019). *Expressibility and entangling capability of parameterized quantum circuits.* Advanced Quantum Technologies.
3. **Kandala, A., et al.** (2017). *Hardware-efficient variational quantum eigensolver.* Nature.
4. **McClean, J.R., et al.** (2018). *Barren plateaus in quantum neural network training landscapes.* Nature Communications.
5. **Mitarai, K., et al.** (2018). *Quantum circuit learning.* Physical Review A.

## Requirements

- Python 3.10+
- PyTorch ≥ 2.1 (**with CUDA 12.1** for GPU acceleration)
- NVIDIA Driver ≥ 452.39 (for MX130 / Maxwell GPUs)
- PennyLane ≥ 0.35
- Qiskit ≥ 1.0
- scikit-learn ≥ 1.3
