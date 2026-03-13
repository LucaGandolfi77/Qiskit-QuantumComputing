# Quantum-Classical Hybrid Neural Network

A research-grade project demonstrating the integration of **Variational Quantum Circuits (VQC)** into classical deep learning pipelines using **PennyLane** and **Qiskit Machine Learning**.

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

## Quick Start

```bash
# Install dependencies
make install

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

## Dataset

**Breast Cancer Wisconsin** (sklearn) — binary classification:
- 569 samples, 30 features
- PCA reduction to 4 dimensions (= n_qubits)
- 80/20 train/test split with stratification

## Device Support

| Device | Classical Layers | Quantum Simulation |
|--------|:----------------:|:------------------:|
| CPU | ✅ | ✅ |
| Apple MPS (M1/M2) | ✅ | ❌ (CPU fallback) |
| NVIDIA CUDA | ✅ | ❌ (CPU fallback) |

> Quantum simulators always run on CPU. The classical pre/post layers
> can leverage MPS or CUDA for acceleration.

## Project Structure

```
quantum/
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
├── Makefile                    # Common commands
└── README.md
```

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `classical_baseline.ipynb` | MLP training, confusion matrix |
| 02 | `quantum_circuit_tutorial.ipynb` | Interactive VQC exploration |
| 03 | `hybrid_model_analysis.ipynb` | Classical vs Hybrid comparison |
| 04 | `quantum_advantage_study.ipynb` | Expressibility, barren plateaus, scalability |

## References

1. **Schuld, M., Bocharov, A., Svore, K.** (2020). *Circuit-centric quantum classifiers.* Physical Review A.
2. **Sim, S., Johnson, P.D., Aspuru-Guzik, A.** (2019). *Expressibility and entangling capability of parameterized quantum circuits.* Advanced Quantum Technologies.
3. **Kandala, A., et al.** (2017). *Hardware-efficient variational quantum eigensolver.* Nature.
4. **McClean, J.R., et al.** (2018). *Barren plateaus in quantum neural network training landscapes.* Nature Communications.
5. **Mitarai, K., et al.** (2018). *Quantum circuit learning.* Physical Review A.

## Requirements

- Python 3.10+
- PyTorch ≥ 2.1
- PennyLane ≥ 0.35
- Qiskit ≥ 1.0
- scikit-learn ≥ 1.3
