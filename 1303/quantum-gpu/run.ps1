# ═══════════════════════════════════════════════════════════
#  Quantum-Classical Hybrid Neural Network — Windows helper
#  Usage:  .\run.ps1 <command>
# ═══════════════════════════════════════════════════════════

param(
    [Parameter(Position=0)]
    [ValidateSet(
        "help","install","install-cuda","check-gpu",
        "train-classical","train-pennylane","train-pennylane-warmup","train-qiskit",
        "evaluate","analyze","analyze-deep",
        "test","test-circuits","test-layers","test-hybrid",
        "tensorboard","clean"
    )]
    [string]$Command = "help"
)

$ErrorActionPreference = "Stop"

function Show-Help {
    Write-Host ""
    Write-Host "  Available commands:" -ForegroundColor Cyan
    Write-Host "    install              Install dependencies (CPU torch fallback)"
    Write-Host "    install-cuda         Install PyTorch with CUDA 12.1, then deps"
    Write-Host "    check-gpu            Verify CUDA & GPU are detected by PyTorch"
    Write-Host ""
    Write-Host "    train-classical      Train classical baseline (Stage 1)"
    Write-Host "    train-pennylane      Train hybrid PennyLane model (Stage 2)"
    Write-Host "    train-pennylane-warmup  PennyLane with quantum warm-up"
    Write-Host "    train-qiskit         Train hybrid Qiskit model (Stage 2)"
    Write-Host ""
    Write-Host "    evaluate             Evaluate best PennyLane checkpoint"
    Write-Host "    analyze              Circuit property analysis (Stage 3-4)"
    Write-Host "    analyze-deep         Deep analysis (more qubits/samples)"
    Write-Host ""
    Write-Host "    test                 Run all tests"
    Write-Host "    test-circuits        Test quantum circuits only"
    Write-Host "    test-layers          Test quantum layers only"
    Write-Host "    test-hybrid          Test hybrid models & training"
    Write-Host ""
    Write-Host "    tensorboard          Launch TensorBoard"
    Write-Host "    clean                Remove outputs and caches"
    Write-Host ""
}

switch ($Command) {
    "help" {
        Show-Help
    }
    "install" {
        pip install -r requirements.txt
    }
    "install-cuda" {
        Write-Host "Installing PyTorch with CUDA 12.1 support..." -ForegroundColor Yellow
        pip install torch --index-url https://download.pytorch.org/whl/cu121
        Write-Host "Installing remaining dependencies..." -ForegroundColor Yellow
        pip install -r requirements.txt
        Write-Host "Done. Run '.\run.ps1 check-gpu' to verify." -ForegroundColor Green
    }
    "check-gpu" {
        python -c @"
import torch
print(f'PyTorch version : {torch.__version__}')
print(f'CUDA available  : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version    : {torch.version.cuda}')
    print(f'GPU device      : {torch.cuda.get_device_name(0)}')
    prop = torch.cuda.get_device_properties(0)
    # Property name changed between PyTorch versions: prefer total_memory
    mem = getattr(prop, 'total_memory', None) or getattr(prop, 'total_mem', None)
    if mem is not None:
        print(f'GPU memory      : {mem / 1024**3:.1f} GB')
    else:
        print('GPU memory      : <unknown>')
    print(f'Compute cap.    : {torch.cuda.get_device_capability(0)}')
else:
    print('WARNING: CUDA not detected — running on CPU only.')
"@
    }
    "train-classical" {
        python train.py --config config/classical_baseline.yaml
    }
    "train-pennylane" {
        python train.py --config config/hybrid_pennylane.yaml
    }
    "train-pennylane-warmup" {
        python train.py --config config/hybrid_pennylane.yaml --quantum-warmup
    }
    "train-qiskit" {
        python train.py --config config/hybrid_qiskit.yaml
    }
    "evaluate" {
        python evaluate.py `
            --config config/hybrid_pennylane.yaml `
            --checkpoint outputs/models/hybrid_pennylane/checkpoint_best.pt
    }
    "analyze" {
        python analyze_circuits.py --n-qubits 4 --max-layers 6 --samples 300
    }
    "analyze-deep" {
        python analyze_circuits.py --n-qubits 6 --max-layers 8 --samples 500
    }
    "test" {
        python -m pytest tests/ -v --tb=short
    }
    "test-circuits" {
        python -m pytest tests/test_circuits.py -v --tb=short
    }
    "test-layers" {
        python -m pytest tests/test_quantum_layers.py -v --tb=short
    }
    "test-hybrid" {
        python -m pytest tests/test_hybrid_models.py -v --tb=short
    }
    "tensorboard" {
        tensorboard --logdir outputs/tensorboard --port 6006
    }
    "clean" {
        Write-Host "Cleaning outputs and caches..." -ForegroundColor Yellow
        Get-ChildItem -Path "outputs/models" -Recurse -Filter "checkpoint_*.pt" -ErrorAction SilentlyContinue | Remove-Item -Force
        if (Test-Path "outputs/tensorboard") { Remove-Item "outputs/tensorboard/*" -Recurse -Force -ErrorAction SilentlyContinue }
        if (Test-Path "outputs/plots") { Get-ChildItem "outputs/plots" -Filter "*.png" | Remove-Item -Force }
        if (Test-Path "outputs/circuits") { Get-ChildItem "outputs/circuits" -Filter "*.png" | Remove-Item -Force }
        Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force
        Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force
        Write-Host "Done." -ForegroundColor Green
    }
}
