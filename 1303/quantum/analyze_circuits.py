#!/usr/bin/env python3
"""
Circuit property analysis CLI.

Computes expressibility and entangling capability for different VQC
configurations and produces comparison plots.

Usage::

    python analyze_circuits.py --n-qubits 4 --max-layers 6 --samples 300
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.quantum.circuits import get_weight_shape
from src.evaluation.circuit_analysis import (
    compute_expressibility,
    compute_entangling_capability,
    make_statevector_fn,
)
from src.evaluation.visualization import plot_circuit_properties, plot_scalability


@click.command()
@click.option("--n-qubits", "-q", type=int, default=4, help="Number of qubits.")
@click.option("--max-layers", "-l", type=int, default=5, help="Maximum circuit depth.")
@click.option("--samples", "-s", type=int, default=200, help="Random samples per estimate.")
@click.option("--output-dir", "-o", type=str, default="outputs/circuits", help="Output directory.")
def main(n_qubits, max_layers, samples, output_dir):
    """Analyze VQC expressibility and entangling capability."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ansatz_types = ["strongly_entangling", "hardware_efficient", "basic_entangler"]
    entanglement_patterns = ["full", "linear", "circular"]

    # ── 1. Compare ansatzes at fixed depth ───────────────
    click.echo(f"\n═══ Circuit Analysis: {n_qubits} qubits ═══\n")

    results: dict[str, dict[str, float]] = {}

    for ansatz in ansatz_types:
        for ent in entanglement_patterns:
            name = f"{ansatz[:8]}_{ent[:4]}"
            click.echo(f"Analysing {name}...", nl=False)

            try:
                n_layers = 3
                w_shape = get_weight_shape(ansatz, n_qubits, n_layers)
                sv_fn = make_statevector_fn(n_qubits, n_layers, ansatz, ent)

                expr = compute_expressibility(sv_fn, n_qubits, w_shape, n_samples=samples)
                ent_cap = compute_entangling_capability(sv_fn, n_qubits, w_shape, n_samples=samples)

                results[name] = {
                    "expressibility": expr,
                    "entangling_capability": ent_cap,
                }
                click.echo(f"  expr={expr:.4f}  ent={ent_cap:.4f}")
            except Exception as e:
                click.echo(f"  SKIPPED ({e})")

    if results:
        plot_circuit_properties(
            results,
            title=f"Circuit Properties ({n_qubits} qubits, depth=3)",
            save_path=output_path / "circuit_comparison.png",
        )

    # ── 2. Depth vs expressibility ───────────────────────
    click.echo(f"\n═══ Depth Analysis (strongly_entangling, full) ═══\n")

    depths = list(range(1, max_layers + 1))
    depth_expr, depth_ent = [], []

    for d in depths:
        w_shape = get_weight_shape("strongly_entangling", n_qubits, d)
        sv_fn = make_statevector_fn(n_qubits, d, "strongly_entangling", "full")

        expr = compute_expressibility(sv_fn, n_qubits, w_shape, n_samples=samples)
        ent_cap = compute_entangling_capability(sv_fn, n_qubits, w_shape, n_samples=samples)
        depth_expr.append(expr)
        depth_ent.append(ent_cap)
        click.echo(f"  depth={d}: expr={expr:.4f}  ent={ent_cap:.4f}")

    plot_scalability(
        depths,
        {"expressibility (KL ↓)": depth_expr, "entangling_cap (Q ↑)": depth_ent},
        title="Expressibility vs Circuit Depth",
        save_path=output_path / "depth_analysis.png",
    )

    # ── 3. Qubit scalability ─────────────────────────────
    click.echo(f"\n═══ Qubit Scalability (depth=3) ═══\n")

    qubit_range = list(range(2, min(n_qubits + 3, 9)))
    qubit_expr, qubit_ent = [], []

    for nq in qubit_range:
        w_shape = get_weight_shape("strongly_entangling", nq, 3)
        sv_fn = make_statevector_fn(nq, 3, "strongly_entangling", "full")

        expr = compute_expressibility(sv_fn, nq, w_shape, n_samples=samples // 2)
        ent_cap = compute_entangling_capability(sv_fn, nq, w_shape, n_samples=samples // 2)
        qubit_expr.append(expr)
        qubit_ent.append(ent_cap)
        click.echo(f"  qubits={nq}: expr={expr:.4f}  ent={ent_cap:.4f}")

    plot_scalability(
        qubit_range,
        {"expressibility (KL ↓)": qubit_expr, "entangling_cap (Q ↑)": qubit_ent},
        title="Scalability: Qubits vs Circuit Properties",
        save_path=output_path / "qubit_scalability.png",
    )

    click.echo(f"\n✓ Analysis complete — results saved to {output_path}")


if __name__ == "__main__":
    main()
