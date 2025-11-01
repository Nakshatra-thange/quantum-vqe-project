"""
Day 3 - Part 2: Compare Different Optimizers (Fixed)
- Ensures parameters come from the ansatz (no mismatches)
- Uses Statevector.from_instruction for energy eval
- Collects energy_history per optimizer for plotting
"""

import numpy as np
import json
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator

print("=" * 70)
print("DAY 3 - PART 2: OPTIMIZER COMPARISON (FIXED)")
print("=" * 70)

# Load previous results (target energy)
with open('h2_classical_results.json', 'r') as f:
    results = json.load(f)
target_energy = results['fci_energy']

# Load Hamiltonian (robust parsing)
with open('h2_hamiltonian.json', 'r') as f:
    ham_data = json.load(f)

try:
    hamiltonian = SparsePauliOp(ham_data['pauli_strings'], ham_data['coefficients'])
except Exception:
    try:
        hamiltonian = SparsePauliOp.from_list(list(zip(ham_data['pauli_strings'], ham_data['coefficients'])))
    except Exception as e:
        raise RuntimeError("Couldn't parse h2_hamiltonian.json") from e

num_qubits = hamiltonian.num_qubits
print(f"\n🎯 Target Energy: {target_energy:.8f} Ha")
print(f"✓ Hamiltonian loaded ({num_qubits} qubits)\n")

# ---------------------------
# Build ansatz (HEA) properly
# ---------------------------
def build_hardware_efficient_ansatz(num_qubits, num_layers):
    qc = QuantumCircuit(num_qubits)
    params = []

    # initial RY layer
    for q in range(num_qubits):
        p = Parameter(f"θ_{len(params)}")
        qc.ry(p, q)
        params.append(p)

    # repeated entangle + RY layers
    for _ in range(num_layers):
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        for q in range(num_qubits):
            p = Parameter(f"θ_{len(params)}")
            qc.ry(p, q)
            params.append(p)

    return qc, params

ansatz, _ = build_hardware_efficient_ansatz(num_qubits, 2)

# Extract parameters directly from the circuit (guarantees match)
parameters = list(ansatz.parameters)
parameters.sort(key=lambda p: p.name)  # stable order

print(f"✓ Ansatz built with {len(parameters)} parameters (stable ordering).")
print(f"  Circuit depth: {ansatz.depth()}\n")

# ---------------------------
# Energy evaluation function
# ---------------------------
sim = AerSimulator(method="statevector")

def compute_energy(params_values):
    if len(params_values) != len(parameters):
        raise ValueError(f"Parameter length mismatch: got {len(params_values)}, expected {len(parameters)}")
    # Bind using explicit dict
    bind_dict = {p: float(v) for p, v in zip(parameters, params_values)}
    bound = ansatz.assign_parameters(bind_dict)
    # Use Statevector for expectation
    sv = Statevector.from_instruction(bound)
    energy = sv.expectation_value(hamiltonian).real
    return float(energy)

# ---------------------------
# Comparison loop over optimizers
# ---------------------------
optimizers = ['COBYLA', 'SLSQP', 'Powell', 'Nelder-Mead']
optimizer_results = {}

# reproducible start
np.random.seed(42)
initial_params = np.random.uniform(-np.pi, np.pi, len(parameters))
initial_energy = compute_energy(initial_params)

print("\n" + "=" * 70)
print("Testing Different Optimizers")
print("=" * 70)
print(f"\n📊 Common starting point:")
print(f"  Initial energy: {initial_energy:.8f} Ha")
print(f"  Target energy:  {target_energy:.8f} Ha")
print(f"  Initial error:  {abs(initial_energy - target_energy):.8f} Ha\n")

for optimizer_name in optimizers:
    print(f"\n{'=' * 70}")
    print(f"Running VQE with {optimizer_name}")
    print(f"{'=' * 70}")

    energy_history = []
    iteration_count = [0]

    def objective(x):
        e = compute_energy(x)
        energy_history.append(e)
        iteration_count[0] += 1
        # occasional progress print
        if iteration_count[0] % 20 == 0:
            print(f"  Iter {iteration_count[0]:3d}: E = {e:.8f} Ha")
        return e

    # run optimization (some methods accept maxiter in options, others ignore)
    start_time = time.time()
    opts = {'maxiter': 200}
    # SLSQP benefits from specifying eps or bounds if needed; we keep defaults here
    result = minimize(objective, initial_params.copy(), method=optimizer_name, options=opts)
    elapsed = time.time() - start_time

    final_energy = float(result.fun)
    error = abs(final_energy - target_energy)
    error_pct = abs((final_energy - target_energy) / target_energy * 100)

    optimizer_results[optimizer_name] = {
        'final_energy': final_energy,
        'error': error,
        'error_percent': error_pct,
        'iterations': int(iteration_count[0]),
        'time': float(elapsed),
        'converged': bool(result.success),
        'energy_history': energy_history
    }

    print(f"\n✓ {optimizer_name} Results:")
    print(f"  Final energy:   {final_energy:.8f} Ha")
    print(f"  Error:          {error:.8f} Ha ({error_pct:.4f}%)")
    print(f"  Iterations:     {iteration_count[0]}")
    print(f"  Time:           {elapsed:.2f} seconds")
    print(f"  Converged:      {result.success}")

# ---------------------------
# Summary & plotting
# ---------------------------
print("\n" + "=" * 70)
print("COMPARISON SUMMARY")
print("=" * 70)

print(f"\n{'Optimizer':<15} {'Final Energy':>15} {'Error (%)':>12} {'Iters':>8} {'Time (s)':>10} {'Success':>10}")
print("-" * 80)
for name in optimizers:
    r = optimizer_results[name]
    print(f"{name:<15} {r['final_energy']:>15.8f} {r['error_percent']:>12.4f} {r['iterations']:>8d} {r['time']:>10.2f} {str(r['converged']):>10}")

best_opt = min(optimizer_results.items(), key=lambda x: x[1]['error'])
print(f"\n🏆 Best Optimizer: {best_opt[0]}")
print(f"   Error: {best_opt[1]['error_percent']:.4f}%")

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Convergence curves
ax1 = axes[0, 0]
for name in optimizers:
    hist = optimizer_results[name]['energy_history']
    if len(hist) > 0:
        ax1.plot(range(len(hist)), hist, linewidth=2, label=name, marker='o', markersize=3, markevery=max(1, len(hist)//20))
ax1.axhline(target_energy, color='red', linestyle='--', linewidth=2, label=f'Target: {target_energy:.6f}')
ax1.set_xlabel('Iteration'); ax1.set_ylabel('Energy (Hartree)'); ax1.set_title('Convergence Comparison'); ax1.legend(); ax1.grid(True, alpha=0.3)

# Error evolution (log scale)
ax2 = axes[0, 1]
for name in optimizers:
    hist = optimizer_results[name]['energy_history']
    if len(hist) > 0:
        errs = [abs(e - target_energy) for e in hist]
        ax2.semilogy(range(len(errs)), errs, linewidth=2, label=name)
ax2.set_xlabel('Iteration'); ax2.set_ylabel('Absolute Error (Hartree)'); ax2.set_title('Error Convergence (Log Scale)'); ax2.legend(); ax2.grid(True, alpha=0.3, which='both')

# Final error comparison
ax3 = axes[1, 0]
opt_names = list(optimizer_results.keys())
errors_pct = [optimizer_results[n]['error_percent'] for n in opt_names]
colors = ['green' if e < 1.0 else 'orange' if e < 5.0 else 'red' for e in errors_pct]
ax3.bar(opt_names, errors_pct, color=colors, alpha=0.8, edgecolor='black')
ax3.axhline(1.0, color='green', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_ylabel('Error (%)'); ax3.set_title('Final Error Comparison'); ax3.grid(axis='y', alpha=0.3)
for i, v in enumerate(errors_pct):
    ax3.text(i, v + 0.05, f"{v:.2f}%", ha='center', fontsize=9, fontweight='bold')

# Efficiency: error vs time
ax4 = axes[1, 1]
times = [optimizer_results[n]['time'] for n in opt_names]
ax4.scatter(times, errors_pct, s=120, c=range(len(opt_names)), cmap='viridis', edgecolors='black')
for i, name in enumerate(opt_names):
    ax4.annotate(name, (times[i], errors_pct[i]), xytext=(5,5), textcoords='offset points')
ax4.axhline(1.0, color='green', linestyle='--', linewidth=1)
ax4.set_xlabel('Computation Time (s)'); ax4.set_ylabel('Error (%)'); ax4.set_title('Efficiency: Error vs Time'); ax4.grid(True, alpha=0.3)

plt.suptitle('Day 3: Optimizer Comparison for VQE', fontsize=14)
plt.tight_layout(rect=[0,0,1,0.96])
plt.savefig('day3_optimizer_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: day3_optimizer_comparison.png")
plt.show()

# Save JSON results
comparison_summary = {
    'target_energy': target_energy,
    'initial_energy': initial_energy,
    'optimizers': optimizer_results,
    'best_optimizer': best_opt[0],
    'best_error_percent': best_opt[1]['error_percent']
}
with open('day3_optimizer_comparison.json', 'w') as f:
    json.dump(comparison_summary, f, indent=2, default=float)

print("✓ Saved: day3_optimizer_comparison.json\n")
print("=" * 70)
print("OPTIMIZER ANALYSIS COMPLETE")
print("=" * 70)
