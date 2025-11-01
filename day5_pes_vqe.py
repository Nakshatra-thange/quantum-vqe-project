"""
Day 5 - Part 2: VQE POTENTIAL ENERGY SURFACE (fixed)
- Ensures ansatz parameter count matches circuit parameters exactly
- Validates parameter lengths before binding
- Uses warm-start (previous geometry) safely
- Uses a simple 4-qubit approximate JW H2 Hamiltonian (as in your original)
"""

import numpy as np
import json
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# classical chemistry (FCI for reference)
from pyscf import gto, scf, fci

# qiskit
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

print("=" * 70)
print("DAY 5 - PART 2: VQE POTENTIAL ENERGY SURFACE (FIXED)")
print("=" * 70)

# -------------------------
# Ansatz builder (correct)
# -------------------------
def build_hardware_efficient_ansatz(num_qubits: int, num_layers: int):
    """
    Hardware-efficient ansatz with parameter count calculated consistently:
      num_params = num_qubits * (num_layers + 1)  <-- single initial RY + one RY per layer
    Note: your previous code used mismatched formulas. This function returns:
      (QuantumCircuit, parameters_list)
    """
    qc = QuantumCircuit(num_qubits)
    # Use the correct param-count formula matching how we insert rotation layers:
    num_params = num_qubits * (num_layers + 1)
    params = ParameterVector('θ', num_params)
    idx = 0

    # initial RY layer
    for q in range(num_qubits):
        qc.ry(params[idx], q)
        idx += 1

    # for each layer: entangle then RY layer
    for _ in range(num_layers):
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        for q in range(num_qubits):
            qc.ry(params[idx], q)
            idx += 1

    # return qc and stable list of Parameter objects
    return qc, list(params)

# -------------------------
# Hamiltonian (approx JW)
# -------------------------
def jordan_wigner_h2_hamiltonian():
    """4-qubit approximate JW H2 Hamiltonian from your original mapping"""
    pauli_dict = {
        'IIII': -0.81054, 'IIIZ': 0.17218, 'IIZI': -0.22575,
        'IZII': 0.17218, 'ZIII': -0.22575, 'IIZZ': 0.12091,
        'IZIZ': 0.16868, 'IZZI': 0.04532, 'ZZII': 0.12091,
        'IIXX': 0.04532, 'IIYY': 0.04532, 'IXIX': 0.16614,
        'IYIY': 0.16614, 'XIXI': 0.04532, 'YIYI': 0.04532,
    }
    return SparsePauliOp(list(pauli_dict.keys()), list(pauli_dict.values()))

# -------------------------
# Energy evaluation helper
# -------------------------
def compute_energy(params_values, hamiltonian, ansatz, parameters):
    """
    Bind parameters safely and compute exact statevector expectation value.
    Validates length and raises informative error if mismatch.
    """
    if len(params_values) != len(parameters):
        raise ValueError(f"Parameter length mismatch: got {len(params_values)}, expected {len(parameters)}")

    bind_dict = {p: float(v) for p, v in zip(parameters, params_values)}
    bound_circuit = ansatz.assign_parameters(bind_dict)

    # exact statevector evaluation
    sv = Statevector.from_instruction(bound_circuit)
    energy = float(sv.expectation_value(hamiltonian).real)
    return energy

# -------------------------
# PES scan parameters
# -------------------------
num_qubits = 4
num_layers = 2               # keep same ansatz across geometries
bond_lengths = np.linspace(0.4, 2.5, 15)

print("\nSTEP 1: Define Bond Length Range")
print(f"Scanning {len(bond_lengths)} bond lengths:")
print(f"  Range: {bond_lengths[0]:.2f} - {bond_lengths[-1]:.2f} Å")
print(f"  Estimated time: ~{len(bond_lengths) * 2:.0f} minutes (depends on machine)")

# -------------------------
# Classical reference (FCI)
# -------------------------
print("\nSTEP 2: Compute Classical Reference (FCI)")
fci_energies = []
hf_energies = []
for i, r in enumerate(bond_lengths):
    mol = gto.M(atom=f'H 0 0 0; H 0 0 {r}',
                basis='sto-3g', charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    hf_energies.append(mf.e_tot)

    myci = fci.FCI(mf)
    fci_energy = myci.kernel()[0]
    fci_energies.append(float(fci_energy))

    if (i + 1) % 5 == 0:
        print(f"  Progress: {i+1}/{len(bond_lengths)}")

fci_energies = np.array(fci_energies)
hf_energies = np.array(hf_energies)
min_idx = int(np.argmin(fci_energies))
print("✓ Classical reference complete")
print(f"\nEquilibrium geometry:\n  Bond length: {bond_lengths[min_idx]:.4f} Å\n  FCI energy: {fci_energies[min_idx]:.8f} Ha")

# -------------------------
# Build ansatz once and get parameters
# -------------------------
ansatz, parameters = build_hardware_efficient_ansatz(num_qubits, num_layers)

# IMPORTANT: extract parameters from the circuit to ensure exact match & stable order
parameters = list(ansatz.parameters)
parameters.sort(key=lambda p: p.name)  # stable ordering
num_params = len(parameters)
print(f"\nSTEP 3: VQE setup")
print(f"  Ansatz: {num_layers} layers, {num_qubits} qubits -> {num_params} parameters")
print(f"  Circuit depth: {ansatz.depth()}, CNOTs: {ansatz.count_ops().get('cx', 0)}")

hamiltonian = jordan_wigner_h2_hamiltonian()

# -------------------------
# Run VQE across geometries
# -------------------------
vqe_energies = []
vqe_iterations = []
vqe_times = []

# warm-start initial parameters (random) — will be replaced by previous optimum
np.random.seed(42)
current_params = np.random.uniform(-np.pi, np.pi, num_params)

print("\nSTEP 4: VQE Potential Energy Curve")
for i, r in enumerate(bond_lengths):
    print(f"\n  [{i+1}/{len(bond_lengths)}] R = {r:.3f} Å")

    # energy objective for this geometry
    energy_history = []
    iteration_count = [0]

    def objective(x):
        e = compute_energy(x, hamiltonian, ansatz, parameters)
        energy_history.append(e)
        iteration_count[0] += 1
        # optionally print occasional progress
        if iteration_count[0] % 25 == 0:
            print(f"    Iter {iteration_count[0]}: E = {e:.8f} Ha")
        return e

    start_time = time.time()
    # minimize with reasonable iteration cap to avoid huge runtimes
    result = minimize(objective, current_params, method='COBYLA',
                      options={'maxiter': 120, 'rhobeg': 0.5, 'tol': 1e-6})
    elapsed = time.time() - start_time

    # collect results
    vqe_energies.append(float(result.fun))
    vqe_iterations.append(int(iteration_count[0]))
    vqe_times.append(float(elapsed))

    # warm-start next geometry using optimized parameters (if optimization succeeded)
    if hasattr(result, 'x') and result.x is not None and len(result.x) == num_params:
        current_params = result.x
    else:
        # fallback: small random perturbation around previous
        current_params = np.random.uniform(-0.1, 0.1, num_params)

    # report
    print(f"    VQE energy: {result.fun:.8f} Ha")
    print(f"    FCI energy: {fci_energies[i]:.8f} Ha")
    print(f"    Error: {abs(result.fun - fci_energies[i]):.6f} Ha")
    print(f"    Iterations: {iteration_count[0]}, Time: {elapsed:.1f}s")

print("\n✓ VQE surface complete")

vqe_energies = np.array(vqe_energies)
vqe_min_idx = int(np.argmin(vqe_energies))

# -------------------------
# Error analysis & plotting
# -------------------------
errors = np.abs(vqe_energies - fci_energies)
errors_pct = np.abs((vqe_energies - fci_energies) / fci_energies * 100)

print("\nSTEP 5: Error Analysis")
print(f"  Mean error: {np.mean(errors):.6f} Ha ({np.mean(errors_pct):.3f}%)")
print(f"  Max error: {np.max(errors):.6f} Ha ({np.max(errors_pct):.3f}%)")
print(f"  Min error: {np.min(errors):.6f} Ha ({np.min(errors_pct):.3f}%)")
print(f"  Std dev: {np.std(errors):.6f} Ha")
print(f"  Largest error at R = {bond_lengths[int(np.argmax(errors))]:.3f} Å")

# Plotting (same layout as before)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
ax1 = axes[0, 0]
ax1.plot(bond_lengths, hf_energies, 'o--', label='Hartree-Fock', color='orange', alpha=0.8)
ax1.plot(bond_lengths, fci_energies, 's-', label='FCI (Exact)', color='blue')
ax1.plot(bond_lengths, vqe_energies, '^-', label='VQE', color='green', alpha=0.8)
ax1.plot(bond_lengths[min_idx], fci_energies[min_idx], 'r*', markersize=14, label='FCI min')
ax1.plot(bond_lengths[vqe_min_idx], vqe_energies[vqe_min_idx], 'g*', markersize=12, label='VQE min')
ax1.set_xlabel('H-H Bond Length (Å)')
ax1.set_ylabel('Energy (Hartree)')
ax1.set_title('H₂ Potential Energy Surface')
ax1.legend(); ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
ax2.plot(bond_lengths, errors_pct, 'ro-'); ax2.axhline(1.0, color='green', linestyle='--', label='1% target')
ax2.set_xlabel('H-H Bond Length (Å)'); ax2.set_ylabel('VQE Error (%)')
ax2.set_title('VQE Error vs Geometry'); ax2.grid(True, alpha=0.3); ax2.legend()

ax3 = axes[1, 0]
ax3.plot(bond_lengths, vqe_iterations, 'bo-'); ax3.axhline(np.mean(vqe_iterations), color='red', linestyle='--', label=f'Avg {np.mean(vqe_iterations):.0f}')
ax3.set_xlabel('H-H Bond Length (Å)'); ax3.set_ylabel('VQE Iterations'); ax3.set_title('Convergence Speed'); ax3.grid(True, alpha=0.3); ax3.legend()

ax4 = axes[1, 1]
corr_fci = fci_energies - hf_energies
corr_vqe = vqe_energies - hf_energies
recovery_pct = (corr_vqe / corr_fci) * 100
ax4.plot(bond_lengths, recovery_pct, 'go-'); ax4.axhline(100, color='blue', linestyle='--', label='100%')
ax4.set_xlabel('H-H Bond Length (Å)'); ax4.set_ylabel('Correlation Energy Recovered (%)'); ax4.set_title('VQE Correlation Capture'); ax4.grid(True, alpha=0.3); ax4.legend()

plt.suptitle('Day 5: VQE Potential Energy Surface (fixed)')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('day5_vqe_pes_fixed.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: day5_vqe_pes_fixed.png")
plt.show()

# Save results
pes_data = {
    'bond_lengths': bond_lengths.tolist(),
    'hf_energies': hf_energies.tolist(),
    'fci_energies': fci_energies.tolist(),
    'vqe_energies': vqe_energies.tolist(),
    'vqe_iterations': vqe_iterations,
    'vqe_times': vqe_times,
    'errors': errors.tolist(),
    'errors_percent': errors_pct.tolist(),
}
with open('day5_vqe_pes_fixed.json', 'w') as f:
    json.dump(pes_data, f, indent=2)

print("✓ Saved: day5_vqe_pes_fixed.json")
print("\nDAY 5 - PART 2 COMPLETE (FIXED)")
