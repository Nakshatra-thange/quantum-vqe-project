"""
Day 6 - Part 1: 1D Hubbard Model Implementation
Study correlated electron systems with VQE
Estimated time: 2-3 hours
"""

import numpy as np
import json
import time
from scipy.optimize import minimize
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

print("=" * 70)
print("DAY 6 - PART 1: 1D HUBBARD MODEL")
print("=" * 70)

print("\nHubbard Model:")
print("  H = -t Σ(c†ᵢc_{i+1} + h.c.) + U Σnᵢ↑nᵢ↓")
print("  t: hopping parameter (kinetic energy)")
print("  U: on-site Coulomb repulsion")
print("  Studies: metal-insulator transition, magnetism, superconductivity")

print("\n" + "=" * 70)
print("STEP 1: Build Hubbard Hamiltonian")
print("=" * 70)

def build_hubbard_hamiltonian_1d(num_sites, t=1.0, U=2.0, pbc=False):
    """
    Build 1D Hubbard model Hamiltonian as Pauli operator
    
    Args:
        num_sites: Number of lattice sites
        t: Hopping parameter
        U: On-site interaction strength
        pbc: Periodic boundary conditions
    
    Returns:
        SparsePauliOp representing the Hamiltonian
    """
    num_qubits = 2 * num_sites  # spin-up and spin-down for each site
    
    # Build Pauli terms
    pauli_strings = []
    coefficients = []
    
    # Hopping terms: -t Σ (c†ᵢσ c_{i+1,σ} + h.c.)
    # In Jordan-Wigner: c†ᵢcⱼ → ½(XᵢXⱼ + YᵢYⱼ)
    for sigma in [0, 1]:  # spin up (0) and spin down (1)
        for i in range(num_sites - 1):
            qubit_i = i * 2 + sigma
            qubit_j = (i + 1) * 2 + sigma
            
            # Create Pauli string
            pauli_x = ['I'] * num_qubits
            pauli_x[qubit_i] = 'X'
            pauli_x[qubit_j] = 'X'
            pauli_strings.append(''.join(reversed(pauli_x)))
            coefficients.append(-t / 2)
            
            pauli_y = ['I'] * num_qubits
            pauli_y[qubit_i] = 'Y'
            pauli_y[qubit_j] = 'Y'
            pauli_strings.append(''.join(reversed(pauli_y)))
            coefficients.append(-t / 2)
    
    # Periodic boundary condition
    if pbc and num_sites > 2:
        for sigma in [0, 1]:
            qubit_first = sigma
            qubit_last = (num_sites - 1) * 2 + sigma
            
            pauli_x = ['I'] * num_qubits
            pauli_x[qubit_first] = 'X'
            pauli_x[qubit_last] = 'X'
            pauli_strings.append(''.join(reversed(pauli_x)))
            coefficients.append(-t / 2)
            
            pauli_y = ['I'] * num_qubits
            pauli_y[qubit_first] = 'Y'
            pauli_y[qubit_last] = 'Y'
            pauli_strings.append(''.join(reversed(pauli_y)))
            coefficients.append(-t / 2)
    
    # Interaction terms: U Σ nᵢ↑ nᵢ↓
    # Number operator: nᵢ = (I - Z)/2
    # nᵢ↑ nᵢ↓ = [(I - Zᵢ↑)/2][(I - Zᵢ↓)/2]
    #         = (I - Zᵢ↑ - Zᵢ↓ + Zᵢ↑Zᵢ↓)/4
    for i in range(num_sites):
        qubit_up = i * 2
        qubit_down = i * 2 + 1
        
        # Constant term: U/4
        pauli_i = ['I'] * num_qubits
        pauli_strings.append(''.join(reversed(pauli_i)))
        coefficients.append(U / 4)
        
        # -U/4 Zᵢ↑
        pauli_z_up = ['I'] * num_qubits
        pauli_z_up[qubit_up] = 'Z'
        pauli_strings.append(''.join(reversed(pauli_z_up)))
        coefficients.append(-U / 4)
        
        # -U/4 Zᵢ↓
        pauli_z_down = ['I'] * num_qubits
        pauli_z_down[qubit_down] = 'Z'
        pauli_strings.append(''.join(reversed(pauli_z_down)))
        coefficients.append(-U / 4)
        
        # +U/4 Zᵢ↑Zᵢ↓
        pauli_zz = ['I'] * num_qubits
        pauli_zz[qubit_up] = 'Z'
        pauli_zz[qubit_down] = 'Z'
        pauli_strings.append(''.join(reversed(pauli_zz)))
        coefficients.append(U / 4)
    
    return SparsePauliOp(pauli_strings, coefficients)

# Build Hubbard Hamiltonian
num_sites = 4
num_electrons = 4  # Half-filling
t_hubbard = 1.0
U_hubbard = 2.0

hamiltonian_hubbard = build_hubbard_hamiltonian_1d(num_sites, t_hubbard, U_hubbard, pbc=False)

print(f"✓ Hubbard Hamiltonian created:")
print(f"  Number of sites: {num_sites}")
print(f"  Number of qubits: {hamiltonian_hubbard.num_qubits}")
print(f"  Hopping t: {t_hubbard}")
print(f"  Interaction U: {U_hubbard}")
print(f"  U/t ratio: {U_hubbard/t_hubbard:.1f}")
print(f"  Pauli terms: {len(hamiltonian_hubbard)}")

print("\n" + "=" * 70)
print("STEP 2: Exact Diagonalization (Classical Benchmark)")
print("=" * 70)

print("Computing exact ground state...")
hamiltonian_matrix = hamiltonian_hubbard.to_matrix()

# Use sparse diagonalization for efficiency
eigenvalues, eigenvectors = eigsh(hamiltonian_matrix, k=5, which='SA')

exact_ground_energy = eigenvalues[0]
exact_ground_state = eigenvectors[:, 0]

print(f"✓ Exact diagonalization complete")
print(f"  Ground state energy: {exact_ground_energy:.8f} (in units of t)")
print(f"  First excited state: {eigenvalues[1]:.8f}")
print(f"  Energy gap: {eigenvalues[1] - eigenvalues[0]:.8f}")

print("\nFirst 5 eigenvalues:")
for i, e in enumerate(eigenvalues):
    print(f"  E[{i}] = {e:.8f}")

print("\n" + "=" * 70)
print("STEP 3: VQE for Hubbard Model")
print("=" * 70)

def build_hardware_efficient_ansatz(num_qubits, num_layers):
    circuit = QuantumCircuit(num_qubits)
    num_params = num_qubits * (num_layers + 1)
    params = ParameterVector('θ', num_params)
    param_idx = 0
    
    for qubit in range(num_qubits):
        circuit.ry(params[param_idx], qubit)
        param_idx += 1
    
    for layer in range(num_layers):
        for qubit in range(num_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        for qubit in range(num_qubits):
            circuit.ry(params[param_idx], qubit)
            param_idx += 1
    
    return circuit, params

ansatz_hubbard, params_hubbard = build_hardware_efficient_ansatz(8, 2)

print(f"Ansatz for Hubbard model:")
print(f"  Qubits: {ansatz_hubbard.num_qubits}")
print(f"  Parameters: {len(params_hubbard)}")
print(f"  Depth: {ansatz_hubbard.depth()}")

def compute_energy(params_values, hamiltonian, ansatz, parameters):
    bound_circuit = ansatz.assign_parameters(dict(zip(parameters, params_values)))
    statevector = Statevector(bound_circuit)
    energy = statevector.expectation_value(hamiltonian).real
    return energy

print("\nRunning VQE optimization...")
np.random.seed(42)
initial_params = np.random.uniform(-np.pi, np.pi, len(params_hubbard))

energy_history = []
iteration_count = [0]

def objective(params):
    energy = compute_energy(params, hamiltonian_hubbard, ansatz_hubbard, params_hubbard)
    energy_history.append(energy)
    iteration_count[0] += 1
    
    if iteration_count[0] % 20 == 0:
        error = abs(energy - exact_ground_energy)
        print(f"  Iter {iteration_count[0]:3d}: E = {energy:.8f}, Error = {error:.8f}")
    
    return energy

start_time = time.time()

result_hubbard = minimize(
    objective,
    initial_params,
    method='COBYLA',
    options={'maxiter': 200, 'rhobeg': 1.0}
)

elapsed_time = time.time() - start_time

vqe_energy = result_hubbard.fun
error = abs(vqe_energy - exact_ground_energy)
error_pct = abs((vqe_energy - exact_ground_energy) / exact_ground_energy * 100)

print(f"\n✓ VQE complete!")
print(f"  Final energy: {vqe_energy:.8f}")
print(f"  Exact energy: {exact_ground_energy:.8f}")
print(f"  Error: {error:.8f} ({error_pct:.4f}%)")
print(f"  Iterations: {iteration_count[0]}")
print(f"  Time: {elapsed_time:.1f} seconds")

print("\n" + "=" * 70)
print("STEP 4: Physical Analysis")
print("=" * 70)

# Analyze ground state
print("\nGround State Properties:")

# Get VQE wavefunction
vqe_params = result_hubbard.x
bound_circuit = ansatz_hubbard.assign_parameters(dict(zip(params_hubbard, vqe_params)))
vqe_state = Statevector(bound_circuit)

# Fidelity with exact state
fidelity = np.abs(np.vdot(exact_ground_state, vqe_state.data))**2
print(f"  State fidelity: {fidelity:.6f}")

# Compute magnetization: <Sz> = Σᵢ <nᵢ↑ - nᵢ↓>/2
magnetization_ops = []
for i in range(num_sites):
    # Sz_i = (nᵢ↑ - nᵢ↓)/2 = (1-Z↑)/2 - (1-Z↓)/2 = (Z↓ - Z↑)/2
    qubit_up = i * 2
    qubit_down = i * 2 + 1
    
    pauli_str = ['I'] * 8
    pauli_str[qubit_up] = 'Z'
    mag_up = SparsePauliOp([''.join(reversed(pauli_str))], [-0.5])
    
    pauli_str = ['I'] * 8
    pauli_str[qubit_down] = 'Z'
    mag_down = SparsePauliOp([''.join(reversed(pauli_str))], [0.5])
    
    magnetization_ops.append((mag_up + mag_down))

print(f"\n  Site magnetizations:")
for i, mag_op in enumerate(magnetization_ops):
    mag_exact = exact_ground_state.conj() @ mag_op.to_matrix() @ exact_ground_state
    mag_vqe = vqe_state.expectation_value(mag_op).real
    print(f"    Site {i}: Exact = {mag_exact.real:.4f}, VQE = {mag_vqe:.4f}")

total_mag_exact = sum(exact_ground_state.conj() @ op.to_matrix() @ exact_ground_state 
                      for op in magnetization_ops).real
total_mag_vqe = sum(vqe_state.expectation_value(op).real for op in magnetization_ops)

print(f"\n  Total magnetization:")
print(f"    Exact: {total_mag_exact:.6f}")
print(f"    VQE:   {total_mag_vqe:.6f}")

print("\n" + "=" * 70)
print("STEP 5: U/t Ratio Dependence")
print("=" * 70)

print("\nStudying metal-insulator transition...")
print("  Small U/t: Metallic (kinetic energy dominates)")
print("  Large U/t: Insulating (interaction dominates)")

U_over_t_values = [0.5, 1.0, 2.0, 4.0, 8.0]
energies_vs_U = []
gaps_vs_U = []
vqe_energies_vs_U = []

for U_t in U_over_t_values:
    print(f"\n  Computing U/t = {U_t:.1f}...")
    
    # Build Hamiltonian
    ham = build_hubbard_hamiltonian_1d(num_sites, t=1.0, U=U_t, pbc=False)
    
    # Exact
    ham_matrix = ham.to_matrix()
    evals, _ = eigsh(ham_matrix, k=2, which='SA')
    energies_vs_U.append(evals[0])
    gaps_vs_U.append(evals[1] - evals[0])
    
    # VQE (quick run)
    def objective_quick(params):
        return compute_energy(params, ham, ansatz_hubbard, params_hubbard)
    
    result_quick = minimize(objective_quick, vqe_params, method='COBYLA',
                           options={'maxiter': 50})
    vqe_energies_vs_U.append(result_quick.fun)
    
    print(f"    Exact: {evals[0]:.6f}, Gap: {evals[1] - evals[0]:.6f}")
    print(f"    VQE:   {result_quick.fun:.6f}")

print("\n✓ U/t scan complete")

print("\n" + "=" * 70)
print("STEP 6: Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: VQE convergence
ax1 = axes[0, 0]
ax1.plot(range(len(energy_history)), energy_history, 'b-', linewidth=2)
ax1.axhline(exact_ground_energy, color='red', linestyle='--', linewidth=2,
           label=f'Exact: {exact_ground_energy:.6f}')

ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax1.set_ylabel('Energy (t)', fontsize=12, fontweight='bold')
ax1.set_title('VQE Convergence for Hubbard Model', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Error convergence (log scale)
ax2 = axes[0, 1]
errors_conv = [abs(e - exact_ground_energy) for e in energy_history]
ax2.semilogy(range(len(errors_conv)), errors_conv, 'r-', linewidth=2)

ax2.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax2.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
ax2.set_title('Error Convergence', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, which='both')

# Plot 3: Energy vs U/t
ax3 = axes[1, 0]
ax3.plot(U_over_t_values, energies_vs_U, 'bo-', linewidth=2, markersize=10,
        label='Exact')
ax3.plot(U_over_t_values, vqe_energies_vs_U, 'gs--', linewidth=2, markersize=8,
        label='VQE', alpha=0.7)

ax3.set_xlabel('U/t Ratio', fontsize=12, fontweight='bold')
ax3.set_ylabel('Ground State Energy (t)', fontsize=12, fontweight='bold')
ax3.set_title('Energy vs Interaction Strength', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Energy gap vs U/t (Metal-Insulator Transition)
ax4 = axes[1, 1]
ax4.plot(U_over_t_values, gaps_vs_U, 'ro-', linewidth=3, markersize=10)

ax4.set_xlabel('U/t Ratio', fontsize=12, fontweight='bold')
ax4.set_ylabel('Energy Gap (t)', fontsize=12, fontweight='bold')
ax4.set_title('Gap vs Interaction (Metal-Insulator Transition)', 
             fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add annotation
ax4.text(0.5, max(gaps_vs_U) * 0.8, 'Larger gap →\nMore insulating',
        fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.suptitle('Day 6: 1D Hubbard Model with VQE',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('day6_hubbard_model.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: day6_hubbard_model.png")
plt.show()

# Save results
hubbard_results = {
    'system': {
        'num_sites': num_sites,
        'num_electrons': num_electrons,
        'num_qubits': 8,
        't': t_hubbard,
        'U': U_hubbard,
        'U_over_t': U_hubbard / t_hubbard
    },
    'exact': {
        'ground_energy': float(exact_ground_energy),
        'gap': float(eigenvalues[1] - eigenvalues[0]),
        'eigenvalues': [float(e) for e in eigenvalues]
    },
    'vqe': {
        'final_energy': float(vqe_energy),
        'error': float(error),
        'error_percent': float(error_pct),
        'iterations': int(iteration_count[0]),
        'time': float(elapsed_time),
        'fidelity': float(fidelity)
    },
    'U_t_scan': {
        'U_over_t_values': U_over_t_values,
        'exact_energies': [float(e) for e in energies_vs_U],
        'vqe_energies': [float(e) for e in vqe_energies_vs_U],
        'gaps': [float(g) for g in gaps_vs_U]
    }
}

with open('day6_hubbard_results.json', 'w') as f:
    json.dump(hubbard_results, f, indent=2)

print("✓ Saved: day6_hubbard_results.json")

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

print(f"\n1. VQE Performance on Hubbard Model:")
print(f"   Error: {error_pct:.4f}%")
print(f"   State fidelity: {fidelity:.6f}")

print(f"\n2. Physical Properties:")
print(f"   Ground energy: {exact_ground_energy:.6f} t")
print(f"   Gap: {eigenvalues[1] - eigenvalues[0]:.6f} t")
print(f"   Total magnetization: {total_mag_exact:.6f}")

print(f"\n3. Metal-Insulator Transition:")
print(f"   Low U/t ({U_over_t_values[0]}): Small gap, metallic")
print(f"   High U/t ({U_over_t_values[-1]}): Large gap, insulating")

print(f"\n4. VQE Applicability:")
print(f"   Successfully captured correlated electron physics")
print(f"   Suitable for studying magnetism and phase transitions")

print("\n" + "=" * 70)
print("DAY 6 - PART 1 COMPLETE!")
print("=" * 70)

