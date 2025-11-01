"""
Day 4 - Part 1: Realistic Noise Simulation & Error Mitigation
Simulate quantum hardware noise and apply mitigation techniques
Estimated time: 2-3 hours
"""


import numpy as np
import json
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
from qiskit.primitives import BackendEstimator

print("=" * 70)
print("DAY 4 - PART 1: NOISE SIMULATION & MITIGATION")
print("=" * 70)

# Load data
with open('h2_classical_results.json', 'r') as f:
    results = json.load(f)
target_energy = results['fci_energy']

with open('h2_hamiltonian.json', 'r') as f:
    ham_data = json.load(f)
hamiltonian = SparsePauliOp(ham_data['pauli_strings'], ham_data['coefficients'])

print(f"\n🎯 Target Energy: {target_energy:.8f} Ha")

# Build ansatz
def build_hardware_efficient_ansatz(num_qubits, num_layers):
    circuit = QuantumCircuit(num_qubits)

    # Parameter vector
    params = ParameterVector('θ', num_qubits * (num_layers + 1))

    idx = 0

    # Initial single-qubit rotations
    for q in range(num_qubits):
        circuit.ry(params[idx], q)
        idx += 1

    # Repeated entangling layers
    for _ in range(num_layers):
        for q in range(num_qubits - 1):
            circuit.cx(q, q + 1)

        for q in range(num_qubits):
            circuit.ry(params[idx], q)
            idx += 1

    return circuit, list(params)


ansatz, parameters = build_hardware_efficient_ansatz(4, 2)

print("\n" + "=" * 70)
print("STEP 1: Create Realistic Noise Model")
print("=" * 70)

def create_noise_model(error_1q=0.001, error_2q=0.01, readout_error=0.05):
    """
    Create a noise model simulating real quantum hardware
    
    Args:
        error_1q: Single-qubit gate error rate (default: 0.1%)
        error_2q: Two-qubit gate error rate (default: 1%)
        readout_error: Measurement error rate (default: 5%)
    
    Returns:
        NoiseModel object
    """
    noise_model = NoiseModel()
    
    # Single-qubit gate errors (depolarizing)
    error_gate1 = depolarizing_error(error_1q, 1)
    noise_model.add_all_qubit_quantum_error(error_gate1, ['ry', 'rz', 'h'])
    
    # Two-qubit gate errors (depolarizing)
    error_gate2 = depolarizing_error(error_2q, 2)
    noise_model.add_all_qubit_quantum_error(error_gate2, ['cx'])
    
    # Readout errors
    # Format: [[P(measure 0|state 0), P(measure 1|state 0)],
    #          [P(measure 0|state 1), P(measure 1|state 1)]]
    prob_meas0_prep1 = readout_error
    prob_meas1_prep0 = readout_error
    readout_matrix = [[1 - prob_meas1_prep0, prob_meas1_prep0],
                      [prob_meas0_prep1, 1 - prob_meas0_prep1]]
    readout_err = ReadoutError(readout_matrix)
    noise_model.add_all_qubit_readout_error(readout_err)
    
    return noise_model

# Create noise models with different levels
noise_levels = {
    'ideal': None,
    'low_noise': create_noise_model(0.0005, 0.005, 0.01),
    'medium_noise': create_noise_model(0.001, 0.01, 0.05),
    'high_noise': create_noise_model(0.005, 0.03, 0.10)
}

print("✓ Created noise models:")
print("  - Ideal: No noise")
print("  - Low: 0.05% 1Q, 0.5% 2Q, 1% readout")
print("  - Medium: 0.1% 1Q, 1% 2Q, 5% readout (typical NISQ)")
print("  - High: 0.5% 1Q, 3% 2Q, 10% readout")

print("\n" + "=" * 70)
print("STEP 2: Energy Measurement with Noise")
print("=" * 70)

def compute_energy_with_shots(params_values, hamiltonian, ansatz, parameters, 
                              backend, shots=1024):
    """
    Compute energy using shot-based simulation (includes noise)
    """
    # Bind parameters
    bound_circuit = ansatz.bind_parameters({p: v for p, v in zip(parameters, params_values)})

    
    # Use BackendEstimator for shot-based simulation
    estimator = BackendEstimator(backend=backend, options={'shots': shots})
    
    # Compute expectation value
    job = estimator.run(bound_circuit, hamiltonian)
    result = job.result()
    energy = result.values[0]
    
    return energy

print("Testing energy measurement with different noise levels...")

# Test parameters (use good values from Day 3 if available)
np.random.seed(42)
test_params = np.random.uniform(-np.pi, np.pi, len(parameters))

test_results = {}
for noise_name, noise_model in noise_levels.items():
    if noise_name == 'ideal':
        # Use statevector for ideal case
        bound_circuit = ansatz.assign_parameters(dict(zip(parameters, test_params)))
        statevector = Statevector(bound_circuit)
        energy = statevector.expectation_value(hamiltonian).real
        backend = None
    else:
        # Use noisy simulator
        backend = AerSimulator(noise_model=noise_model)
        energy = compute_energy_with_shots(test_params, hamiltonian, ansatz, 
                                          parameters, backend, shots=4096)
    
    test_results[noise_name] = energy
    error = abs(energy - target_energy)
    error_pct = abs((energy - target_energy) / target_energy * 100)
    
    print(f"\n  {noise_name:15s}: E = {energy:.8f} Ha")
    print(f"  {'':15s}  Error = {error:.8f} Ha ({error_pct:.4f}%)")

print("\n" + "=" * 70)
print("STEP 3: VQE with Different Noise Levels")
print("=" * 70)

vqe_noise_results = {}

for noise_name, noise_model in noise_levels.items():
    print(f"\n{'=' * 70}")
    print(f"Running VQE with {noise_name} noise")
    print(f"{'=' * 70}")
    
    # Setup backend
    if noise_name == 'ideal':
        backend = None
        use_shots = False
    else:
        backend = AerSimulator(noise_model=noise_model)
        use_shots = True
    
    energy_history = []
    iteration_count = [0]
    
    def objective(params):
        if use_shots:
            energy = compute_energy_with_shots(params, hamiltonian, ansatz, 
                                              parameters, backend, shots=2048)
        else:
            bound_circuit = ansatz.assign_parameters(dict(zip(parameters, params)))
            statevector = Statevector(bound_circuit)
            energy = statevector.expectation_value(hamiltonian).real
        
        energy_history.append(energy)
        iteration_count[0] += 1
        
        if iteration_count[0] % 20 == 0:
            print(f"  Iter {iteration_count[0]:3d}: E = {energy:.8f} Ha")
        
        return energy
    
    # Initialize
    np.random.seed(42)
    initial_params = np.random.uniform(-np.pi, np.pi, len(parameters))
    
    start_time = time.time()
    
    # Run VQE (fewer iterations for noisy cases)
    max_iters = 200 if noise_name == 'ideal' else 150
    
    result = minimize(
        objective,
        initial_params,
        method='COBYLA',
        options={'maxiter': max_iters, 'rhobeg': 1.0}
    )
    
    elapsed_time = time.time() - start_time
    
    final_energy = result.fun
    error = abs(final_energy - target_energy)
    error_pct = abs((final_energy - target_energy) / target_energy * 100)
    
    vqe_noise_results[noise_name] = {
        'final_energy': float(final_energy),
        'error': float(error),
        'error_percent': float(error_pct),
        'iterations': int(iteration_count[0]),
        'time': float(elapsed_time),
        'energy_history': [float(e) for e in energy_history]
    }
    
    print(f"\n✓ {noise_name} Results:")
    print(f"  Final energy:   {final_energy:.8f} Ha")
    print(f"  Error:          {error:.8f} Ha ({error_pct:.4f}%)")
    print(f"  Iterations:     {iteration_count[0]}")
    print(f"  Time:           {elapsed_time:.1f} seconds")

print("\n" + "=" * 70)
print("STEP 4: Noise Impact Analysis")
print("=" * 70)

print(f"\n{'Noise Level':<15} {'Final Energy':>15} {'Error (%)':>12} {'Degradation':>15}")
print("-" * 65)

ideal_error = vqe_noise_results['ideal']['error_percent']
for noise_name in ['ideal', 'low_noise', 'medium_noise', 'high_noise']:
    res = vqe_noise_results[noise_name]
    degradation = res['error_percent'] - ideal_error
    print(f"{noise_name:<15} {res['final_energy']:>15.8f} {res['error_percent']:>12.4f} "
          f"{degradation:>14.4f}%")

print("\n📊 Noise Impact:")
medium_degradation = vqe_noise_results['medium_noise']['error_percent'] - ideal_error
high_degradation = vqe_noise_results['high_noise']['error_percent'] - ideal_error

print(f"  Medium noise: +{medium_degradation:.4f}% error")
print(f"  High noise:   +{high_degradation:.4f}% error")

print("\n" + "=" * 70)
print("STEP 5: Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Convergence with different noise levels
ax1 = axes[0, 0]
colors = {'ideal': 'blue', 'low_noise': 'green', 
          'medium_noise': 'orange', 'high_noise': 'red'}

for noise_name in ['ideal', 'low_noise', 'medium_noise', 'high_noise']:
    history = vqe_noise_results[noise_name]['energy_history']
    ax1.plot(range(len(history)), history, linewidth=2, 
            label=noise_name.replace('_', ' ').title(),
            color=colors[noise_name], alpha=0.8)

ax1.axhline(target_energy, color='black', linestyle='--', linewidth=2,
            label=f'Target: {target_energy:.6f}')
ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax1.set_ylabel('Energy (Hartree)', fontsize=12, fontweight='bold')
ax1.set_title('VQE Convergence: Noise Impact', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Final error comparison
ax2 = axes[0, 1]
noise_names = ['ideal', 'low_noise', 'medium_noise', 'high_noise']
errors = [vqe_noise_results[n]['error_percent'] for n in noise_names]
colors_bar = ['blue', 'green', 'orange', 'red']

bars = ax2.bar(range(len(noise_names)), errors, color=colors_bar, 
              alpha=0.7, edgecolor='black', linewidth=2)
ax2.axhline(1.0, color='darkgreen', linestyle='--', linewidth=2, 
            label='1% target', alpha=0.7)

ax2.set_xticks(range(len(noise_names)))
ax2.set_xticklabels([n.replace('_', '\n') for n in noise_names], fontsize=10)
ax2.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
ax2.set_title('Final Error vs Noise Level', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

for i, err in enumerate(errors):
    ax2.text(i, err + 0.2, f'{err:.2f}%', ha='center', fontsize=9, fontweight='bold')

# Plot 3: Error evolution (log scale)
ax3 = axes[1, 0]
for noise_name in ['ideal', 'low_noise', 'medium_noise', 'high_noise']:
    history = vqe_noise_results[noise_name]['energy_history']
    errors_abs = [abs(e - target_energy) for e in history]
    ax3.semilogy(range(len(errors_abs)), errors_abs, linewidth=2,
                label=noise_name.replace('_', ' ').title(),
                color=colors[noise_name], alpha=0.8)

ax3.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax3.set_ylabel('Absolute Error (Hartree)', fontsize=12, fontweight='bold')
ax3.set_title('Error Convergence (Log Scale)', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, which='both')

# Plot 4: Noise degradation quantification
ax4 = axes[1, 1]
noise_labels = ['Low', 'Medium', 'High']
degradations = [
    vqe_noise_results['low_noise']['error_percent'] - ideal_error,
    vqe_noise_results['medium_noise']['error_percent'] - ideal_error,
    vqe_noise_results['high_noise']['error_percent'] - ideal_error
]

bars = ax4.bar(noise_labels, degradations, color=['green', 'orange', 'red'],
              alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_ylabel('Additional Error (%)', fontsize=12, fontweight='bold')
ax4.set_title('Noise-Induced Error Degradation', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for i, deg in enumerate(degradations):
    ax4.text(i, deg + 0.1, f'+{deg:.2f}%', ha='center', 
            fontsize=10, fontweight='bold')

plt.suptitle('Day 4: VQE Performance Under Realistic Noise', 
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('day4_noise_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: day4_noise_analysis.png")
plt.show()

# Save results
noise_analysis = {
    'target_energy': target_energy,
    'vqe_results': vqe_noise_results,
    'noise_models': {
        'low': {'1q_error': 0.0005, '2q_error': 0.005, 'readout': 0.01},
        'medium': {'1q_error': 0.001, '2q_error': 0.01, 'readout': 0.05},
        'high': {'1q_error': 0.005, '2q_error': 0.03, 'readout': 0.10}
    }
}

with open('day4_noise_analysis.json', 'w') as f:
    json.dump(noise_analysis, f, indent=2)

print("✓ Saved: day4_noise_analysis.json")

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

print(f"\n1. Ideal Performance:")
print(f"   Error: {ideal_error:.4f}%")

print(f"\n2. Realistic Hardware (Medium Noise):")
print(f"   Error: {vqe_noise_results['medium_noise']['error_percent']:.4f}%")
print(f"   Degradation: +{medium_degradation:.4f}%")

if vqe_noise_results['medium_noise']['error_percent'] < 5.0:
    print(f"   ✅ Still achieves good accuracy despite noise!")
else:
    print(f"   ⚠️  Significant degradation - mitigation needed")

print(f"\n3. Noise Sources:")
print(f"   - Gate errors: Depolarizing noise on gates")
print(f"   - Readout errors: Measurement bit flips")
print(f"   - Both contribute to increased final error")

print("\n" + "=" * 70)
print("DAY 4 - PART 1 COMPLETE!")
print("=" * 70)
print("\n✅ Achievements:")
print("  - Created realistic noise models")
print("  - Tested VQE under different noise levels")
print("  - Quantified noise impact on accuracy")
print("  - Identified typical NISQ performance")
print("\n🔜 Next: Error mitigation techniques!")