"""
Day 3 - Part 3: Ansatz Depth Study
Compare VQE performance with different numbers of layers
Estimated time: 1 hour
"""

import numpy as np
import json
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector

print("=" * 70)
print("DAY 3 - PART 3: ANSATZ DEPTH STUDY")
print("=" * 70)

# Load data
with open('h2_classical_results.json', 'r') as f:
    results = json.load(f)
target_energy = results['fci_energy']

with open('h2_hamiltonian.json', 'r') as f:
    ham_data = json.load(f)
hamiltonian = SparsePauliOp(ham_data['pauli_strings'], ham_data['coefficients'])

print(f"\n🎯 Target Energy: {target_energy:.8f} Ha")

def build_hardware_efficient_ansatz(num_qubits, num_layers):
    circuit = QuantumCircuit(num_qubits)
    num_params = num_qubits * (num_layers + 1)  # ✅ FIXED
    params = ParameterVector('θ', num_params)
    param_idx = 0
    
    # Initial single-qubit rotations
    for qubit in range(num_qubits):
        circuit.ry(params[param_idx], qubit)
        param_idx += 1

    # Each layer
    for _ in range(num_layers):
        # Entangling CX chain
        for qubit in range(num_qubits - 1):
            circuit.cx(qubit, qubit + 1)

        # Another set of RY rotations
        for qubit in range(num_qubits):
            circuit.ry(params[param_idx], qubit)
            param_idx += 1

    return circuit, params


def compute_energy(params_values, hamiltonian, ansatz, parameters):
    bound_circuit = ansatz.assign_parameters(dict(zip(parameters, params_values)))
    statevector = Statevector(bound_circuit)
    energy = statevector.expectation_value(hamiltonian).real
    return energy

print("\n" + "=" * 70)
print("Testing Different Ansatz Depths")
print("=" * 70)

# Test different numbers of layers
num_qubits = 4
layer_counts = [1, 2, 3, 4]
num_runs = 3  # Multiple runs per configuration for statistics

depth_results = {}

for num_layers in layer_counts:
    print(f"\n{'=' * 70}")
    print(f"Testing {num_layers} layer(s)")
    print(f"{'=' * 70}")
    
    ansatz, parameters = build_hardware_efficient_ansatz(num_qubits, num_layers)
    
    print(f"\n📊 Ansatz properties:")
    print(f"  Layers: {num_layers}")
    print(f"  Parameters: {len(parameters)}")
    print(f"  Circuit depth: {ansatz.depth()}")
    print(f"  CNOT count: {ansatz.count_ops().get('cx', 0)}")
    
    # Run multiple times with different initializations
    run_energies = []
    run_errors = []
    run_times = []
    run_iterations = []
    
    for run in range(num_runs):
        print(f"\n  Run {run + 1}/{num_runs}...")
        
        # Random initialization
        np.random.seed(42 + run)  # Different seed each run
        initial_params = np.random.uniform(-np.pi, np.pi, len(parameters))
        
        energy_history = []
        iteration_count = [0]
        
        def objective(params):
            energy = compute_energy(params, hamiltonian, ansatz, parameters)
            energy_history.append(energy)
            iteration_count[0] += 1
            return energy
        
        start_time = time.time()
        
        result = minimize(
            objective,
            initial_params,
            method='COBYLA',
            options={'maxiter': 200, 'tol': 1e-6}
        )
        
        elapsed_time = time.time() - start_time
        
        final_energy = result.fun
        error = abs(final_energy - target_energy)
        error_pct = abs((final_energy - target_energy) / target_energy * 100)
        
        run_energies.append(final_energy)
        run_errors.append(error_pct)
        run_times.append(elapsed_time)
        run_iterations.append(iteration_count[0])
        
        print(f"    Final energy: {final_energy:.8f} Ha")
        print(f"    Error: {error_pct:.4f}%")
        print(f"    Iterations: {iteration_count[0]}, Time: {elapsed_time:.1f}s")
    
    # Calculate statistics
    depth_results[num_layers] = {
        'num_parameters': len(parameters),
        'circuit_depth': ansatz.depth(),
        'cnot_count': ansatz.count_ops().get('cx', 0),
        'energies': run_energies,
        'errors_percent': run_errors,
        'times': run_times,
        'iterations': run_iterations,
        'mean_energy': float(np.mean(run_energies)),
        'std_energy': float(np.std(run_energies)),
        'mean_error': float(np.mean(run_errors)),
        'std_error': float(np.std(run_errors)),
        'mean_time': float(np.mean(run_times)),
        'best_energy': float(min(run_energies)),
        'best_error': float(min(run_errors))
    }
    
    print(f"\n  Statistics ({num_runs} runs):")
    print(f"    Mean error: {depth_results[num_layers]['mean_error']:.4f} ± "
          f"{depth_results[num_layers]['std_error']:.4f}%")
    print(f"    Best error: {depth_results[num_layers]['best_error']:.4f}%")
    print(f"    Mean time: {depth_results[num_layers]['mean_time']:.1f}s")

print("\n" + "=" * 70)
print("DEPTH COMPARISON SUMMARY")
print("=" * 70)

print(f"\n{'Layers':<8} {'Params':<8} {'Depth':<8} {'CNOTs':<8} {'Mean Error (%)':<18} "
      f"{'Best Error (%)':<18} {'Time (s)':<12}")
print("-" * 90)

for num_layers in layer_counts:
    res = depth_results[num_layers]
    print(f"{num_layers:<8} {res['num_parameters']:<8} {res['circuit_depth']:<8} "
          f"{res['cnot_count']:<8} {res['mean_error']:<9.4f}±{res['std_error']:<7.4f} "
          f"{res['best_error']:<18.4f} {res['mean_time']:<12.1f}")

# Find optimal configuration
best_config = min(depth_results.items(), key=lambda x: x[1]['best_error'])
print(f"\n🏆 Best Configuration: {best_config[0]} layer(s)")
print(f"   Best error: {best_config[1]['best_error']:.4f}%")
print(f"   Mean error: {best_config[1]['mean_error']:.4f}%")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))

# Plot 1: Error vs Layers
ax1 = plt.subplot(2, 3, 1)
layers = list(depth_results.keys())
mean_errors = [depth_results[l]['mean_error'] for l in layers]
std_errors = [depth_results[l]['std_error'] for l in layers]
best_errors = [depth_results[l]['best_error'] for l in layers]

ax1.errorbar(layers, mean_errors, yerr=std_errors, marker='o', markersize=10,
            linewidth=2, capsize=5, label='Mean ± Std', color='blue')
ax1.plot(layers, best_errors, 's--', markersize=8, linewidth=2, 
         label='Best', color='green')
ax1.axhline(1.0, color='red', linestyle='--', linewidth=2, 
            alpha=0.5, label='1% target')

ax1.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
ax1.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
ax1.set_title('Accuracy vs Ansatz Depth', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(layers)

# Plot 2: Parameters vs Error
ax2 = plt.subplot(2, 3, 2)
params = [depth_results[l]['num_parameters'] for l in layers]
ax2.scatter(params, mean_errors, s=200, c=layers, cmap='viridis', 
           edgecolors='black', linewidth=2, alpha=0.7)

for i, l in enumerate(layers):
    ax2.annotate(f'{l}L', (params[i], mean_errors[i]), 
                xytext=(5, 5), textcoords='offset points', fontsize=10)

ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax2.set_xlabel('Number of Parameters', fontsize=12, fontweight='bold')
ax2.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
ax2.set_title('Parameters vs Accuracy', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Plot 3: Circuit Depth vs Error
ax3 = plt.subplot(2, 3, 3)
depths = [depth_results[l]['circuit_depth'] for l in layers]
ax3.scatter(depths, mean_errors, s=200, c=layers, cmap='plasma',
           edgecolors='black', linewidth=2, alpha=0.7)

for i, l in enumerate(layers):
    ax3.annotate(f'{l}L', (depths[i], mean_errors[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=10)

ax3.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_xlabel('Circuit Depth', fontsize=12, fontweight='bold')
ax3.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
ax3.set_title('Circuit Depth vs Accuracy', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Plot 4: Computation Time
ax4 = plt.subplot(2, 3, 4)
times = [depth_results[l]['mean_time'] for l in layers]
colors_time = ['green' if e < 1.0 else 'orange' for e in mean_errors]

bars = ax4.bar(layers, times, color=colors_time, alpha=0.7, 
              edgecolor='black', linewidth=2)
ax4.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
ax4.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax4.set_title('Computation Time vs Layers', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_xticks(layers)

for i, (l, t) in enumerate(zip(layers, times)):
    ax4.text(l, t + 1, f'{t:.1f}s', ha='center', fontsize=9)

# Plot 5: Convergence Consistency (error bars)
ax5 = plt.subplot(2, 3, 5)
for l in layers:
    errors = depth_results[l]['errors_percent']
    x_positions = [l] * len(errors)
    ax5.scatter(x_positions, errors, s=100, alpha=0.6)

ax5.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax5.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
ax5.set_ylabel('Error (%) - Individual Runs', fontsize=12, fontweight='bold')
ax5.set_title('Convergence Consistency', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3)
ax5.set_xticks(layers)

# Plot 6: Resource Efficiency
ax6 = plt.subplot(2, 3, 6)
cnots = [depth_results[l]['cnot_count'] for l in layers]

# Create composite score: lower error and fewer CNOTs is better
efficiency_score = [1/(e + 0.01) / (c + 1) * 100 for e, c in zip(mean_errors, cnots)]

bars = ax6.bar(layers, efficiency_score, color='purple', alpha=0.7,
              edgecolor='black', linewidth=2)
ax6.set_xlabel('Number of Layers', fontsize=12, fontweight='bold')
ax6.set_ylabel('Efficiency Score (higher is better)', fontsize=12, fontweight='bold')
ax6.set_title('Resource Efficiency', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')
ax6.set_xticks(layers)

plt.suptitle('Day 3: Ansatz Depth Study for VQE', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('day3_ansatz_depth_study.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: day3_ansatz_depth_study.png")
plt.show()

# Save results
with open('day3_ansatz_depth_study.json', 'w') as f:
    json.dump(depth_results, f, indent=2)

print("✓ Saved: day3_ansatz_depth_study.json")

print("\n" + "=" * 70)
print("CONCLUSIONS")
print("=" * 70)

print("\n📊 Key Findings:")
print(f"\n1. Optimal Depth: {best_config[0]} layer(s)")
print(f"   - Achieves {best_config[1]['best_error']:.4f}% error")
print(f"   - Uses {best_config[1]['num_parameters']} parameters")
print(f"   - Circuit depth: {best_config[1]['circuit_depth']}")

if best_config[1]['best_error'] < 1.0:
    print(f"\n2. ✅ Target Achieved!")
    print(f"   - VQE successfully found ground state within 1% error")
else:
    print(f"\n2. ⚠️  Target Not Reached")
    print(f"   - Best error: {best_config[1]['best_error']:.4f}%")
    print(f"   - Consider: more layers, different optimizer, or better initialization")

print(f"\n3. Trade-offs:")
print(f"   - More layers → Better accuracy but slower computation")
print(f"   - Fewer layers → Faster but may not reach target")
print(f"   - Sweet spot: Balance accuracy and resource requirements")

print(f"\n4. Consistency:")
max_std = max(depth_results[l]['std_error'] for l in layers)
if max_std < 0.5:
    print(f"   - ✅ High consistency across runs (std < 0.5%)")
else:
    print(f"   - ⚠️  Some variability in convergence (max std: {max_std:.4f}%)")

print("\n" + "=" * 70)
print("DAY 3 - PART 3 COMPLETE!")
print("=" * 70)
print("\n🎉 Full Day 3 Achievements:")
print("  ✅ Built and tested VQE with HEA ansatz")
print("  ✅ Compared 4 different optimizers")
print("  ✅ Studied ansatz depth systematically")
print("  ✅ Identified optimal configuration")
print(f"  ✅ Best result: {best_config[1]['best_error']:.4f}% error")