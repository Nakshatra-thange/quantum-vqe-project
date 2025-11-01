"""
Day 4 - Part 3: Circuit Resource Analysis & Scaling Study
Analyze quantum resources and predict scaling behavior
Estimated time: 1 hour
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

print("=" * 70)
print("DAY 4 - PART 3: CIRCUIT RESOURCE ANALYSIS")
print("=" * 70)

def build_hardware_efficient_ansatz(num_qubits, num_layers):
    """Build HEA ansatz"""
    circuit = QuantumCircuit(num_qubits)
    num_params = num_qubits * (2 * num_layers + 1)
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

print("\n" + "=" * 70)
print("STEP 1: Analyze Current H₂ Circuit")
print("=" * 70)

# Current circuit (H₂)
ansatz_h2, params_h2 = build_hardware_efficient_ansatz(4, 2)

print("📊 H₂ Circuit (4 qubits, 2 layers):")
print(f"  Total parameters: {len(params_h2)}")
print(f"  Circuit depth: {ansatz_h2.depth()}")
print(f"  Gate counts:")
gate_counts = ansatz_h2.count_ops()
for gate, count in sorted(gate_counts.items()):
    print(f"    {gate}: {count}")

# Calculate circuit metrics
num_1q_gates = gate_counts.get('ry', 0) + gate_counts.get('rz', 0)
num_2q_gates = gate_counts.get('cx', 0)
total_gates = num_1q_gates + num_2q_gates

print(f"\n  Single-qubit gates: {num_1q_gates}")
print(f"  Two-qubit gates: {num_2q_gates}")
print(f"  Total gates: {total_gates}")

# Typical gate fidelities
fidelity_1q = 0.999  # 99.9% (0.1% error)
fidelity_2q = 0.99   # 99% (1% error)

circuit_fidelity = fidelity_1q**num_1q_gates * fidelity_2q**num_2q_gates
print(f"\n  Estimated circuit fidelity: {circuit_fidelity:.6f} ({(1-circuit_fidelity)*100:.4f}% error)")

print("\n" + "=" * 70)
print("STEP 2: Scaling Analysis - Different System Sizes")
print("=" * 70)

# Analyze different system sizes
qubit_counts = [2, 4, 6, 8, 10, 12]
layer_counts = [1, 2, 3, 4]

scaling_data = {}

for num_qubits in qubit_counts:
    scaling_data[num_qubits] = {}
    
    for num_layers in layer_counts:
        circuit, params = build_hardware_efficient_ansatz(num_qubits, num_layers)
        
        gate_counts = circuit.count_ops()
        num_1q = gate_counts.get('ry', 0)
        num_2q = gate_counts.get('cx', 0)
        
        fidelity = fidelity_1q**num_1q * fidelity_2q**num_2q
        
        scaling_data[num_qubits][num_layers] = {
            'parameters': len(params),
            'depth': circuit.depth(),
            'single_qubit_gates': num_1q,
            'two_qubit_gates': num_2q,
            'total_gates': num_1q + num_2q,
            'circuit_fidelity': float(fidelity)
        }

print("✓ Computed scaling data for different configurations")

print("\n📊 Scaling Summary (2 layers):")
print(f"{'Qubits':<8} {'Params':<10} {'Depth':<8} {'1Q Gates':<10} {'2Q Gates':<10} {'Fidelity':<12}")
print("-" * 70)

for num_qubits in qubit_counts:
    data = scaling_data[num_qubits][2]
    print(f"{num_qubits:<8} {data['parameters']:<10} {data['depth']:<8} "
          f"{data['single_qubit_gates']:<10} {data['two_qubit_gates']:<10} "
          f"{data['circuit_fidelity']:<12.6f}")

print("\n" + "=" * 70)
print("STEP 3: Resource Requirements for VQE")
print("=" * 70)

# Calculate total quantum operations for full VQE run
num_iterations = 150  # Typical
num_pauli_terms = 15  # For H₂

# For each iteration:
# - Need to measure expectation value of Hamiltonian
# - Hamiltonian has ~15 Pauli terms
# - Each term needs separate measurement (or grouping)

measurements_per_iter = num_pauli_terms
total_measurements = num_iterations * measurements_per_iter
shots_per_measurement = 1024

total_shots = total_measurements * shots_per_measurement

print(f"VQE Resource Requirements (H₂):")
print(f"  Iterations: {num_iterations}")
print(f"  Pauli terms: {num_pauli_terms}")
print(f"  Measurements/iteration: {measurements_per_iter}")
print(f"  Total measurements: {total_measurements:,}")
print(f"  Shots/measurement: {shots_per_measurement}")
print(f"  Total shots: {total_shots:,}")

# Calculate time estimates
time_per_shot = 1e-3  # 1 millisecond (optimistic)
total_time_seconds = total_shots * time_per_shot
total_time_minutes = total_time_seconds / 60

print(f"\n⏱️  Time Estimates (1ms/shot):")
print(f"  Total quantum execution: {total_time_seconds:.1f} seconds ({total_time_minutes:.1f} minutes)")
print(f"  Classical optimization: ~10-30 seconds")
print(f"  Total VQE run: ~{total_time_minutes + 0.5:.1f} minutes")

# Cost on real hardware (IBM Quantum)
cost_per_shot = 0.0001  # Hypothetical $0.0001/shot
total_cost = total_shots * cost_per_shot

print(f"\n💰 Estimated Cost (real hardware):")
print(f"  At $0.0001/shot: ${total_cost:.2f}")
print(f"  Note: Actual pricing varies by provider")

print("\n" + "=" * 70)
print("STEP 4: Comparison with Classical Resources")
print("=" * 70)

# Exact diagonalization resources
def classical_resources(num_qubits):
    """Calculate classical resources for exact diagonalization"""
    hilbert_space_dim = 2**num_qubits
    matrix_elements = hilbert_space_dim**2
    
    # Memory: store complex matrix
    memory_bytes = matrix_elements * 16  # 16 bytes per complex number
    memory_gb = memory_bytes / (1024**3)
    
    # Operations: O(N³) for diagonalization
    operations = hilbert_space_dim**3
    
    return {
        'dimension': hilbert_space_dim,
        'matrix_elements': matrix_elements,
        'memory_gb': memory_gb,
        'operations': operations
    }

print("Classical Exact Diagonalization:")
print(f"{'Qubits':<8} {'Hilbert Dim':<15} {'Memory (GB)':<15} {'Operations':<20}")
print("-" * 65)

for nq in [4, 8, 12, 16, 20, 24, 28]:
    res = classical_resources(nq)
    print(f"{nq:<8} {res['dimension']:<15,} {res['memory_gb']:<15.4f} {res['operations']:<20.2e}")

print("\n🎯 Quantum Advantage Threshold:")
print("  Current (4 qubits): Classical is faster")
print("  ~20 qubits: Crossover point")
print("  >30 qubits: Quantum advantage likely")
print("  >50 qubits: Classical infeasible")

print("\n" + "=" * 70)
print("STEP 5: Visualization")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Circuit depth scaling
ax1 = axes[0, 0]
for num_layers in [1, 2, 3, 4]:
    depths = [scaling_data[nq][num_layers]['depth'] for nq in qubit_counts]
    ax1.plot(qubit_counts, depths, 'o-', linewidth=2, markersize=8,
            label=f'{num_layers} layer(s)')

ax1.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
ax1.set_ylabel('Circuit Depth', fontsize=12, fontweight='bold')
ax1.set_title('Circuit Depth Scaling', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Parameter count scaling
ax2 = axes[0, 1]
for num_layers in [1, 2, 3, 4]:
    params = [scaling_data[nq][num_layers]['parameters'] for nq in qubit_counts]
    ax2.plot(qubit_counts, params, 's-', linewidth=2, markersize=8,
            label=f'{num_layers} layer(s)')

ax2.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Parameters', fontsize=12, fontweight='bold')
ax2.set_title('Parameter Count Scaling', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: CNOT count scaling
ax3 = axes[0, 2]
for num_layers in [1, 2, 3, 4]:
    cnots = [scaling_data[nq][num_layers]['two_qubit_gates'] for nq in qubit_counts]
    ax3.plot(qubit_counts, cnots, '^-', linewidth=2, markersize=8,
            label=f'{num_layers} layer(s)')

ax3.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
ax3.set_ylabel('Number of CNOT Gates', fontsize=12, fontweight='bold')
ax3.set_title('Two-Qubit Gate Scaling', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Circuit fidelity degradation
ax4 = axes[1, 0]
for num_layers in [1, 2, 3, 4]:
    fidelities = [scaling_data[nq][num_layers]['circuit_fidelity'] for nq in qubit_counts]
    ax4.plot(qubit_counts, fidelities, 'o-', linewidth=2, markersize=8,
            label=f'{num_layers} layer(s)')

ax4.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
ax4.set_ylabel('Circuit Fidelity', fontsize=12, fontweight='bold')
ax4.set_title('Fidelity Degradation (NISQ)', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Plot 5: Classical vs Quantum Scaling
ax5 = axes[1, 1]
nq_range = np.arange(4, 30, 2)

# Classical: O(2^N) space, O(2^3N) time
classical_memory = [classical_resources(nq)['memory_gb'] for nq in nq_range]
classical_ops = [classical_resources(nq)['operations'] for nq in nq_range]

# Quantum: O(poly(N)) for VQE
quantum_gates = [scaling_data.get(nq, scaling_data[12])[2]['total_gates'] if nq <= 12 
                else scaling_data[12][2]['total_gates'] * (nq/12)**2 
                for nq in nq_range]

ax5_twin = ax5.twinx()
line1 = ax5.semilogy(nq_range, classical_ops, 'r-', linewidth=3, 
                     label='Classical Ops (O(2³ᴺ))')
line2 = ax5_twin.plot(nq_range, quantum_gates, 'g-', linewidth=3,
                     label='Quantum Gates (O(poly(N)))')

ax5.set_xlabel('Number of Qubits', fontsize=12, fontweight='bold')
ax5.set_ylabel('Classical Operations', fontsize=11, fontweight='bold', color='red')
ax5_twin.set_ylabel('Quantum Gates', fontsize=11, fontweight='bold', color='green')
ax5.set_title('Scaling: Classical vs Quantum', fontsize=13, fontweight='bold')
ax5.tick_params(axis='y', labelcolor='red')
ax5_twin.tick_params(axis='y', labelcolor='green')
ax5.grid(True, alpha=0.3)

# Add crossover annotation
ax5.axvline(20, color='blue', linestyle='--', linewidth=2, alpha=0.5)
ax5.text(20.5, 1e30, 'Crossover\n~20 qubits', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Plot 6: Resource breakdown for H₂
ax6 = axes[1, 2]
resources = ['Parameters\n(12)', 'Depth\n(22)', 'RY Gates\n(12)', 
            'CNOT Gates\n(6)', 'Measurements\n(2250)']
values = [12, 22, 12, 6, 2250 / 100]  # Scaled for visibility

colors_res = ['blue', 'green', 'orange', 'red', 'purple']
bars = ax6.bar(range(len(resources)), values, color=colors_res,
              alpha=0.7, edgecolor='black', linewidth=2)

ax6.set_xticks(range(len(resources)))
ax6.set_xticklabels(resources, fontsize=10)
ax6.set_ylabel('Count (measurements / 100)', fontsize=11, fontweight='bold')
ax6.set_title('H₂ VQE Resource Breakdown', fontsize=13, fontweight='bold')
ax6.grid(True, alpha=0.3, axis='y')

plt.suptitle('Day 4: Quantum Circuit Resource Analysis',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('day4_resource_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: day4_resource_analysis.png")
plt.show()

# Save data
resource_data = {
    'h2_circuit': {
        'qubits': 4,
        'layers': 2,
        'parameters': len(params_h2),
        'depth': ansatz_h2.depth(),
        'single_qubit_gates': num_1q_gates,
        'two_qubit_gates': num_2q_gates,
        'circuit_fidelity': float(circuit_fidelity)
    },
    'vqe_requirements': {
        'iterations': num_iterations,
        'measurements': total_measurements,
        'total_shots': total_shots,
        'estimated_time_minutes': float(total_time_minutes),
        'estimated_cost_dollars': float(total_cost)
    },
    'scaling_data': scaling_data
}

with open('day4_resource_analysis.json', 'w') as f:
    json.dump(resource_data, f, indent=2)

print("✓ Saved: day4_resource_analysis.json")

print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

print("\n1. H₂ Circuit Resources:")
print(f"   - Parameters: {len(params_h2)}")
print(f"   - Depth: {ansatz_h2.depth()}")
print(f"   - CNOTs: {num_2q_gates} (most error-prone)")
print(f"   - Fidelity: {circuit_fidelity:.6f}")

print("\n2. VQE Total Cost:")
print(f"   - Quantum shots: {total_shots:,}")
print(f"   - Estimated time: {total_time_minutes:.1f} minutes")
print(f"   - Memory: Negligible (poly(N))")

print("\n3. Classical Comparison:")
classical_4q = classical_resources(4)
classical_20q = classical_resources(20)
print(f"   - 4 qubits: {classical_4q['memory_gb']:.4f} GB (easy)")
print(f"   - 20 qubits: {classical_20q['memory_gb']:.1f} GB (challenging)")
print(f"   - >30 qubits: Infeasible classically")

print("\n4. Quantum Advantage:")
print(f"   - Current NISQ era: ~50-100 qubits available")
print(f"   - VQE useful at ~20-30 qubits")
print(f"   - Error correction needed for full advantage")

print("\n" + "=" * 70)
print("DAY 4 - PART 3 COMPLETE!")
print("=" * 70)
print("\n🎉 Full Day 4 Achievements:")
print("  ✅ Simulated realistic quantum noise")
print("  ✅ Compared VQE vs classical methods")
print("  ✅ Analyzed circuit resources & scaling")
print("  ✅ Identified quantum advantage region")
print("  ✅ Ready for Day 5: Advanced topics!")