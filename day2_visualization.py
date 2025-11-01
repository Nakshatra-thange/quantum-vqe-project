"""
Day 2 - Part 2: Visualization and Analysis
Create publication-quality plots of classical results
Estimated time: 30-45 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from qiskit.quantum_info import SparsePauliOp

# Set publication-quality style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

print("=" * 70)
print("DAY 2 VISUALIZATION & ANALYSIS")
print("=" * 70)

# Load results
with open('h2_classical_results.json', 'r') as f:
    results = json.load(f)

print("\nLoaded classical results:")
print(f"  Exact energy: {results['exact_energy']:.8f} Ha")
print(f"  FCI energy: {results['fci_energy']:.8f} Ha")
print(f"  HF energy: {results['hf_energy']:.8f} Ha")

# Recreate Hamiltonian for analysis
def jordan_wigner_transform_h2():
    pauli_dict = {
        'IIII': -0.81054, 'IIIZ': 0.17218, 'IIZI': -0.22575,
        'IZII': 0.17218, 'ZIII': -0.22575, 'IIZZ': 0.12091,
        'IZIZ': 0.16868, 'IZZI': 0.04532, 'ZZII': 0.12091,
        'IIXX': 0.04532, 'IIYY': 0.04532, 'IXIX': 0.16868,
        'IYIY': 0.16868, 'XIXI': 0.04532, 'YIYI': 0.04532,
    }
    pauli_strings = list(pauli_dict.keys())
    coeffs = list(pauli_dict.values())
    return SparsePauliOp(pauli_strings, coeffs)

qubit_hamiltonian = jordan_wigner_transform_h2()
hamiltonian_matrix = qubit_hamiltonian.to_matrix()
eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)
ground_state_vector = eigenvectors[:, 0]

# Create figure with subplots
fig = plt.figure(figsize=(16, 10))

# ============================================================
# PLOT 1: Energy Level Diagram
# ============================================================
ax1 = plt.subplot(2, 3, 1)

# Plot energy levels
energy_levels = eigenvalues[:8].real  # First 8 states
x_positions = np.zeros(len(energy_levels))

for i, energy in enumerate(energy_levels):
    color = 'red' if i == 0 else 'blue'
    linewidth = 3 if i == 0 else 1.5
    ax1.hlines(energy, -0.3, 0.3, colors=color, linewidth=linewidth)
    ax1.text(0.35, energy, f'E[{i}] = {energy:.4f}', 
             fontsize=9, va='center')

# Highlight ground state
ax1.hlines(energy_levels[0], -0.3, 0.3, colors='red', linewidth=3, 
           label='Ground State')

# Mark energy gap
gap = energy_levels[1] - energy_levels[0]
ax1.annotate('', xy=(0.7, energy_levels[1]), xytext=(0.7, energy_levels[0]),
            arrowprops=dict(arrowstyle='<->', color='green', lw=2))
ax1.text(0.75, (energy_levels[0] + energy_levels[1])/2, 
         f'Gap\n{gap:.3f} Ha', fontsize=9, color='green')

ax1.set_xlim(-0.5, 1.5)
ax1.set_ylabel('Energy (Hartree)', fontsize=12, fontweight='bold')
ax1.set_title('Energy Level Diagram', fontsize=13, fontweight='bold')
ax1.set_xticks([])
ax1.grid(True, alpha=0.3)
ax1.legend()

# ============================================================
# PLOT 2: Ground State Wavefunction
# ============================================================
ax2 = plt.subplot(2, 3, 2)

basis_labels = [format(i, '04b') for i in range(16)]
probabilities = np.abs(ground_state_vector)**2

colors = ['red' if p > 0.1 else 'blue' for p in probabilities]
bars = ax2.bar(range(16), probabilities, color=colors, alpha=0.7, edgecolor='black')

ax2.set_xlabel('Basis State |q₃q₂q₁q₀⟩', fontsize=11, fontweight='bold')
ax2.set_ylabel('Probability', fontsize=11, fontweight='bold')
ax2.set_title('Ground State Wavefunction', fontsize=13, fontweight='bold')
ax2.set_xticks(range(16))
ax2.set_xticklabels(basis_labels, rotation=45, ha='right', fontsize=8)
ax2.grid(True, alpha=0.3, axis='y')

# Annotate significant states
for i, (label, prob) in enumerate(zip(basis_labels, probabilities)):
    if prob > 0.05:
        ax2.text(i, prob + 0.02, f'{prob:.2f}', ha='center', fontsize=8)

# ============================================================
# PLOT 3: Hamiltonian Structure (Sparsity Pattern)
# ============================================================
ax3 = plt.subplot(2, 3, 3)

# Create sparsity pattern
sparse_matrix = np.abs(hamiltonian_matrix) > 1e-10
ax3.spy(sparse_matrix, markersize=3, color='blue')
ax3.set_title('Hamiltonian Matrix Sparsity', fontsize=13, fontweight='bold')
ax3.set_xlabel('Column Index', fontsize=11)
ax3.set_ylabel('Row Index', fontsize=11)

# Add text info
nonzero = np.count_nonzero(sparse_matrix)
total = sparse_matrix.size
sparsity = 100 * (1 - nonzero/total)
ax3.text(0.95, 0.95, f'Sparsity: {sparsity:.1f}%\nNonzero: {nonzero}/{total}',
         transform=ax3.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=9)

# ============================================================
# PLOT 4: Pauli Term Contributions
# ============================================================
ax4 = plt.subplot(2, 3, 4)

# Get Pauli terms and coefficients
pauli_labels = [p.to_label() for p in qubit_hamiltonian.paulis]
coeffs = [c.real for c in qubit_hamiltonian.coeffs]

# Sort by absolute value
sorted_indices = np.argsort(np.abs(coeffs))[::-1]
top_n = 12
top_indices = sorted_indices[:top_n]

top_labels = [pauli_labels[i] for i in top_indices]
top_coeffs = [coeffs[i] for i in top_indices]

colors = ['red' if c > 0 else 'blue' for c in top_coeffs]
bars = ax4.barh(range(top_n), top_coeffs, color=colors, alpha=0.7, edgecolor='black')

ax4.set_yticks(range(top_n))
ax4.set_yticklabels(top_labels, fontsize=9, family='monospace')
ax4.set_xlabel('Coefficient Value', fontsize=11, fontweight='bold')
ax4.set_title(f'Top {top_n} Pauli Term Contributions', fontsize=13, fontweight='bold')
ax4.axvline(0, color='black', linewidth=0.8)
ax4.grid(True, alpha=0.3, axis='x')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='red', alpha=0.7, label='Positive'),
                   Patch(facecolor='blue', alpha=0.7, label='Negative')]
ax4.legend(handles=legend_elements, loc='lower right')

# ============================================================
# PLOT 5: Energy Comparison
# ============================================================
ax5 = plt.subplot(2, 3, 5)

methods = ['HF', 'FCI\n(Exact)', 'Exact\nDiag']
energies = [results['hf_energy'], results['fci_energy'], results['exact_energy']]
colors_methods = ['orange', 'green', 'red']

bars = ax5.bar(methods, energies, color=colors_methods, alpha=0.7, 
               edgecolor='black', linewidth=2)

ax5.set_ylabel('Energy (Hartree)', fontsize=11, fontweight='bold')
ax5.set_title('Classical Methods Comparison', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels
for i, (method, energy) in enumerate(zip(methods, energies)):
    ax5.text(i, energy + 0.002, f'{energy:.6f}', ha='center', fontsize=9)

# Add correlation energy annotation
corr_energy = results['correlation_energy']
ax5.annotate('', xy=(0, results['hf_energy']), xytext=(0, results['fci_energy']),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax5.text(0.15, (results['hf_energy'] + results['fci_energy'])/2, 
         f'Correlation\n{corr_energy:.4f} Ha', fontsize=8, color='purple')

# ============================================================
# PLOT 6: VQE Target Accuracy
# ============================================================
ax6 = plt.subplot(2, 3, 6)

exact = results['exact_energy']
error_thresholds = [0.1, 1.0, 5.0, 10.0]  # Percentage errors
target_energies = [exact * (1 + err/100) for err in error_thresholds]

ax6.hlines(exact, 0, 5, colors='green', linewidth=3, label='Exact Energy', linestyles='solid')

for i, (err, target) in enumerate(zip(error_thresholds, target_energies)):
    if target > exact:  # Only plot higher energies
        ax6.hlines(target, 0, 5, colors='red', linewidth=1.5, 
                   linestyles='--', alpha=0.7)
        ax6.text(5.1, target, f'{err:.1f}% error', fontsize=8, va='center')

# Shade acceptable region (< 1% error)
acceptable_upper = exact * 1.01
ax6.fill_between([0, 5], exact, acceptable_upper, alpha=0.2, color='green',
                 label='< 1% Error (Target)')

ax6.set_xlim(0, 6)
ax6.set_ylim(exact - 0.005, exact * 1.12)
ax6.set_ylabel('Energy (Hartree)', fontsize=11, fontweight='bold')
ax6.set_title('VQE Accuracy Targets', fontsize=13, fontweight='bold')
ax6.set_xticks([])
ax6.legend(loc='upper right')
ax6.grid(True, alpha=0.3, axis='y')

# Add target box
target_1pct = exact * 0.01
ax6.text(0.5, exact + 0.002, 
         f'Target: E = {exact:.6f} ± {abs(target_1pct):.6f} Ha\n(< 1% error)',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
         fontsize=9, fontweight='bold')

# ============================================================
# Final adjustments
# ============================================================
plt.suptitle('Day 2: Classical Benchmark Analysis for H₂ Molecule', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
plt.savefig('day2_classical_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: day2_classical_analysis.png")

plt.show()

# ============================================================
# Print detailed statistics
# ============================================================
print("\n" + "=" * 70)
print("DETAILED STATISTICS")
print("=" * 70)

print(f"\n📊 Energy Analysis:")
print(f"  Hartree-Fock:     {results['hf_energy']:.8f} Ha")
print(f"  FCI (PySCF):      {results['fci_energy']:.8f} Ha")
print(f"  Exact Diag:       {results['exact_energy']:.8f} Ha")
print(f"  Correlation:      {results['correlation_energy']:.8f} Ha ({abs(results['correlation_energy']/results['hf_energy'])*100:.2f}%)")
print(f"  Energy gap:       {results['energy_gap']:.8f} Ha ({results['energy_gap']*27.2114:.4f} eV)")

print(f"\n🎯 VQE Targets (1% error):")
error_threshold = abs(results['exact_energy'] * 0.01)
print(f"  Target energy:    {results['exact_energy']:.8f} Ha")
print(f"  Acceptable range: [{results['exact_energy'] - error_threshold:.8f}, "
      f"{results['exact_energy'] + error_threshold:.8f}] Ha")
print(f"  Maximum error:    ±{error_threshold:.8f} Ha")

print(f"\n🔬 Hamiltonian Properties:")
print(f"  Number of qubits:  {results['num_qubits']}")
print(f"  Number of terms:   {results['num_pauli_terms']}")
print(f"  Matrix size:       {2**results['num_qubits']}×{2**results['num_qubits']}")
print(f"  Sparsity:          {sparsity:.2f}%")

print(f"\n📈 Ground State Properties:")
dominant_states = [(i, p) for i, p in enumerate(probabilities) if p > 0.05]
print(f"  Dominant basis states (> 5% probability):")
for idx, prob in dominant_states:
    label = format(idx, '04b')
    print(f"    |{label}⟩: {prob:.4f} ({prob*100:.1f}%)")

print(f"\n  Total probability: {np.sum(probabilities):.8f} (should be 1.0)")
print(f"  Entropy: {-np.sum(probabilities * np.log2(probabilities + 1e-10)):.4f} bits")

print("\n" + "=" * 70)
print("DAY 2 COMPLETE!")
print("=" * 70)
print("\n✅ You now have:")
print("  1. Exact ground state energy (benchmark)")
print("  2. Complete Hamiltonian (qubit operators)")
print("  3. Energy spectrum visualization")
print("  4. Target accuracy defined (< 1% error)")
print("\n🚀 Ready for Day 3: VQE Implementation!")
print("  Tomorrow you'll build the quantum circuit and optimize it!")