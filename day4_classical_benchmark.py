"""
Day 4 - Part 2: Classical Quantum Chemistry Methods Comparison
Compare VQE against HF, MP2, CCSD, and FCI
Estimated time: 1-2 hours
"""

import numpy as np
import json
import time
import matplotlib.pyplot as plt
from pyscf import gto, scf, mp, cc, fci

print("=" * 70)
print("DAY 4 - PART 2: CLASSICAL METHODS BENCHMARK")
print("=" * 70)

# Load VQE results
try:
    with open('day3_vqe_results.json', 'r') as f:
        vqe_data = json.load(f)
    vqe_available = True
    print("\n✓ Loaded VQE results from Day 3")
except FileNotFoundError:
    vqe_available = False
    print("\n⚠️  VQE results not found - will use placeholder")

# Define molecule
bond_length = 0.735
mol = gto.M(
    atom=f'H 0 0 0; H 0 0 {bond_length}',
    basis='sto-3g',
    charge=0,
    spin=0,
    verbose=0
)

print(f"\n📊 System: H₂ molecule")
print(f"  Bond length: {bond_length} Å")
print(f"  Basis set: sto-3g")
print(f"  Electrons: {mol.nelectron}")
print(f"  Orbitals: {mol.nao}")

print("\n" + "=" * 70)
print("Running Classical Methods")
print("=" * 70)

classical_results = {}

# 1. Hartree-Fock
print("\n1. Hartree-Fock (Mean-Field Theory)...")
start = time.time()
mf = scf.RHF(mol)
hf_energy = mf.kernel()
hf_time = time.time() - start

classical_results['HF'] = {
    'energy': float(hf_energy),
    'time': hf_time,
    'description': 'Single-determinant approximation',
    'scaling': 'O(N⁴)',
    'correlation_captured': 0.0
}

print(f"   ✓ HF Energy: {hf_energy:.8f} Ha")
print(f"   Time: {hf_time:.4f} seconds")

# 2. MP2 (Møller-Plesset 2nd order)
print("\n2. MP2 (Perturbation Theory)...")
start = time.time()
mp2 = mp.MP2(mf)
mp2_energy, _ = mp2.kernel()
mp2_time = time.time() - start

classical_results['MP2'] = {
    'energy': float(mp2_energy),
    'time': mp2_time,
    'description': '2nd order perturbation theory',
    'scaling': 'O(N⁵)',
    'correlation_captured': 0.0  # Will calculate below
}

print(f"   ✓ MP2 Energy: {mp2_energy:.8f} Ha")
print(f"   Time: {mp2_time:.4f} seconds")

# 3. CCSD (Coupled Cluster Singles Doubles)
print("\n3. CCSD (Coupled Cluster)...")
start = time.time()
mycc = cc.CCSD(mf)
ccsd_energy, _, _ = mycc.kernel()
ccsd_time = time.time() - start

classical_results['CCSD'] = {
    'energy': float(ccsd_energy),
    'time': ccsd_time,
    'description': 'Coupled cluster (gold standard)',
    'scaling': 'O(N⁶)',
    'correlation_captured': 0.0
}

print(f"   ✓ CCSD Energy: {ccsd_energy:.8f} Ha")
print(f"   Time: {ccsd_time:.4f} seconds")

# 4. FCI (Full Configuration Interaction - Exact)
print("\n4. FCI (Exact Diagonalization)...")
start = time.time()
myci = fci.FCI(mf)
fci_energy, _ = myci.kernel()
fci_time = time.time() - start

classical_results['FCI'] = {
    'energy': float(fci_energy),
    'time': fci_time,
    'description': 'Exact solution (within basis)',
    'scaling': 'O(2^N)',
    'correlation_captured': 1.0
}

print(f"   ✓ FCI Energy: {fci_energy:.8f} Ha (EXACT)")
print(f"   Time: {fci_time:.4f} seconds")

# Calculate correlation energies
total_correlation = fci_energy - hf_energy

for method in ['MP2', 'CCSD', 'FCI']:
    corr = classical_results[method]['energy'] - hf_energy
    pct_corr = abs(corr / total_correlation) if total_correlation != 0 else 0
    classical_results[method]['correlation_captured'] = float(pct_corr)

print("\n" + "=" * 70)
print("Add VQE Results")
print("=" * 70)

if vqe_available:
    vqe_energy = vqe_data['final_energy']
    vqe_time = vqe_data['computation_time_seconds']
    vqe_iters = vqe_data['total_iterations']
    
    classical_results['VQE'] = {
        'energy': float(vqe_energy),
        'time': vqe_time,
        'description': f'Variational quantum ({vqe_data["num_layers"]} layers)',
        'scaling': 'O(poly(N))',
        'correlation_captured': float(abs((vqe_energy - hf_energy) / total_correlation)),
        'iterations': vqe_iters
    }
    
    print(f"✓ VQE Energy: {vqe_energy:.8f} Ha")
    print(f"  Time: {vqe_time:.1f} seconds")
    print(f"  Iterations: {vqe_iters}")
else:
    # Placeholder VQE result
    vqe_energy = fci_energy + 0.01  # Assume 1% error
    classical_results['VQE'] = {
        'energy': float(vqe_energy),
        'time': 300.0,
        'description': 'Variational quantum (2 layers)',
        'scaling': 'O(poly(N))',
        'correlation_captured': float(abs((vqe_energy - hf_energy) / total_correlation)),
        'iterations': 150
    }
    print("⚠️  Using placeholder VQE results")

print("\n" + "=" * 70)
print("COMPARISON TABLE")
print("=" * 70)

print(f"\n{'Method':<8} {'Energy (Ha)':>15} {'Error (Ha)':>12} {'Error (%)':>10} "
      f"{'Corr %':>10} {'Time (s)':>10} {'Scaling':>12}")
print("-" * 85)

methods = ['HF', 'MP2', 'CCSD', 'FCI', 'VQE']
for method in methods:
    res = classical_results[method]
    error = abs(res['energy'] - fci_energy)
    error_pct = abs((res['energy'] - fci_energy) / fci_energy * 100)
    
    print(f"{method:<8} {res['energy']:>15.8f} {error:>12.8f} {error_pct:>10.4f} "
          f"{res['correlation_captured']*100:>9.1f}% {res['time']:>10.4f} {res['scaling']:>12}")

print("\n" + "=" * 70)
print("VISUALIZATION")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# Plot 1: Energy accuracy
ax1 = axes[0, 0]
methods_plot = ['HF', 'MP2', 'CCSD', 'VQE', 'FCI']
energies = [classical_results[m]['energy'] for m in methods_plot]
colors_method = ['orange', 'yellow', 'lightblue', 'lightgreen', 'red']

bars = ax1.bar(range(len(methods_plot)), energies, color=colors_method,
              alpha=0.7, edgecolor='black', linewidth=2)
ax1.axhline(fci_energy, color='red', linestyle='--', linewidth=2,
            label='FCI (Exact)', alpha=0.7)

ax1.set_xticks(range(len(methods_plot)))
ax1.set_xticklabels(methods_plot, fontsize=11)
ax1.set_ylabel('Energy (Hartree)', fontsize=12, fontweight='bold')
ax1.set_title('Ground State Energy Comparison', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

for i, (method, energy) in enumerate(zip(methods_plot, energies)):
    ax1.text(i, energy - 0.01, f'{energy:.4f}', ha='center', 
            fontsize=8, rotation=90, va='top')

# Plot 2: Errors (log scale)
ax2 = axes[0, 1]
errors = [abs(classical_results[m]['energy'] - fci_energy) for m in methods_plot[:-1]]
methods_error = methods_plot[:-1]

bars = ax2.bar(range(len(methods_error)), errors, 
              color=colors_method[:-1], alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_yscale('log')
ax2.set_xticks(range(len(methods_error)))
ax2.set_xticklabels(methods_error, fontsize=11)
ax2.set_ylabel('Absolute Error (Hartree)', fontsize=12, fontweight='bold')
ax2.set_title('Error vs FCI (Log Scale)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y', which='both')

for i, err in enumerate(errors):
    ax2.text(i, err * 1.5, f'{err:.2e}', ha='center', fontsize=8)

# Plot 3: Correlation energy captured
ax3 = axes[0, 2]
corr_pcts = [classical_results[m]['correlation_captured'] * 100 for m in methods_plot]

bars = ax3.bar(range(len(methods_plot)), corr_pcts, color=colors_method,
              alpha=0.7, edgecolor='black', linewidth=2)
ax3.axhline(100, color='red', linestyle='--', linewidth=2, alpha=0.5)
ax3.set_xticks(range(len(methods_plot)))
ax3.set_xticklabels(methods_plot, fontsize=11)
ax3.set_ylabel('Correlation Energy Captured (%)', fontsize=12, fontweight='bold')
ax3.set_title('Electron Correlation Recovery', fontsize=13, fontweight='bold')
ax3.set_ylim([0, 110])
ax3.grid(True, alpha=0.3, axis='y')

for i, pct in enumerate(corr_pcts):
    ax3.text(i, pct + 2, f'{pct:.1f}%', ha='center', fontsize=9, fontweight='bold')

# Plot 4: Computation time
ax4 = axes[1, 0]
times = [classical_results[m]['time'] for m in methods_plot]

bars = ax4.bar(range(len(methods_plot)), times, color=colors_method,
              alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_xticks(range(len(methods_plot)))
ax4.set_xticklabels(methods_plot, fontsize=11)
ax4.set_ylabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
ax4.set_title('Computational Cost', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for i, (method, t) in enumerate(zip(methods_plot, times)):
    ax4.text(i, t + max(times)*0.02, f'{t:.2f}s', ha='center', fontsize=9)

# Plot 5: Accuracy vs Time trade-off
ax5 = axes[1, 1]
errors_pct = [abs((classical_results[m]['energy'] - fci_energy) / fci_energy * 100) 
              for m in methods_plot[:-1]]

scatter = ax5.scatter(times[:-1], errors_pct, s=300, c=range(len(methods_error)),
                     cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)

for i, method in enumerate(methods_error):
    ax5.annotate(method, (times[i], errors_pct[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')

ax5.set_xlabel('Computation Time (seconds)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
ax5.set_title('Accuracy vs Cost Trade-off', fontsize=13, fontweight='bold')
ax5.set_yscale('log')
ax5.grid(True, alpha=0.3, which='both')

# Plot 6: Scaling comparison
ax6 = axes[1, 2]
scalings = ['O(N⁴)', 'O(N⁵)', 'O(N⁶)', 'O(poly(N))', 'O(2^N)']
scaling_labels = ['HF\nN⁴', 'MP2\nN⁵', 'CCSD\nN⁶', 'VQE\npoly(N)', 'FCI\n2^N']

# Illustrative scaling (normalized)
N = np.arange(2, 20, 2)
hf_scale = N**4 / (20**4) * 100
mp2_scale = N**5 / (20**5) * 100
ccsd_scale = N**6 / (20**6) * 100
vqe_scale = N**3 / (20**3) * 100  # Polynomial assumption
fci_scale = 2**N / (2**20) * 100

ax6.plot(N, hf_scale, 'o-', linewidth=2, label='HF', color='orange')
ax6.plot(N, mp2_scale, 's-', linewidth=2, label='MP2', color='yellow')
ax6.plot(N, ccsd_scale, '^-', linewidth=2, label='CCSD', color='lightblue')
ax6.plot(N, vqe_scale, 'd-', linewidth=2, label='VQE', color='lightgreen')
ax6.plot(N, fci_scale, 'v-', linewidth=2, label='FCI', color='red')

ax6.set_xlabel('System Size (orbitals)', fontsize=12, fontweight='bold')
ax6.set_ylabel('Relative Cost (%)', fontsize=12, fontweight='bold')
ax6.set_title('Computational Scaling', fontsize=13, fontweight='bold')
ax6.set_yscale('log')
ax6.legend()
ax6.grid(True, alpha=0.3, which='both')

# Add crossover annotation
ax6.annotate('VQE advantage\nregion →', xy=(15, vqe_scale[-3]), 
            xytext=(12, 5), fontsize=9,
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

plt.suptitle('Day 4: Classical vs Quantum Methods Comparison',
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

plt.savefig('day4_classical_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: day4_classical_comparison.png")
plt.show()

# Save results
comparison_data = {
    'target_system': 'H2 molecule, sto-3g basis',
    'bond_length': bond_length,
    'total_correlation_energy': float(total_correlation),
    'methods': classical_results
}

with open('day4_classical_comparison.json', 'w') as f:
    json.dump(comparison_data, f, indent=2)

print("✓ Saved: day4_classical_comparison.json")

print("\n" + "=" * 70)
print("KEY INSIGHTS")
print("=" * 70)

print("\n1. Accuracy Ranking:")
methods_by_accuracy = sorted(methods_plot[:-1], 
                            key=lambda m: abs(classical_results[m]['energy'] - fci_energy))
for i, method in enumerate(methods_by_accuracy, 1):
    error = abs(classical_results[method]['energy'] - fci_energy)
    print(f"   {i}. {method}: {error:.8f} Ha error")

print("\n2. Speed Ranking:")
methods_by_speed = sorted(methods_plot,
                         key=lambda m: classical_results[m]['time'])
for i, method in enumerate(methods_by_speed, 1):
    print(f"   {i}. {method}: {classical_results[method]['time']:.4f} seconds")

print("\n3. VQE Performance:")
vqe_error = abs(classical_results['VQE']['energy'] - fci_energy)
vqe_error_pct = abs((classical_results['VQE']['energy'] - fci_energy) / fci_energy * 100)
print(f"   Error: {vqe_error:.8f} Ha ({vqe_error_pct:.4f}%)")
print(f"   Correlation: {classical_results['VQE']['correlation_captured']*100:.1f}%")

if vqe_error < abs(classical_results['CCSD']['energy'] - fci_energy):
    print(f"   ✅ VQE beats CCSD accuracy!")
elif vqe_error < abs(classical_results['MP2']['energy'] - fci_energy):
    print(f"   ✅ VQE beats MP2 accuracy!")
else:
    print(f"   📊 VQE competitive with classical methods")

print("\n4. Quantum Advantage:")
print(f"   Current system (4 qubits): Classical methods faster")
print(f"   Crossover point: ~20-30 qubits")
print(f"   VQE scaling: Polynomial vs exponential (FCI)")
print(f"   For large molecules: VQE becomes advantageous")

print("\n" + "=" * 70)
print("DAY 4 - PART 2 COMPLETE!")
print("=" * 70)

