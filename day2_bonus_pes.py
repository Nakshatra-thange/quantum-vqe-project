"""
Day 2 - BONUS: H₂ Potential Energy Surface
Compute exact energy as a function of bond length
This curve is what VQE should reproduce!
Estimated time: 30 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci
import json

print("=" * 70)
print("BONUS: H₂ POTENTIAL ENERGY SURFACE")
print("=" * 70)

# Define bond lengths to scan (in Angstroms)
bond_lengths = np.linspace(0.4, 2.5, 20)
print(f"\nScanning {len(bond_lengths)} bond lengths: {bond_lengths[0]:.2f} - {bond_lengths[-1]:.2f} Å")

# Storage for results
hf_energies = []
fci_energies = []
correlation_energies = []

print("\nComputing energies...")
for i, r in enumerate(bond_lengths):
    # Create molecule
    mol = gto.M(
        atom=f'H 0 0 0; H 0 0 {r}',
        basis='sto-3g',
        charge=0,
        spin=0,
        verbose=0
    )
    
    # Hartree-Fock
    mf = scf.RHF(mol)
    mf.kernel()
    hf_energy = mf.e_tot
    
    # Full CI
    myci = fci.FCI(mf)
    fci_energy = myci.kernel()[0]
    
    # Store results
    hf_energies.append(hf_energy)
    fci_energies.append(fci_energy)
    correlation_energies.append(fci_energy - hf_energy)
    
    # Progress indicator
    if (i + 1) % 5 == 0:
        print(f"  Progress: {i+1}/{len(bond_lengths)} ({(i+1)/len(bond_lengths)*100:.0f}%)")

print("✓ Calculation complete!")

# Find equilibrium
fci_energies_array = np.array(fci_energies)
min_idx = np.argmin(fci_energies_array)
equilibrium_distance = bond_lengths[min_idx]
equilibrium_energy = fci_energies_array[min_idx]

print(f"\n📍 Equilibrium geometry:")
print(f"  Bond length: {equilibrium_distance:.4f} Å")
print(f"  Energy: {equilibrium_energy:.8f} Ha")

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Potential Energy Curves
ax1 = axes[0, 0]
ax1.plot(bond_lengths, hf_energies, 'o-', label='Hartree-Fock', 
         color='orange', linewidth=2, markersize=6)
ax1.plot(bond_lengths, fci_energies, 's-', label='FCI (Exact)', 
         color='blue', linewidth=2, markersize=6)
ax1.axvline(equilibrium_distance, color='red', linestyle='--', 
            linewidth=1.5, alpha=0.5, label=f'Equilibrium ({equilibrium_distance:.3f} Å)')
ax1.axhline(equilibrium_energy, color='green', linestyle=':', 
            linewidth=1, alpha=0.5)

ax1.set_xlabel('H-H Bond Length (Å)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Energy (Hartree)', fontsize=12, fontweight='bold')
ax1.set_title('H₂ Potential Energy Surface', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Annotate minimum
ax1.plot(equilibrium_distance, equilibrium_energy, 'r*', 
         markersize=15, label='Minimum')
ax1.text(equilibrium_distance + 0.1, equilibrium_energy - 0.01, 
         f'Min: {equilibrium_energy:.4f} Ha\nat {equilibrium_distance:.3f} Å',
         fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# Plot 2: Correlation Energy
ax2 = axes[0, 1]
ax2.plot(bond_lengths, correlation_energies, 'o-', 
         color='purple', linewidth=2, markersize=6)
ax2.axvline(equilibrium_distance, color='red', linestyle='--', 
            linewidth=1.5, alpha=0.5)
ax2.axhline(0, color='black', linestyle='-', linewidth=0.8)

ax2.set_xlabel('H-H Bond Length (Å)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Correlation Energy (Hartree)', fontsize=12, fontweight='bold')
ax2.set_title('Correlation Energy vs Bond Length', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Annotate interesting regions
max_corr_idx = np.argmax(np.abs(correlation_energies))
ax2.text(bond_lengths[max_corr_idx], correlation_energies[max_corr_idx] - 0.002,
         f'Max correlation\n{correlation_energies[max_corr_idx]:.4f} Ha',
         fontsize=9, ha='center',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

# Plot 3: Energy Error (HF vs FCI)
ax3 = axes[1, 0]
error_energies = np.array(fci_energies) - np.array(hf_energies)
error_percent = (error_energies / np.array(hf_energies)) * 100

ax3.plot(bond_lengths, error_percent, 'o-', 
         color='red', linewidth=2, markersize=6)
ax3.axvline(equilibrium_distance, color='green', linestyle='--', 
            linewidth=1.5, alpha=0.5)

ax3.set_xlabel('H-H Bond Length (Å)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Error (%)', fontsize=12, fontweight='bold')
ax3.set_title('HF Error Relative to FCI', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)

# Add 1% reference line
ax3.axhline(1.0, color='orange', linestyle=':', linewidth=2, 
            label='1% error (VQE target)', alpha=0.7)
ax3.legend()

# Plot 4: Dissociation Analysis
ax4 = axes[1, 1]

# Compute dissociation energy
dissociation_energy = fci_energies_array[-1] - equilibrium_energy
two_h_atoms_energy = fci_energies_array[-1]  # Energy at R → ∞

ax4.plot(bond_lengths, fci_energies, 's-', 
         color='blue', linewidth=2, markersize=6, label='Total Energy')
ax4.axhline(two_h_atoms_energy, color='red', linestyle='--', 
            linewidth=2, label='Dissociation Limit', alpha=0.7)
ax4.axhline(equilibrium_energy, color='green', linestyle='--', 
            linewidth=2, label='Ground State', alpha=0.7)

# Shade bound region
ax4.fill_between(bond_lengths, equilibrium_energy, two_h_atoms_energy, 
                 alpha=0.2, color='yellow', label='Bound Region')

ax4.set_xlabel('H-H Bond Length (Å)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Energy (Hartree)', fontsize=12, fontweight='bold')
ax4.set_title('Bond Dissociation', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Annotate dissociation energy
ax4.annotate('', xy=(2.2, equilibrium_energy), xytext=(2.2, two_h_atoms_energy),
            arrowprops=dict(arrowstyle='<->', color='purple', lw=2))
ax4.text(2.25, (equilibrium_energy + two_h_atoms_energy)/2, 
         f'D₀ = {dissociation_energy:.4f} Ha\n({dissociation_energy*27.2114:.2f} eV)',
         fontsize=9, color='purple')

plt.suptitle('H₂ Molecule: Complete Energy Analysis', 
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
plt.savefig('day2_potential_energy_surface.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: day2_potential_energy_surface.png")
plt.show()

# Save data for future use
pe_surface_data = {
    'bond_lengths': bond_lengths.tolist(),
    'hf_energies': hf_energies,
    'fci_energies': fci_energies,
    'correlation_energies': correlation_energies,
    'equilibrium_distance': float(equilibrium_distance),
    'equilibrium_energy': float(equilibrium_energy),
    'dissociation_energy': float(dissociation_energy),
    'dissociation_limit': float(two_h_atoms_energy)
}

with open('h2_potential_surface.json', 'w') as f:
    json.dump(pe_surface_data, f, indent=2)

print("✓ Saved: h2_potential_surface.json")

# Print summary
print("\n" + "=" * 70)
print("SUMMARY: H₂ POTENTIAL ENERGY SURFACE")
print("=" * 70)
print(f"\n🎯 Equilibrium Properties:")
print(f"  Bond length:       {equilibrium_distance:.4f} Å")
print(f"  Ground energy:     {equilibrium_energy:.8f} Ha")
print(f"  Dissociation:      {dissociation_energy:.8f} Ha ({dissociation_energy*27.2114:.4f} eV)")
print(f"  Dissociation limit: {two_h_atoms_energy:.8f} Ha")

print(f"\n📊 At Equilibrium (R = {equilibrium_distance:.3f} Å):")
eq_idx = min_idx
print(f"  HF energy:         {hf_energies[eq_idx]:.8f} Ha")
print(f"  FCI energy:        {fci_energies[eq_idx]:.8f} Ha")
print(f"  Correlation:       {correlation_energies[eq_idx]:.8f} Ha")
print(f"  HF error:          {error_percent[eq_idx]:.4f}%")

print(f"\n🔬 Correlation Analysis:")
print(f"  Min correlation:   {min(correlation_energies):.8f} Ha (at {bond_lengths[np.argmin(correlation_energies)]:.3f} Å)")
print(f"  Max correlation:   {max(correlation_energies):.8f} Ha (at {bond_lengths[max_corr_idx]:.3f} Å)")
print(f"  At equilibrium:    {correlation_energies[eq_idx]:.8f} Ha")

print(f"\n💡 VQE Challenge:")
print(f"  This curve is what VQE should reproduce!")
print(f"  At each bond length, VQE must find ground state within < 1% error")
print(f"  Hardest region: Dissociation (R > 1.5 Å) - strong correlation")

print("\n" + "=" * 70)
print("DAY 2 BONUS COMPLETE!")
print("=" * 70)