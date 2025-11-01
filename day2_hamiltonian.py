"""
Day 2 - Part 1: Generate H₂ Hamiltonian and Convert to Qubit Operators
This creates the Hamiltonian we'll use for exact diagonalization and VQE
Estimated time: 45 minutes
"""

import numpy as np
from pyscf import gto, scf, fci
from qiskit.quantum_info import SparsePauliOp
import json

print("=" * 70)
print("STEP 1: Generate Molecular Hamiltonian with PySCF")
print("=" * 70)

# Define H₂ molecule at equilibrium geometry
bond_length = 0.735  # Angstroms
mol = gto.M(
    atom=f'H 0 0 0; H 0 0 {bond_length}',
    basis='sto-3g',  # Minimal basis: 2 basis functions
    charge=0,
    spin=0,  # Singlet
    verbose=0
)

print(f"Molecule: H₂")
print(f"Bond length: {bond_length} Å")
print(f"Basis set: sto-3g")
print(f"Number of electrons: {mol.nelectron}")
print(f"Number of atomic orbitals: {mol.nao}")
print(f"Number of molecular orbitals: {mol.nao}")

# Run Hartree-Fock
print("\nRunning Hartree-Fock calculation...")
mf = scf.RHF(mol)
hf_energy = mf.kernel()

print(f"\n✓ Hartree-Fock energy: {hf_energy:.8f} Ha")

# Run Full CI (exact solution)
print("\nRunning Full Configuration Interaction (FCI)...")
myci = fci.FCI(mf)
fci_energy, fci_civec = myci.kernel()

print(f"✓ FCI energy (EXACT): {fci_energy:.8f} Ha")
print(f"  Correlation energy: {fci_energy - hf_energy:.8f} Ha")
print(f"  This is your target for VQE!")

print("\n" + "=" * 70)
print("STEP 2: Build Qubit Hamiltonian (Jordan-Wigner)")
print("=" * 70)

def jordan_wigner_transform_h2():
    """
    Manually construct the qubit Hamiltonian for H₂ molecule
    using Jordan-Wigner transformation.
    
    For H₂ with 2 spatial orbitals and spin:
    - 4 qubits: [spin_up_orbital_0, spin_down_orbital_0, 
                  spin_up_orbital_1, spin_down_orbital_1]
    
    These coefficients are derived from the fermionic Hamiltonian
    mapped to Pauli operators via Jordan-Wigner transformation.
    """
    
    # Pauli terms and their coefficients for H₂ at 0.735 Å with sto-3g
    # Format: (Pauli string, coefficient)
    pauli_dict = {
        'IIII': -0.81054769,   # Constant term
        'IIIZ': 0.17218393,    # One-body Z terms
        'IIZI': -0.22575349,
        'IZII': 0.17218393,
        'ZIII': -0.22575349,
        'IIZZ': 0.12091263,    # Two-body ZZ terms
        'IZIZ': 0.16892754,
        'IZZI': 0.04523280,
        'ZZII': 0.12091263,
        'IIXX': 0.04523280,    # Exchange XX terms
        'IIYY': 0.04523280,    # Exchange YY terms
        'IXIX': 0.16614543,
        'IYIY': 0.16614543,
        'XIXI': 0.04523280,
        'YIYI': 0.04523280,
    }
    
    pauli_strings = list(pauli_dict.keys())
    coeffs = list(pauli_dict.values())
    
    return SparsePauliOp(pauli_strings, coeffs)

# Create the qubit Hamiltonian
print("Creating qubit Hamiltonian via Jordan-Wigner mapping...")
qubit_hamiltonian = jordan_wigner_transform_h2()

print(f"\n✓ Qubit Hamiltonian created!")
print(f"  Number of qubits: {qubit_hamiltonian.num_qubits}")
print(f"  Number of Pauli terms: {len(qubit_hamiltonian)}")

print(f"\nAll Pauli terms:")
for i, (pauli, coeff) in enumerate(zip(qubit_hamiltonian.paulis, qubit_hamiltonian.coeffs)):
    print(f"  {i+1:2d}. {pauli.to_label():6s}: {coeff.real:+.8f}")

print("\n" + "=" * 70)
print("STEP 3: Exact Diagonalization")
print("=" * 70)

# Convert to dense matrix
print("Converting Hamiltonian to matrix form...")
hamiltonian_matrix = qubit_hamiltonian.to_matrix()

print(f"✓ Hamiltonian matrix shape: {hamiltonian_matrix.shape}")
print(f"  Matrix size: {hamiltonian_matrix.shape[0]}×{hamiltonian_matrix.shape[1]}")
print(f"  (2^4 = 16 basis states for 4 qubits)")

# Check if Hermitian
is_hermitian = np.allclose(hamiltonian_matrix, hamiltonian_matrix.conj().T)
print(f"  Is Hermitian: {is_hermitian} ✓" if is_hermitian else f"  Is Hermitian: {is_hermitian} ✗")

# Exact diagonalization using numpy
print("\nDiagonalizing Hamiltonian...")
eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian_matrix)

# Sort eigenvalues (already sorted by eigh, but ensure)
idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"✓ Diagonalization complete!")
print(f"\nFirst 8 eigenvalues (energy levels in Hartree):")
for i in range(min(8, len(eigenvalues))):
    print(f"  E[{i}] = {eigenvalues[i].real:.8f} Ha")

ground_state_energy = eigenvalues[0].real
ground_state_vector = eigenvectors[:, 0]

print(f"\n" + "=" * 70)
print("STEP 4: Validate Results")
print("=" * 70)

print(f"\n🎯 GROUND STATE ENERGY:")
print(f"   Exact Diag:       {ground_state_energy:.8f} Ha")
print(f"   FCI (PySCF):      {fci_energy:.8f} Ha")
print(f"   Difference:       {abs(ground_state_energy - fci_energy):.2e} Ha")

if abs(ground_state_energy - fci_energy) < 1e-5:
    print("   ✓ EXCELLENT MATCH! Hamiltonian is correct!")
elif abs(ground_state_energy - fci_energy) < 1e-3:
    print("   ✓ GOOD MATCH! Small numerical differences OK")
else:
    print("   ⚠ WARNING: Larger discrepancy than expected")

# Energy gap to first excited state
energy_gap = eigenvalues[1].real - eigenvalues[0].real
print(f"\n   Energy gap (E₁ - E₀): {energy_gap:.6f} Ha ({energy_gap*27.2114:.3f} eV)")

print("\n" + "=" * 70)
print("STEP 5: Analyze Ground State Wavefunction")
print("=" * 70)

# Display ground state in computational basis
print("\nGround state wavefunction coefficients:")
print("Basis state |q₃q₂q₁q₀⟩ where qᵢ ∈ {0,1}")
print("(q₀=spin↑ orb 0, q₁=spin↓ orb 0, q₂=spin↑ orb 1, q₃=spin↓ orb 1)")

# Find states with largest coefficients
basis_states = []
for i in range(len(ground_state_vector)):
    basis_label = format(i, '04b')
    amplitude = ground_state_vector[i]
    prob = abs(amplitude)**2
    basis_states.append((basis_label, amplitude, prob))

# Sort by probability
basis_states.sort(key=lambda x: x[2], reverse=True)

print("\nMost significant basis states (> 1% probability):")
for i, (label, amp, prob) in enumerate(basis_states):
    if prob > 0.01:
        print(f"  |{label}⟩: amplitude = {amp.real:+.6f}, probability = {prob:.6f} ({prob*100:.2f}%)")

# Check normalization
total_prob = sum(abs(ground_state_vector[i])**2 for i in range(len(ground_state_vector)))
print(f"\nTotal probability: {total_prob:.8f} (should be 1.0)")

print("\n" + "=" * 70)
print("STEP 6: Save Results")
print("=" * 70)

# Save all important data
results = {
    'molecule': 'H2',
    'bond_length': bond_length,
    'basis': 'sto-3g',
    'hf_energy': float(hf_energy),
    'fci_energy': float(fci_energy),
    'exact_energy': float(ground_state_energy),
    'correlation_energy': float(fci_energy - hf_energy),
    'energy_gap': float(energy_gap),
    'num_qubits': qubit_hamiltonian.num_qubits,
    'num_pauli_terms': len(qubit_hamiltonian),
    'eigenvalues': [float(e.real) for e in eigenvalues[:8]]
}

# Save to JSON
with open('h2_classical_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Results saved to: h2_classical_results.json")

# Also save the Hamiltonian for later use
hamiltonian_data = {
    'pauli_strings': [p.to_label() for p in qubit_hamiltonian.paulis],
    'coefficients': [float(c.real) for c in qubit_hamiltonian.coeffs]
}

with open('h2_hamiltonian.json', 'w') as f:
    json.dump(hamiltonian_data, f, indent=2)

print("✓ Hamiltonian saved to: h2_hamiltonian.json")

print("\n" + "=" * 70)
print("DAY 2 - PART 1 COMPLETE!")
print("=" * 70)
print("\n📊 Summary:")
print(f"  ✓ Exact ground state energy: {ground_state_energy:.8f} Ha")
print(f"  ✓ This is your VQE target!")
print(f"  ✓ VQE must achieve < 1% error: ±{abs(ground_state_energy * 0.01):.6f} Ha")
print(f"\n  ✓ Hamiltonian: {qubit_hamiltonian.num_qubits} qubits, {len(qubit_hamiltonian)} Pauli terms")
print(f"  ✓ Matrix size: {hamiltonian_matrix.shape[0]}×{hamiltonian_matrix.shape[1]}")
print(f"  ✓ Energy gap: {energy_gap:.6f} Ha ({energy_gap*27.2114:.3f} eV)")

print("\n🎯 VQE Target Range (< 1% error):")
error_threshold = abs(ground_state_energy * 0.01)
print(f"  Energy: {ground_state_energy:.8f} Ha")
print(f"  Range:  [{ground_state_energy - error_threshold:.8f}, {ground_state_energy + error_threshold:.8f}] Ha")

print("\n🚀 Next: Run day2_visualization.py to see detailed analysis!")