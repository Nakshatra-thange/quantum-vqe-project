"""
Day 5 - Part 1: ERROR MITIGATION (fixed)
- Ensures parameter lists come from the ansatz (no mismatches)
- Validates parameter lengths and reports helpful errors
- Uses AerSimulator and BackendEstimator if available in your Qiskit
"""

import numpy as np
import json
import time
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError

# NOTE: BackendEstimator may or may not be present depending on qiskit packaging.
# The code below uses a function compute_energy_with_shots that calls BackendEstimator
# if available; otherwise it falls back to AerSimulator.run() + result parsing.
try:
    from qiskit.primitives import BackendEstimator
    _HAS_BACKEND_ESTIMATOR = True
except Exception:
    _HAS_BACKEND_ESTIMATOR = False

print("=" * 70)
print("DAY 5 - PART 1: ERROR MITIGATION (FIXED)")
print("=" * 70)

# ---- Load references -----------------------------------------------------
with open('h2_classical_results.json', 'r') as f:
    results = json.load(f)
target_energy = results['fci_energy']

with open('h2_hamiltonian.json', 'r') as f:
    ham_data = json.load(f)

# Build SparsePauliOp robustly
try:
    hamiltonian = SparsePauliOp(ham_data['pauli_strings'], ham_data['coefficients'])
except Exception:
    hamiltonian = SparsePauliOp.from_list(list(zip(ham_data['pauli_strings'], ham_data['coefficients'])))

print(f"\n🎯 Target Energy: {target_energy:.8f} Ha")

# ---- Ansatz builder (correct param count) -------------------------------
def build_hardware_efficient_ansatz(num_qubits, num_layers):
    """
    Returns (circuit, parameters_list)
    ParameterVector length = num_qubits * (num_layers + 1)
    """
    circuit = QuantumCircuit(num_qubits)
    num_params = num_qubits * (num_layers + 1)   # <- correct formula
    params = ParameterVector('θ', num_params)
    idx = 0

    # initial layer
    for q in range(num_qubits):
        circuit.ry(params[idx], q)
        idx += 1

    # repeated layers
    for _ in range(num_layers):
        for q in range(num_qubits - 1):
            circuit.cx(q, q + 1)
        for q in range(num_qubits):
            circuit.ry(params[idx], q)
            idx += 1

    # return circuit and a stable list of Parameter objects (in the defined order)
    return circuit, list(params)

# build ansatz
num_qubits = 4
num_layers = 2
ansatz, parameters = build_hardware_efficient_ansatz(num_qubits, num_layers)

# To be extra safe, extract parameters directly from the circuit (should be same order)
param_from_circuit = list(ansatz.parameters)
param_from_circuit.sort(key=lambda p: p.name)

# sanity check
if len(param_from_circuit) != len(parameters):
    # Prefer the circuit's parameter objects (stable)
    parameters = param_from_circuit
else:
    # Use circuit ordering anyway
    parameters = param_from_circuit

num_params = len(parameters)
print(f"\n✓ Ansatz built: {num_qubits} qubits, {num_layers} layers, {num_params} parameters")
print(f"  Circuit depth: {ansatz.depth()}, CNOTs: {ansatz.count_ops().get('cx', 0)}\n")

# ---- Noise model helper -------------------------------------------------
def create_noise_model(error_scale=1.0, include_readout=True):
    nm = NoiseModel()
    err1 = 0.001 * error_scale
    err2 = 0.01 * error_scale

    nm.add_all_qubit_quantum_error(depolarizing_error(err1, 1), ['ry', 'rz'])
    nm.add_all_qubit_quantum_error(depolarizing_error(err2, 2), ['cx'])

    if include_readout:
        rm = [[0.95, 0.05], [0.05, 0.95]]
        nm.add_all_qubit_readout_error(ReadoutError(rm))

    return nm

# ---- Energy evaluation (shots) ------------------------------------------
# This function binds parameters safely (checks length) and evaluates using BackendEstimator
# if available; otherwise falls back to AerSimulator.run(...) and reading result.values if possible.
def compute_energy_with_shots(params_values, hamiltonian, ansatz, parameters, backend, shots=2048):
    if len(params_values) != len(parameters):
        raise ValueError(f"Parameter length mismatch: got {len(params_values)}, expected {len(parameters)}")

    bind_dict = {p: float(v) for p, v in zip(parameters, params_values)}
    bound_circuit = ansatz.assign_parameters(bind_dict)

    # Use BackendEstimator if available (preferred)
    if _HAS_BACKEND_ESTIMATOR:
        estimator = BackendEstimator(backend=backend, options={'shots': shots})
        job = estimator.run(bound_circuit, hamiltonian)
        res = job.result()
        # result.values is an array of expectation estimates
        return float(res.values[0])
    else:
        # Fall back: run the circuit and try to extract expectation.
        # Note: this fallback uses the statevector when backend supports it (no noise).
        run_result = backend.run(bound_circuit, shots=shots).result()
        # Try to read 'values' or 'expectation' if returned (some backends do)
        # If not available, raise an informative error to install BackendEstimator.
        if hasattr(run_result, 'values') and len(run_result.values) > 0:
            return float(run_result.values[0])
        # If the backend returned counts only, we cannot compute Pauli expectation easily here.
        raise RuntimeError("BackendEstimator not available and backend.run() did not return expectation values. "
                           "Install a Qiskit version providing BackendEstimator or run with an Aer primitive that returns expectation results.")

# ---- Baseline VQE with noise (no mitigation) ----------------------------
print("\n" + "=" * 70)
print("STEP 1: Baseline - VQE with Noise (No Mitigation)")
print("=" * 70)

noise_model = create_noise_model(error_scale=1.0, include_readout=True)
backend = AerSimulator(noise_model=noise_model)

# initialize parameters freshly (match current ansatz)
np.random.seed(42)
initial_params = np.random.uniform(-np.pi, np.pi, num_params)

energy_history_baseline = []
iteration_count = [0]

def objective_baseline(x):
    # compute energy with shots; will validate param length
    e = compute_energy_with_shots(x, hamiltonian, ansatz, parameters, backend, shots=1024)
    energy_history_baseline.append(e)
    iteration_count[0] += 1
    if iteration_count[0] % 20 == 0:
        print(f"  Iter {iteration_count[0]:3d}: E = {e:.8f} Ha")
    return e

start = time.time()
result_baseline = minimize(objective_baseline, initial_params, method='COBYLA', options={'maxiter': 80})
baseline_time = time.time() - start

baseline_energy = float(result_baseline.fun)
baseline_error = abs(baseline_energy - target_energy)
baseline_error_pct = abs((baseline_energy - target_energy) / target_energy * 100)

print(f"\n✓ Baseline Results (no mitigation):")
print(f"  Final energy: {baseline_energy:.8f} Ha")
print(f"  Error: {baseline_error:.8f} Ha ({baseline_error_pct:.4f}%)")
print(f"  Time: {baseline_time:.1f} s, Iterations: {iteration_count[0]}")

# store optimized params (for reuse in mitigation steps)
optimized_params = result_baseline.x

# ---- Zero-noise extrapolation (ZNE) ------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Zero-Noise Extrapolation (ZNE)")
print("=" * 70)

noise_scales = [1.0, 2.0, 3.0]
energies_zne = []

for scale in noise_scales:
    nm = create_noise_model(error_scale=scale, include_readout=True)
    backend_scaled = AerSimulator(noise_model=nm)
    e = compute_energy_with_shots(optimized_params, hamiltonian, ansatz, parameters, backend_scaled, shots=2048)
    energies_zne.append(float(e))
    print(f"  scale {scale}x -> E = {e:.8f}")

# quadratic fit
def poly_model(x, a, b, c): return a + b * x + c * x**2
popt, _ = curve_fit(poly_model, noise_scales, energies_zne)
zne_energy = float(popt[0])
zne_error = abs(zne_energy - target_energy)
zne_error_pct = abs((zne_energy - target_energy) / target_energy * 100)

print(f"\n✓ ZNE extrapolated E = {zne_energy:.8f} Ha  (Error {zne_error_pct:.4f}%)")

# ---- Readout mitigation (simple approach) ------------------------------
print("\n" + "=" * 70)
print("STEP 3: Readout Error Mitigation (approximate)")
print("=" * 70)

# For this simple pipeline we emulate readout-correction by removing readout errors from model
nm_no_readout = create_noise_model(error_scale=1.0, include_readout=False)
backend_no_readout = AerSimulator(noise_model=nm_no_readout)
energy_no_readout = compute_energy_with_shots(optimized_params, hamiltonian, ansatz, parameters, backend_no_readout, shots=2048)
corrected_error = abs(energy_no_readout - target_energy)
corrected_error_pct = abs((energy_no_readout - target_energy) / target_energy * 100)

print(f"\n✓ Readout-corrected E = {energy_no_readout:.8f} Ha  (Error {corrected_error_pct:.4f}%)")

# ---- Combined mitigation: ZNE on readout-corrected runs -----------------
print("\n" + "=" * 70)
print("STEP 4: Combined Mitigation (ZNE on readout-corrected)")
print("=" * 70)

energies_zne_nr = []
for s in noise_scales:
    nm = create_noise_model(error_scale=s, include_readout=False)
    backend_s = AerSimulator(noise_model=nm)
    e = compute_energy_with_shots(optimized_params, hamiltonian, ansatz, parameters, backend_s, shots=2048)
    energies_zne_nr.append(float(e))
    print(f"  scale {s}x (no readout) -> E = {e:.8f}")

popt2, _ = curve_fit(poly_model, noise_scales, energies_zne_nr)
combined_energy = float(popt2[0])
combined_error = abs(combined_energy - target_energy)
combined_error_pct = abs((combined_energy - target_energy) / target_energy * 100)

print(f"\n✓ Combined (ZNE + readout) E = {combined_energy:.8f} Ha  (Error {combined_error_pct:.4f}%)")

# ---- Summary & save ----------------------------------------------------
results_out = {
    'target_energy': target_energy,
    'baseline': {'energy': baseline_energy, 'error_pct': baseline_error_pct},
    'zne': {'energy': zne_energy, 'error_pct': zne_error_pct},
    'readout_corrected': {'energy': float(energy_no_readout), 'error_pct': corrected_error_pct},
    'combined': {'energy': combined_energy, 'error_pct': combined_error_pct}
}

with open('day5_error_mitigation_results.json', 'w') as f:
    json.dump(results_out, f, indent=2)

print("\nSaved: day5_error_mitigation_results.json")
print("\n" + "=" * 70)
print("DAY 5 - PART 1 COMPLETE (FIXED)")
print("=" * 70)
