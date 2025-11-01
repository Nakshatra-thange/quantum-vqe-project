import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

print("="*70)
print("DAY 3: VQE IMPLEMENTATION (Working Version for Qiskit 2.x)")
print("="*70)

# Reference ground state energy for H2 (FCI at 0.735Å)
FCI_GROUND_STATE_ENERGY = -1.13730604
print(f"\n🎯 Target Energy (FCI): {FCI_GROUND_STATE_ENERGY:.8f} Ha\n")

# ===== HAMILTONIAN (15 Pauli terms) =====
coeffs = [
   -1.052373245772859,
   0.39793742484318045,
   -0.39793742484318045,
   -0.01128010425623538,
   0.18093119978423156,
   0.18093119978423156,
   0.16868898170361212,
   -0.18093119978423156,
   -0.18093119978423156,
   -0.16868898170361212,
   0.120911847271373,
   0.120911847271373,
   0.16586841921814402,
   0.16586841921814402,
   0.16586841921814402,
]

paulis = [
   "IIII",
   "ZIII", "IZII",
   "ZZII", "IIZZ",
   "XXII", "YYII",
   "IIXX", "IIYY",
   "ZXZX", "YZYZ",
   "XZXZ", "YZYZ",
   "ZIZI", "IZIZ"
]

hamiltonian = SparsePauliOp.from_list(list(zip(paulis, coeffs)))
num_qubits = hamiltonian.num_qubits

print(f"✓ Hamiltonian loaded ({num_qubits} qubits)\n")


# ===== ANSATZ (Hardware Efficient Ansatz) =====
def build_ansatz(num_qubits, layers):
    qc = QuantumCircuit(num_qubits)
    params = []

    # First rotation layer
    for q in range(num_qubits):
        theta = Parameter(f"θ_{len(params)}")
        qc.ry(theta, q)
        params.append(theta)

    # Repeated entangling + rotation layers
    for _ in range(layers):
        for q in range(num_qubits - 1):
            qc.cx(q, q+1)

        for q in range(num_qubits):
            theta = Parameter(f"θ_{len(params)}")
            qc.ry(theta, q)
            params.append(theta)

    return qc, params

layers = 2
ansatz, parameters = build_ansatz(num_qubits, layers)

expected = num_qubits * (layers + 1)
if len(parameters) != expected:
    raise RuntimeError(f"Parameter mismatch: expected {expected}, got {len(parameters)}")

print(f"✓ Ansatz built: {len(parameters)} parameters, circuit depth ≈ {ansatz.depth()}\n")


# ===== ENERGY FUNCTION =====
sim = AerSimulator(method="statevector")

def compute_energy(param_values):
    bound = ansatz.assign_parameters(dict(zip(parameters, param_values)))
    sv = Statevector.from_instruction(bound)
    return sv.expectation_value(hamiltonian).real


# ===== RUN VQE =====
init_params = np.random.uniform(0, 2*np.pi, len(parameters))
print("🚀 Starting VQE Optimization...\n")

result = minimize(compute_energy, init_params, method="COBYLA", options={"maxiter": 200})

print("\n================ RESULTS ================")
print(f"Optimized Energy = {result.fun:.8f} Ha")
print(f"Reference (FCI)  = {FCI_GROUND_STATE_ENERGY:.8f} Ha")
print(f"Absolute Error   = {abs(result.fun - FCI_GROUND_STATE_ENERGY):.6f} Ha")
print("Converged        =", result.success)
print("=========================================\n")
