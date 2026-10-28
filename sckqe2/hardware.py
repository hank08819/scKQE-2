"""Run the encoding on quantum hardware and audit it against the exact values (Supplementary Note 5).

The circuit of one block puts each score x_q on its own qubit: three layers of R_Y(2^-t s x_q) then R_Z(s x_q), t = 0, 1, 2,
with no entangling gate (entangle=True inserts a chain of controlled-Z gates between R_Y and R_Z, the control of the paper).
`observables` lists the 3l single-qubit Paulis and the l(l-1)/2 ZZ pairs read by KQE-2, in the order of sckqe2.encode.
`exact_expectations` gives their exact values for the entanglement-free circuit, so every hardware number can be checked.
Requires qiskit (pip install qiskit; qiskit-ibm-runtime to submit to IBM processors).
"""
import numpy as np
from .encode import bloch_vectors, T_LAYERS


def block_circuit(x, s=0.19, layers=T_LAYERS, entangle=False):
    from qiskit import QuantumCircuit
    l = len(x); qc = QuantumCircuit(l)
    for t in range(layers):
        for q in range(l):
            qc.ry(float((2.0 ** -t) * s * x[q]), q)
        if entangle:
            for q in range(l - 1):
                qc.cz(q, q + 1)
        for q in range(l):
            qc.rz(float(s * x[q]), q)
    return qc


def observables(l):
    """SparsePauliOp list: X_q, Y_q, Z_q for every qubit q, then Z_i Z_j for i < j (qiskit little-endian labels)."""
    from qiskit.quantum_info import SparsePauliOp
    lab = lambda d: "".join(d.get(q, "I") for q in reversed(range(l)))
    ops = [SparsePauliOp(lab({q: P})) for q in range(l) for P in "XYZ"]
    ops += [SparsePauliOp(lab({i: "Z", j: "Z"})) for i in range(l) for j in range(i + 1, l)]
    return ops


def exact_expectations(x, s=0.19):
    """Exact <X_q>, <Y_q>, <Z_q> and <Z_i Z_j> of the entanglement-free block circuit, in the order of `observables`."""
    R = bloch_vectors(np.asarray(x, np.float64)[None, :], s)[0]; l = len(x)
    return np.concatenate([R.reshape(-1), [R[i, 2] * R[j, 2] for i in range(l) for j in range(i + 1, l)]])


def simulate(x, s=0.19, entangle=False):
    """Noise-free expectations of the circuit by qiskit's statevector (checks the closed form; with entangle=True, the control)."""
    from qiskit.quantum_info import Statevector
    sv = Statevector(block_circuit(x, s, entangle=entangle))
    return np.array([sv.expectation_value(o).real for o in observables(len(x))])
