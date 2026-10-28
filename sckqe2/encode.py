"""Quantum encoding without entanglement (Methods; Theorems 2-4).

The L scores of a cell are split into blocks of `qubits` consecutive scores. Each score x of a block is encoded on its own
qubit by three re-uploading layers R_Y(2^-t s x) then R_Z(s x), t = 0, 1, 2, starting from |0>; no entangling gate.
Features per block: <X>, <Y>, <Z> of every qubit, then <Z_i Z_j> of every pair i < j; all blocks are concatenated,
centered and divided by one global standard deviation.

Two backends give the same features:
- backend="statevector" (default): the batched statevector simulator used for the paper, complex64 on the GPU when
  available. It reproduces the published results bit for bit on the same hardware type.
- backend="closed_form": float64 formulas for single-qubit states (no statevector); exact, used to check the simulator.
"""
import numpy as np

T_LAYERS = 3


def lipschitz(s, T=T_LAYERS):
    """Lambda = s (T + sum_t 2^-t): bound on how fast any Bloch vector moves with its own score (Theorem 2)."""
    return s * (T + sum(2.0 ** -t for t in range(T)))


# ------------------------------------------------------------------ closed form (float64)
def bloch_vectors(E, s=0.19, T=T_LAYERS):
    """Bloch vectors (<X>, <Y>, <Z>) of every score's qubit, shape (cells, L, 3); every vector has unit length."""
    E = np.asarray(E, np.float64)
    a0 = np.ones(E.shape, complex); a1 = np.zeros(E.shape, complex)
    for t in range(T):
        ang = (2.0 ** -t) * s * E; c, sn = np.cos(ang / 2), np.sin(ang / 2)
        a0, a1 = c * a0 - sn * a1, sn * a0 + c * a1                       # R_Y
        ph = np.exp(-0.5j * s * E); a0, a1 = a0 * ph, a1 * np.conj(ph)     # R_Z
    cr = a0 * np.conj(a1)
    return np.stack([2 * cr.real, -2 * cr.imag, np.abs(a0) ** 2 - np.abs(a1) ** 2], -1)


def _features_from_bloch(R, qubits):
    n, nb = R.shape[0], R.shape[1] // qubits
    R = R[:, :nb * qubits].reshape(n, nb, qubits, 3)
    pairs = [(i, j) for i in range(qubits) for j in range(i + 1, qubits)]
    out = []
    for b in range(nb):
        z = R[:, b, :, 2]
        out.append(np.concatenate([R[:, b].reshape(n, 3 * qubits), np.stack([z[:, i] * z[:, j] for i, j in pairs], 1)], 1))
    return np.concatenate(out, 1)


# ------------------------------------------------------------------ batched statevector (paper implementation)
def _statevector_features(E, s, qubits, chunk=4000, device=None):
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    l = qubits; n, L = E.shape; nb = L // l

    def apply_1q(psi, G, k):
        B = psi.shape[0]; psi = psi.reshape(B, 2 ** k, 2, 2 ** (l - k - 1))
        return torch.einsum("bij,bajc->baic", G, psi).reshape(B, 2 ** l)

    def ry(th):
        c, sn = torch.cos(th / 2), torch.sin(th / 2)
        return torch.stack([torch.stack([c, -sn], -1), torch.stack([sn, c], -1)], -2).to(torch.complex64)

    def rz(th):
        e = torch.exp(-0.5j * th.to(torch.complex64)); z = torch.zeros_like(e)
        return torch.stack([torch.stack([e, z], -1), torch.stack([z, e.conj()], -1)], -2)

    w = torch.full((T_LAYERS, l), s, device=dev)
    for r in range(T_LAYERS):
        w[r] *= 0.5 ** r
    v = torch.full((T_LAYERS, l), s, device=dev); b0 = torch.zeros(T_LAYERS, l, device=dev)
    cz = torch.ones(2 ** l, device=dev, dtype=torch.complex64)         # no entangling gate (identity mask)
    idx = torch.arange(2 ** l, device=dev); bits = torch.stack([((idx >> (l - 1 - i)) & 1) for i in range(l)], 1).float()
    zsig = 1 - 2 * bits; pairs = [(i, j) for i in range(l) for j in range(i + 1, l)]
    zz = torch.stack([zsig[:, i] * zsig[:, j] for i, j in pairs], 1)
    Et = torch.tensor(np.asarray(E[:, :nb * l], np.float32), device=dev); out = []
    with torch.no_grad():
        for s0 in range(0, n, chunk):
            x = Et[s0:s0 + chunk]; m = len(x); x = x.reshape(m, nb, l).permute(1, 0, 2).reshape(nb * m, l)
            psi = torch.zeros(nb * m, 2 ** l, dtype=torch.complex64, device=dev); psi[:, 0] = 1
            for r in range(T_LAYERS):
                for k in range(l): psi = apply_1q(psi, ry(x[:, k] * w[r, k] + b0[r, k]), k)
                psi = psi * cz
                for k in range(l): psi = apply_1q(psi, rz(x[:, k] * v[r, k] + b0[r, k]), k)
            B = psi.shape[0]; feats = []
            for k in range(l):
                rr = psi.reshape(B, 2 ** k, 2, 2 ** (l - k - 1)); rho = torch.einsum("baic,bajc->bij", rr, rr.conj())
                feats += [2 * rho[:, 0, 1].real, -2 * rho[:, 0, 1].imag, (rho[:, 0, 0] - rho[:, 1, 1]).real]
            P = torch.cat([torch.stack(feats, 1), (psi.abs() ** 2) @ zz], 1)
            out.append(P.reshape(nb, m, -1).permute(1, 0, 2).reshape(m, -1).cpu().numpy())
    return np.concatenate(out)


def encode(E, s=0.19, qubits=6, backend="statevector", standardize=True, chunk=4000, device=None):
    """Quantum features of the scores E (cells x L): B * (3 * qubits + qubits * (qubits - 1) / 2) per cell."""
    if backend == "statevector":
        F = _statevector_features(np.asarray(E), s, qubits, chunk=chunk, device=device)
    elif backend == "closed_form":
        F = _features_from_bloch(bloch_vectors(E, s), qubits).astype(np.float32)
    else:
        raise ValueError("backend must be 'statevector' or 'closed_form'")
    if standardize:
        F = F - F.mean(0); F = F / F.std()
    return F.astype(np.float32)


def fidelity(Ra, Rb):
    """Fidelity of two product states from their Bloch vectors (..., L, 3): prod_q (1 + r_a . r_b) / 2 (Theorem 4)."""
    return np.prod((1 + (np.asarray(Ra) * np.asarray(Rb)).sum(-1)) / 2, axis=-1)
