"""Tests of scKQE-2: closed form against a brute-force statevector, the two backends against each other, the exact
decomposition against a dense SVD, bit-identical reruns, the rotation identity behind quantum alignment, and the rule."""
import numpy as np
import scipy.sparse as sp
import pytest
from sckqe2.encode import bloch_vectors, encode, fidelity, lipschitz, T_LAYERS
from sckqe2.decompose import exact_scores
from sckqe2.preprocess import idf_matrix_inplace, to_csr32
from sckqe2.rule import data_entropy

X_ = np.array([[0, 1], [1, 0]], complex); Y_ = np.array([[0, -1j], [1j, 0]]); Z_ = np.diag([1.0, -1.0]).astype(complex)
ry = lambda a: np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]], complex)
rz = lambda a: np.diag([np.exp(-0.5j * a), np.exp(0.5j * a)])


def brute_state(x, s):
    l = len(x); psi = np.zeros(2 ** l, complex); psi[0] = 1
    op = lambda U, q: np.kron(np.kron(np.eye(2 ** q), U), np.eye(2 ** (l - q - 1)))
    for t in range(T_LAYERS):
        for q in range(l): psi = op(ry((2.0 ** -t) * s * x[q]), q) @ psi
        for q in range(l): psi = op(rz(s * x[q]), q) @ psi
    return psi, op


def test_closed_form_matches_statevector():
    rng = np.random.RandomState(0); l, s = 4, 0.19
    for _ in range(20):
        x = rng.randn(l) * 3; psi, op = brute_state(x, s); R = bloch_vectors(x[None, :], s)[0]
        for q in range(l):
            for P, j in ((X_, 0), (Y_, 1), (Z_, 2)):
                assert abs(np.vdot(psi, op(P, q) @ psi).real - R[q, j]) < 1e-12
        # <Z_i Z_j> of a product state = <Z_i><Z_j>
        assert abs(np.vdot(psi, op(Z_, 0) @ op(Z_, 1) @ psi).real - R[0, 2] * R[1, 2]) < 1e-12


def test_unit_bloch_vectors_and_fidelity():
    rng = np.random.RandomState(1); E = rng.randn(50, 12) * 2; R = bloch_vectors(E, 0.19)
    assert np.allclose((R ** 2).sum(-1), 1, atol=1e-12)
    assert np.allclose(fidelity(R[0], R[0]), 1)
    assert lipschitz(0.19) == pytest.approx(0.19 * (3 + 1 + 0.5 + 0.25))


def test_backends_agree():
    torch = pytest.importorskip("torch")
    rng = np.random.RandomState(2); E = rng.randn(300, 132).astype(np.float32)
    a = encode(E, backend="closed_form"); b = encode(E, backend="statevector", device="cpu")
    assert a.shape == b.shape == (300, 22 * 33)
    assert np.max(np.abs(a - b)) < 1e-4


def test_exact_decomposition_matches_dense_svd():
    rng = np.random.RandomState(3); M = sp.random(400, 120, density=0.1, random_state=rng, format="csr", dtype=np.float32)
    Xi = idf_matrix_inplace(to_csr32(M).copy())
    E = exact_scores(Xi, 10, whiten=True, tol=1e-12)
    D = Xi.toarray().astype(np.float64); D -= D.mean(0); U = np.linalg.svd(D, full_matrices=False)[0][:, :10] * np.sqrt(400)
    for i in range(10):
        assert abs(abs(np.corrcoef(E[:, i], U[:, i])[0, 1]) - 1) < 1e-6


def test_bit_identical_rerun():
    rng = np.random.RandomState(4); M = sp.random(600, 300, density=0.08, random_state=rng, format="csr", dtype=np.float32)
    M.data = np.ceil(M.data * 10)
    f = lambda: encode(exact_scores(idf_matrix_inplace(to_csr32(M).copy()), 12, tol=1e-6), backend="closed_form")
    assert np.array_equal(f(), f())


def test_alignment_rotation_identity():
    a, x, d = 0.19, 1.7, 0.6
    for R in (ry, rz):
        assert np.allclose(R(a * x) @ R(-a * d), R(a * (x - d)), atol=1e-15)


def test_data_entropy():
    assert data_entropy([1, 1, 1, 1]) == pytest.approx(2.0)
    assert data_entropy([5]) == pytest.approx(0.0)


# ------------------------------------------------------------------ added in 2.0: PSC and certified exact nearest states
def _two_block_graph(sizes, p_in=0.3, p_out=0.002, seed=0):
    rng = np.random.RandomState(seed); n = sum(sizes); lab = np.repeat(np.arange(len(sizes)), sizes)
    P = np.where(lab[:, None] == lab[None, :], p_in, p_out); A = (rng.rand(n, n) < P).astype(float); A = np.triu(A, 1); A = A + A.T
    return sp.csr_matrix(A), lab


def test_psc_merges_split_pieces_and_is_deterministic():
    from sckqe2.psc import merge_to_k
    A, lab = _two_block_graph([60, 60, 20])
    pre = lab.copy(); pre[:30] = 3                     # split the first block in two pieces
    out = merge_to_k(pre, A, 3)
    assert len(np.unique(out)) == 3
    assert len(np.unique(out[:60])) == 1               # the two pieces of block 0 are joined again
    assert np.array_equal(out, merge_to_k(pre, A, 3))


def test_leiden_psc_reproducible():
    pytest.importorskip("igraph"); pytest.importorskip("leidenalg")
    from sckqe2.psc import leiden_psc
    A, lab = _two_block_graph([80, 80, 40], seed=1)
    a, _ = leiden_psc(A, 3); b, _ = leiden_psc(A, 3)
    assert np.array_equal(a, b) and len(np.unique(a)) == 3


def test_certified_knn_equals_brute_force():
    pytest.importorskip("torch")
    from sckqe2.qstate import certified_knn, state_distance, bloch_chunked
    rng = np.random.RandomState(5); E = rng.randn(400, 24) * 1.5; R = bloch_chunked(E, 0.19)
    I, D, cert = certified_knn(R, k=10, candidates=30, device="cpu")
    full = np.stack([state_distance(R[i][None], R) for i in range(len(R))])
    for i in range(len(R)):
        assert np.allclose(np.sort(full[i])[:10], np.sort(D[i]), atol=1e-4)
    assert cert.mean() > 0.5


def test_fidelity_lower_bound_theorem4():
    from sckqe2.qstate import state_distance
    rng = np.random.RandomState(6); E = rng.randn(200, 16) * 2; R = bloch_vectors(E, 0.19)
    f = R.reshape(len(R), -1)
    for i in range(0, 200, 7):
        d = state_distance(R[i][None], R); feat = ((f - f[i]) ** 2).sum(1) / 4
        assert np.all(d >= feat - 1e-9)


def test_qiskit_circuit_matches_closed_form():
    pytest.importorskip("qiskit")
    from sckqe2.hardware import simulate, exact_expectations
    rng = np.random.RandomState(7)
    for _ in range(5):
        x = rng.randn(6) * 2
        assert np.allclose(simulate(x), exact_expectations(x), atol=1e-10)
