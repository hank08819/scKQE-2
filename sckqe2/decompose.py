"""Expression directions by an exact singular value decomposition (Methods): ARPACK (implicitly restarted Lanczos) on the
implicitly centered matrix, from a fixed start vector of ones. No random number is drawn, so two runs give the same scores."""
import numpy as np


def exact_scores(Xi, L=132, whiten=True, tol=1e-6, return_sv=False):
    """Leading L directions of the column-centered Xi (not formed). Sign convention: the largest-magnitude entry of every
    score vector is positive. whiten=True returns U * sqrt(n) (unit variance per direction)."""
    from scipy.sparse.linalg import LinearOperator, svds
    n, p = Xi.shape
    mu = np.asarray(Xi.mean(0)).ravel().astype(np.float64); XiT = Xi.T
    mv = lambda v: Xi @ np.asarray(v, np.float64).ravel() - float(mu @ np.asarray(v, np.float64).ravel())
    rmv = lambda u: XiT @ np.asarray(u, np.float64).ravel() - mu * float(np.asarray(u, np.float64).sum())
    mm = lambda V: Xi @ np.asarray(V, np.float64) - (mu @ np.asarray(V, np.float64))[None, :]
    rmm = lambda U: XiT @ np.asarray(U, np.float64) - np.outer(mu, np.asarray(U, np.float64).sum(0))
    A = LinearOperator((n, p), matvec=mv, rmatvec=rmv, matmat=mm, rmatmat=rmm, dtype=np.float64)
    v0 = np.ones(min(n, p)) / np.sqrt(min(n, p))
    U, s, _ = svds(A, k=L, tol=tol, v0=v0, solver="arpack", maxiter=100000)
    o = np.argsort(s)[::-1]; U = U[:, o]; s = s[o]
    U = U * np.sign(U[np.argmax(np.abs(U), axis=0), np.arange(L)])[None, :]
    if whiten:
        E = (U * np.sqrt(n)).astype(np.float32)
    else:
        E = U * s[None, :]; E = (E / E[:, 0].std()).astype(np.float32)
    return (E, s) if return_sv else E
