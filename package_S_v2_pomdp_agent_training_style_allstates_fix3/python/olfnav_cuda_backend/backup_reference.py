from __future__ import annotations

import numpy as np


def reference_backup_numpy(B: np.ndarray, Gamma: np.ndarray, T: np.ndarray, O: np.ndarray, R: np.ndarray, gamma: float):
    """Slow but explicit NumPy/Python backup reference for small validation.

    Shapes:
      B     [nB,nS]
      Gamma [nG,nS]
      T     [nA,nS,nS]
      O     [nO,nA,nS]
      R     [nA,nS]
    Returns:
      BKP [nB,nS], actions [nB]
    """
    B = np.asarray(B, dtype=np.float64)
    Gamma = np.asarray(Gamma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    O = np.asarray(O, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    nB, nS = B.shape
    nG = Gamma.shape[0]
    nA = T.shape[0]
    nO = O.shape[0]

    # GAO[i,o,a,s] = sum_sp O[o,a,sp] * T[a,s,sp] * Gamma[i,sp]
    GAO = np.empty((nG, nO, nA, nS), dtype=np.float64)
    for i in range(nG):
        for o in range(nO):
            for a in range(nA):
                weighted = O[o, a, :] * Gamma[i, :]
                GAO[i, o, a, :] = T[a] @ weighted

    BKP = np.empty((nB, nS), dtype=np.float64)
    actions = np.empty((nB,), dtype=np.int32)
    for j in range(nB):
        GAB = np.empty((nA, nS), dtype=np.float64)
        for a in range(nA):
            acc = np.array(R[a], copy=True)
            for o in range(nO):
                scores = GAO[:, o, a, :] @ B[j]
                i_star = int(np.argmax(scores))
                acc += gamma * GAO[i_star, o, a, :]
            GAB[a, :] = acc
        values = GAB @ B[j]
        a_star = int(np.argmax(values))
        actions[j] = a_star
        BKP[j, :] = GAB[a_star, :]
    return BKP, actions
