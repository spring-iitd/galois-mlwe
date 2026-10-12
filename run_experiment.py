#!/usr/bin/env python3
"""
Three-condition primal-attack experiment on Module-LWE.

Tests whether the Galois action sigma_j: x -> x^j in Z[x]/(x^n+1) strengthens
the primal lattice attack. Each Module-LWE instance is embedded via Kannan's
embedding into a lattice of dimension d = 2kn + 1 and reduced with BKZ under
three input bases:

  (A)  baseline   -- a standard basis B of the Kannan lattice Lambda
  (D') redundant  -- B plus integer combinations of its own rows
                     (the same lattice Lambda from a different starting point)
  (D)  augmented  -- a generating set of Lambda + Lambda_j, where Lambda_j is
                     the lattice obtained by applying sigma_j

All three conditions share the same underlying instance, so differences are
attributable to the basis alone. Recovery is verified coordinate-wise against
the known secret rather than by the heuristic "entries bounded by eta" test,
which admits false positives at small q.

Usage:
    python3 run_experiment.py --instances 8 --cores 16 --timeout 3600

Output:
    figure2_data.csv -- recovery rates per (n, q, beta) for all three conditions
"""

import argparse
import csv
import math
import random
import sys
import time
from multiprocessing import Pool

import numpy as np
from fpylll import BKZ, LLL, IntegerMatrix
from fpylll.algorithms.bkz2 import BKZReduction
from fpylll.util import ReductionError

# ── Fixed scheme parameters (ML-KEM / FIPS 203 style) ──
K = 2      # module rank
ETA = 2    # CBD parameter for secret and error
J = 3      # Galois automorphism exponent: sigma_j(x) = x^j

# ── Experiment grid: (ring degree n, modulus q, block sizes) ──
# q=97 gives gap > 1 (attack feasible); q=17 gives gap ~ 1 (control).
CONFIGS = [
    (16, 97, list(range(2, 30, 2))),
    (16, 17, list(range(2, 30, 2))),
    (32, 97, list(range(2, 50, 2))),
    (32, 17, list(range(2, 50, 2))),
]


def cbd(eta, size, rng):
    """Centered binomial distribution with parameter eta."""
    return np.array([
        sum(rng.randint(0, 1) for _ in range(eta)) -
        sum(rng.randint(0, 1) for _ in range(eta))
        for _ in range(size)
    ], dtype=int)


def negacyclic(poly, n):
    """Negacyclic matrix of a polynomial in Z[x]/(x^n + 1)."""
    mat = np.zeros((n, n), dtype=int)
    for j in range(n):
        for i in range(n):
            idx = i - j
            mat[i, j] = poly[idx] if idx >= 0 else -poly[idx + n]
    return mat


def galois_matrix(j, n):
    """Permutation-with-signs matrix P_j realising sigma_j: x -> x^j."""
    P = np.zeros((n, n), dtype=int)
    for i in range(n):
        r = (i * j) % (2 * n)
        P[r % n, i] = 1 if r < n else -1
    return P


def gen_instance(n, q, seed):
    """Sample a Module-LWE instance (A, s, e, t) with t = A s + e mod q."""
    rng = random.Random(seed)
    m = K * n
    A = np.zeros((m, m), dtype=int)
    for rb in range(K):
        for cb in range(K):
            A[rb * n:(rb + 1) * n, cb * n:(cb + 1) * n] = negacyclic(
                np.array([rng.randint(0, q - 1) for _ in range(n)]), n)
    s = cbd(ETA, m, rng)
    e = cbd(ETA, m, rng)
    t = (A @ s + e) % q
    return A, s, e, t


def build_basis(A, t, n, q):
    """Kannan embedding basis; target vector is y = (s | e | 1)."""
    m = K * n
    d = 2 * m + 1
    B = np.zeros((d, d), dtype=int)
    for i in range(m):
        B[i, i] = 1
        B[i, m:2 * m] = (-A[:, i]) % q
    B[m, m:2 * m] = t % q
    B[m, 2 * m] = 1
    for j in range(m):
        B[m + 1 + j, m + j] = q
    return B


def make_redundant_basis(basis, d):
    """Append integer combinations of existing rows: same lattice, more rows."""
    rng = np.random.RandomState(12345)
    extra = np.zeros((d, d), dtype=int)
    for i in range(d):
        n_terms = rng.randint(2, 5)
        row_indices = rng.choice(d, size=n_terms, replace=False)
        coeffs = rng.randint(-3, 4, size=n_terms)
        for idx, coeff in zip(row_indices, coeffs):
            extra[i] += coeff * basis[idx]
    return np.vstack([basis, extra])


def to_fplll(b):
    nr, nc = b.shape
    mat = IntegerMatrix(nr, nc)
    for i in range(nr):
        for j in range(nc):
            mat[i, j] = int(b[i, j])
    return mat


def extract_square(gen):
    """Reduce an overdetermined generating set to a square basis via LLL."""
    nr, nc = gen.shape
    if nr == nc:
        return gen.copy()
    mat = to_fplll(gen)
    LLL.reduction(mat)
    rows = []
    for i in range(mat.nrows):
        row = np.array([mat[i][j] for j in range(nc)], dtype=int)
        if np.any(row != 0):
            rows.append(row)
    return np.array(rows[:nc], dtype=int)


def check_recovery(mat, s, e, A, t, n, q):
    """Verify against the known secret, not the heuristic norm bound."""
    m = K * n
    d = mat.ncols
    for i in range(mat.nrows):
        row = np.array([mat[i][j] for j in range(d)], dtype=int)
        c = row[2 * m]
        if c == 0:
            continue
        if np.max(np.abs(row[m:2 * m])) > ETA:
            continue
        if np.array_equal((A @ row[:m] + row[m:2 * m]) % q, (c * t) % q):
            return True
    return False


def compute_gamma(q, n, k=K):
    """Gap gamma = GH(Lambda) / ||y||. gamma > 1 means y realises lambda_1."""
    m = k * n
    d = 2 * m + 1
    log_gh = 0.5 * math.log(d / (2 * math.pi * math.e)) + (m / d) * math.log(q)
    return math.exp(log_gh - 0.5 * math.log(d))


def worker_job(job):
    """One (n, q, beta, condition, instance) trial. Runs in a worker process."""
    n, q, beta, cond, inst_idx, inst_seed = job
    m = K * n
    d = 2 * m + 1

    A, s, e, t = gen_instance(n, q, inst_seed)
    basis_A = build_basis(A, t, n, q)

    if cond == 'A':
        basis = basis_A
    elif cond == 'Dp':
        basis = make_redundant_basis(basis_A, d)
    elif cond == 'D':
        P = galois_matrix(J, n)
        Pb = np.zeros((m, m), dtype=int)
        for b in range(K):
            Pb[b * n:(b + 1) * n, b * n:(b + 1) * n] = P
        Aj = Pb @ A @ Pb.T
        tj = Pb @ t
        basis = np.vstack([basis_A, build_basis(Aj, tj, n, q)])

    if basis.shape[0] > basis.shape[1]:
        basis = extract_square(basis)

    try:
        mat = to_fplll(basis.copy())
        LLL.reduction(mat)
        if beta >= 3:
            BKZReduction(mat)(BKZ.Param(block_size=beta, max_loops=8))
        recovered = check_recovery(mat, s, e, A, t, n, q)
    except ReductionError:
        recovered = False

    return (n, q, beta, cond, inst_idx, recovered)


def main():
    parser = argparse.ArgumentParser(
        description='Three-condition primal-attack experiment on Module-LWE')
    parser.add_argument('--instances', type=int, default=8,
                        help='independent instances per curve (default: 8)')
    parser.add_argument('--cores', type=int, default=16,
                        help='worker processes (default: 16)')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='per-configuration time limit in seconds')
    parser.add_argument('--output', default='figure2_data.csv',
                        help='output CSV path')
    args = parser.parse_args()

    print('Galois action / primal attack -- three-condition experiment')
    print(f'Date: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'Parameters: k={K}, eta={ETA}, j={J}')
    print(f'Instances: {args.instances}, cores: {args.cores}, '
          f'timeout: {args.timeout}s/config')
    print(f'Python: {sys.version.split()[0]}, numpy: {np.__version__}')
    print()

    pool = Pool(processes=args.cores)
    all_rows = []
    t_total = time.time()

    for n, q, betas in CONFIGS:
        d = 2 * K * n + 1
        gamma = compute_gamma(q, n)
        print(f'--- n={n}, q={q}, d={d}, gamma={gamma:.2f} ---')

        base_seed = q * 100000 + n * 1000
        inst_seeds = [base_seed + i for i in range(args.instances)]
        t0 = time.time()

        for beta in betas:
            if time.time() - t0 > args.timeout:
                print(f'  TIMEOUT at beta={beta} ({args.timeout}s limit)')
                break

            jobs = [(n, q, beta, cond, i, seed)
                    for cond in ('A', 'Dp', 'D')
                    for i, seed in enumerate(inst_seeds)]

            t_beta = time.time()
            results = pool.map(worker_job, jobs, chunksize=1)
            dt = time.time() - t_beta

            counts = {'A': 0, 'Dp': 0, 'D': 0}
            for (_, _, _, cond, _, rec) in results:
                if rec:
                    counts[cond] += 1

            rates = {c: counts[c] / args.instances for c in counts}
            print(f'  beta={beta:>3}: A={rates["A"]:.3f}  '
                  f"D'={rates['Dp']:.3f}  D={rates['D']:.3f}  "
                  f'({dt:.0f}s, total {time.time() - t0:.0f}s)')

            all_rows.append({
                'n': n, 'q': q, 'd': d, 'gamma': round(gamma, 2), 'beta': beta,
                'baseline': rates['A'],
                'redundant': rates['Dp'],
                'augmented': rates['D'],
            })

        rows_here = [r for r in all_rows if r['n'] == n and r['q'] == q]
        match = sum(1 for r in rows_here if r['baseline'] == r['redundant'])
        print(f"  D'=A in {match}/{len(rows_here)} block sizes  "
              f'({time.time() - t0:.0f}s total)')

        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'n', 'q', 'd', 'gamma', 'beta',
                'baseline', 'redundant', 'augmented'])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f'  [saved {len(all_rows)} rows]\n')

    pool.close()
    pool.join()

    total = time.time() - t_total
    print(f'Saved {args.output} ({len(all_rows)} rows)')
    print(f'Total time: {total:.0f}s ({total / 60:.1f} min)')


if __name__ == '__main__':
    main()
