# The Galois Orbit Cannot Help the Primal Attack on Module-LWE

Reproduction code for the experiments in *Poster: The Galois Orbit Cannot Help
the Primal Attack on Module-LWE*, PQQS '26 (Milpitas, CA, USA).

Narayan Ji Mishra, Shivam Kumar, Vireshwar Kumar, Ashok Kumar Bhateja —
Indian Institute of Technology Delhi.

The Galois automorphisms of the cyclotomic ring `Z[x]/(x^n + 1)` act on a
Module-LWE instance, producing an orbit of equivalent descriptions. A natural
question is whether an attacker can combine these descriptions — stacking the
rotated lattices — to obtain a shorter target or a denser sublattice, as happens
in the overstretched NTRU setting. This code tests that question directly and
finds the answer is no: the Galois-augmented lattice collapses to `Z^d`, and the
attack never succeeds.

<p align="center">
  <img src="figures/figure_n16_n32.png" width="720" alt="Recovery rate against block size">
</p>

## Experiment design

Each trial samples a Module-LWE instance `(A, t = As + e mod q)` with module
rank `k = 2` and secret/error from a centered binomial distribution with
`eta = 2`, matching the ML-KEM (FIPS 203) error distribution. The instance is
embedded via Kannan's embedding into a lattice `Lambda` of dimension
`d = 2kn + 1`, where the target vector is `y = (s | e | 1)`.

Three conditions share every underlying instance and differ only in the basis
handed to BKZ:

| Condition | Basis | Lattice | Purpose |
|---|---|---|---|
| **(A)** baseline | a standard basis `B` of `Lambda` | `Lambda` | reference |
| **(D′)** redundant | `B` plus integer combinations of its own rows | `Lambda` | controls for generating-set size |
| **(D)** augmented | `B` stacked with the `sigma_j`-rotated basis | `Lambda + Lambda_j` | the Galois strategy under test |

Condition (D′) is the control that matters: it supplies a strictly larger
generating set of the *same* lattice. If (D′) behaved differently from (A), any
effect seen in (D) could be attributed to redundancy rather than to Galois
structure. It does not, so it cannot.

### Why the gap parameter governs everything

The feasibility of the primal attack is set by

```
gamma = GH(Lambda) / ||y||
```

the ratio of the Gaussian heuristic to the target norm. When `gamma > 1` the
target approximately realises `lambda_1(Lambda)`, so BKZ at sufficient block
size will find it. When `gamma < 1` the target is *not* the shortest vector and
no block size suffices — membership `y ∈ Lambda` by construction says nothing
about solvability.

The moduli are chosen to straddle this boundary at each ring degree:

| `n` | `q` | `d` | `gamma` | regime |
|---|---|---|---|---|
| 16 | 97 | 65 | 2.30 | feasible |
| 16 | 17 | 65 | 0.98 | borderline |
| 32 | 97 | 129 | 2.34 | feasible |
| 32 | 17 | 129 | 0.99 | infeasible |

Note that the *smaller* modulus is the harder one. A small `q` packs the lattice
more tightly while the secret and error stay bounded by `eta`, so the target
becomes less distinguishable from generic short vectors.

### Methodology notes

- **Fresh reduction per block size.** Each `beta` runs on an independent copy of
  the lattice, not progressively, so the curve measures the cost of reaching
  that block size rather than the cumulative effect of all smaller ones.
- **Recovery verified against the known secret.** The heuristic test "all
  entries bounded by `eta`" admits false positives at small `q`, so recovery is
  checked coordinate-wise against the secret that was sampled.
- **Deterministic seeding.** Instance seeds derive from `q * 100000 + n * 1000 + i`,
  so a rerun reproduces the same instances.
- **Shared instances across conditions.** All three bases are built from the same
  `(A, s, e, t)`, so differences are attributable to the basis alone.

## Results

The data in `data/figure2_data.csv` covers 74 block sizes across four
configurations, eight instances each.

**(A) and (D′) agree at every single `(beta, instance)` pair.** A redundant
generating set of the same lattice neither helps nor hinders BKZ — the extra
rows are linearly dependent and are eliminated during LLL preprocessing.

**(D) never recovers the secret, in any configuration, at any block size.** The
Galois-augmented generating set spans `Lambda + Lambda_j`, whose reduced basis
has determinant 1: the sum is all of `Z^d`. The modular structure the primal
attack depends on is destroyed. Every non-trivial automorphism exponent `j`
tested saturates the same way, so orbit size is not a free variable.

**Attack cost rises sharply with ring degree.** At `n = 16, q = 97` recovery is
essentially complete from `beta = 4`. At `n = 32, q = 97` — nearly the same gap —
the onset shifts to `beta ≈ 30` and reaches only 0.5 at `beta = 44` before the
timeout. Doubling the ring degree roughly doubles the lattice dimension and
pushes the required block size up steeply, consistent with the `2^Theta(beta)`
cost of sieving.

**Below the gap threshold the attack simply fails.** At `n = 32, q = 17`
(`gamma = 0.99`) recovery is exactly zero across all 23 block sizes tested
through `beta = 46`. At `n = 16, q = 17` (`gamma = 0.98`) the curve is noisy and
non-monotonic — the gap sits right at the threshold, and with eight instances
individual BKZ outcomes behave like coin flips.

## Reading the figure

Conditions (A) and (D′) produce identical curves, which would make one invisible
under the other. The figure therefore draws (D′) as a thick pale band and (A) as
a thin saturated line on top of it: wherever the thin line tracks the centre of
the band, the two conditions agree. Any divergence would be immediately visible
as the line leaving the band. Condition (D) is drawn last, on top, so its flat
zero is not obscured.

## Installation

Requires Python 3.8+ and a C toolchain for `fpylll`.

```bash
# Ubuntu / Debian
sudo apt-get install -y build-essential libgmp-dev libmpfr-dev
pip install -r requirements.txt
```

If `pip install fpylll` fails to build, conda is usually easier:

```bash
conda install -c conda-forge fpylll
```

Verify:

```bash
python3 -c "from fpylll import IntegerMatrix, BKZ, LLL; print('fpylll OK')"
```

## Running

```bash
# Full experiment (~4 hours on 16 cores)
python3 run_experiment.py --instances 8 --cores 16 --timeout 3600

# Regenerate the figure from the committed data
python3 plot_figure.py --input data/figure2_data.csv --output figures/figure_n16_n32
```

Options for `run_experiment.py`:

| Flag | Default | Meaning |
|---|---|---|
| `--instances` | 8 | independent instances per curve |
| `--cores` | 16 | worker processes |
| `--timeout` | 3600 | per-configuration time limit, seconds |
| `--output` | `figure2_data.csv` | output path |

Each `(n, q, beta, condition, instance)` trial is independent, so the work
parallelises cleanly. The script dispatches one block size at a time
(3 conditions × 8 instances = 24 jobs) and prints progress after each, writing
the CSV incrementally after every configuration completes — so a run interrupted
partway still leaves usable data.

Runtime is dominated by `n = 32`, where BKZ at large block sizes in dimension 129
is expensive. The reference run took 253 minutes on 16 cores, with both `n = 32`
configurations hitting the one-hour timeout before exhausting their block-size
range.

## Output format

`data/figure2_data.csv`:

| Column | Meaning |
|---|---|
| `n` | ring degree |
| `q` | modulus |
| `d` | lattice dimension, `2kn + 1` |
| `gamma` | gap `GH(Lambda) / ||y||` |
| `beta` | BKZ block size |
| `baseline` | recovery rate, condition (A) |
| `redundant` | recovery rate, condition (D′) |
| `augmented` | recovery rate, condition (D) |

Recovery rates are fractions of instances recovered, so with 8 instances the
values are multiples of 0.125.

## Files

```
run_experiment.py          three-condition experiment, parallelised
plot_figure.py             publication figure
data/figure2_data.csv      reference results (74 rows, 4 configurations)
figures/                   figure as PDF (vector) and PNG (300 dpi)
requirements.txt           Python dependencies
```

## Scope

The argument covers power-of-two cyclotomics, where the Galois action is
realised by a signed permutation matrix. The proof uses only that this matrix is
monomial. In prime cyclotomics the action is unimodular but not orthogonal, and
that case is not settled here.

## Citation

```bibtex
@inproceedings{mishra2026galois,
  author    = {Mishra, Narayan Ji and Kumar, Shivam and
               Kumar, Vireshwar and Bhateja, Ashok Kumar},
  title     = {Poster: The Galois Orbit Cannot Help the Primal Attack on Module-LWE},
  booktitle = {Post-Quantum and Quantum-based Security (PQQS '26)},
  year      = {2026},
  address   = {Milpitas, CA, USA},
  publisher = {ACM},
  doi       = {10.1145/3839338.3857268}
}
```

## License

Code is released under the MIT License — see [LICENSE](LICENSE).

The paper itself is published under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
