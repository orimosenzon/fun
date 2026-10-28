"""Precompute the numbers the zeta video needs, so rendering stays fast.

Writes data/zeta_data.npz with:
  zeros            imaginary parts of the first N_ZEROS nontrivial zeros
  crit_t, crit_z   zeta(1/2 + it) sampled for t in [0, T_MAX]
  x_grid           sample points for the prime-counting plots
  pi_exact         pi(x) on x_grid
  R_main           Riemann's R(x) (no zeros)
  approx_K         R(x) - sum over the first K zero pairs, for K in K_LIST
"""
from pathlib import Path

import mpmath as mp
import numpy as np

mp.mp.dps = 20

N_ZEROS = 100
K_LIST = [1, 3, 10, 30, 100]
T_MAX = 50
X_MAX = 100
OUT = Path(__file__).resolve().parent.parent / "data" / "zeta_data.npz"


def mobius(n):
    result, m, p = 1, n, 2
    while p * p <= m:
        if m % p == 0:
            m //= p
            if m % p == 0:
                return 0
            result = -result
        p += 1
    return -result if m > 1 else result


def primes_upto(n):
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = False
    return np.nonzero(sieve)[0]


# R(x) = sum_n mu(n)/n * li(x^(1/n)); terms vanish once x^(1/n) < 2
MU = [(n, mobius(n)) for n in range(1, 12) if mobius(n) != 0]


def R_real(x):
    lx = mp.log(x)
    return sum(mu / n * mp.li(mp.e ** (lx / n)) for n, mu in MU if lx / n > 0.3)


def R_zero_pair(x, gamma):
    # Contribution of the zero pair 1/2 +- i*gamma: 2 Re R(x^rho),
    # with li(x^(rho/n)) taken as Ei(rho * log(x) / n) to stay on the right branch.
    lx = mp.log(x)
    rho = mp.mpc(0.5, gamma)
    total = 0
    for n, mu in MU:
        if lx / n < 0.3:
            break
        total += mu / n * mp.ei(rho * lx / n)
    return 2 * mp.re(total)


def main():
    print("zeros...")
    zeros = np.array([float(mp.im(mp.zetazero(k))) for k in range(1, N_ZEROS + 1)])

    print("critical line...")
    crit_t = np.linspace(0, T_MAX, 2501)
    crit_z = np.array([complex(mp.zeta(mp.mpc(0.5, t))) for t in crit_t])

    print("prime counting...")
    x_grid = np.linspace(1.5, X_MAX, 1200)
    primes = primes_upto(X_MAX)
    pi_exact = np.searchsorted(primes, x_grid, side="right").astype(float)
    R_main = np.array([float(R_real(x)) for x in x_grid])

    pair_terms = np.array([[float(R_zero_pair(x, g)) for x in x_grid] for g in zeros])
    cumulative = np.cumsum(pair_terms, axis=0)
    # the -1/log(x) + arctan(pi/log x)/pi tail from the trivial zeros is tiny; include it anyway
    tail = np.array([float(-1 / mp.log(x) + mp.atan(mp.pi / mp.log(x)) / mp.pi) for x in x_grid])
    approx = {f"approx_{K}": R_main + tail - cumulative[K - 1] for K in K_LIST}

    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, zeros=zeros, crit_t=crit_t, crit_z=crit_z, x_grid=x_grid,
             pi_exact=pi_exact, R_main=R_main, primes=primes, **approx)
    err = np.abs(approx[f"approx_{K_LIST[-1]}"] - pi_exact)
    print(f"saved {OUT}; median |error| with {K_LIST[-1]} zeros: {np.median(err):.3f}")


if __name__ == "__main__":
    main()
