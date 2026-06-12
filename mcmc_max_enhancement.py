"""Find the (k, cos theta) that maximise the drift-driven enhancement of the
baryon perturbation, by exploring eq. D50 (Shalaby & Broderick 2026,
arXiv:2604.22665) with an MCMC.

The figure of merit is the Fig. 5 enhancement at z = z_eval (default 0),

    E(k, mu) = |delta_b(z_eval; k, mu, drift)| / |delta_b(z_eval; k, no drift)| ,

with mu = cos(theta) the angle between the wave vector and the DM drift.
emcee samples ln-prob = beta * ln E with flat priors ln k in
[ln kmin, ln kmax] (default 5e2..1e4 /Mpc) and mu in (0, 1); the chain
concentrates around the maximally enhanced modes and the best sample is used
to draw a Fig. 5-style plot of that mode against the no-drift case.

Outputs:
    mcmc_enhancement_samples.png   - sample cloud in (k, mu) coloured by E
    figure5_best_enhancement.png   - Fig. 5-style plot at the best (k, mu)
    outputs/mcmc_enhancement_chain.npz - flat chain + ln-prob

Usage:
    python mcmc_max_enhancement.py [--backend class|camb] [--kmin 5e2]
        [--kmax 1e4] [--z-eval 0] [--nwalkers 32] [--nsteps 400]
        [--burn 100] [--beta 10] [--seed 42]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import emcee

from cosmo_params import A_INIT, C_CMS, H0_S, MPC_CM, Z_INIT, \
    primordial_curvature_rms
from linear_growth import rhs, evolve, TABLE_DIR

_TABLES = {}


def initial_conditions(backend, k):
    """Cached version of linear_growth.load_initial_conditions (the table is
    read once; the MCMC calls this tens of thousands of times)."""
    if backend not in _TABLES:
        path = os.path.join(TABLE_DIR, f"{backend}_transfer_z{Z_INIT:g}.txt")
        _TABLES[backend] = np.loadtxt(path)
    tab = _TABLES[backend]
    lnk = np.log(tab[:, 0])
    d_b, d_c, th_b, th_c = (np.interp(np.log(k), lnk, tab[:, i])
                            for i in (1, 2, 3, 4))
    norm = primordial_curvature_rms(k)
    theta_to_t = (C_CMS / MPC_CM) / (A_INIT * H0_S)
    return [d_b * norm + 0j, th_b * norm * theta_to_t + 0j,
            d_c * norm + 0j, th_c * norm * theta_to_t + 0j]


def delta_b_final(backend, k, mu, drift, a_end):
    """|delta_b| at a_end from eq. D50 (no dense output: final value only)."""
    sol = solve_ivp(rhs, (np.log(A_INIT), np.log(a_end)),
                    initial_conditions(backend, k), method='DOP853',
                    rtol=1e-8, atol=1e-14, args=(k, mu, drift, True))
    if not sol.success:
        return np.nan
    return abs(sol.y[0, -1])


def enhancement(backend, k, mu, a_end):
    num = delta_b_final(backend, k, mu, True, a_end)
    den = delta_b_final(backend, k, 0.0, False, a_end)
    return num / den


def make_lnprob(backend, lnk_min, lnk_max, a_end, beta):
    def lnprob(x):
        lnk, mu = x
        if not (lnk_min <= lnk <= lnk_max and 0.0 < mu < 1.0):
            return -np.inf
        E = enhancement(backend, np.exp(lnk), mu, a_end)
        if not np.isfinite(E) or E <= 0:
            return -np.inf
        return beta * np.log(E)
    return lnprob


def plot_samples(flat, lnp, beta, best, out):
    k, mu, E = np.exp(flat[:, 0]), flat[:, 1], np.exp(lnp / beta)
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    sc = ax.scatter(k, mu, c=E, s=4, cmap='viridis', rasterized=True)
    ax.scatter([best[0]], [best[1]], marker='*', s=250, ec='r', fc='gold',
               lw=1.2, zorder=5,
               label=rf'best: $k={best[0]:.0f}/$Mpc, $\cos\theta={best[1]:.3f}$,'
                     rf' $E={best[2]:.2f}$')
    ax.set_xscale('log')
    ax.set_xlabel(r'$k$ [1/Mpc]')
    ax.set_ylabel(r'$\cos\theta$')
    ax.set_title(r'MCMC samples of the enhancement '
                 r'$E = |\delta_b|_{\rm drift}/|\delta_b|_{\rm no\,drift}$'
                 '\n(eq. D50, evaluated at $z = z_{\\rm eval}$)', fontsize=10)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, which='both', ls=':', lw=0.5, alpha=0.6)
    fig.colorbar(sc, ax=ax, label=r'$E$')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"sample-cloud figure saved to {out}")


def plot_best_fig5(backend, best, out):
    k_b, mu_b, E_b = best
    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    for label, mu, drift, color, ls in [
            (rf'$\cos\theta = {mu_b:.3f}$ (MCMC best)', mu_b, True, 'b', '-'),
            (r'$v_{\rm DM} = 0$  no relative drift', 0.0, False, 'r', '-'),
            (r'$\cos\theta = 1$  supersonic', 1.0, True, 'r', '--')]:
        zp1, y = evolve(initial_conditions(backend, k_b), k_b, mu, drift)
        ax.loglog(zp1, np.abs(y[0]), color=color, ls=ls, lw=1.4, label=label)
    ax.grid(True, which='both', ls=':', lw=0.5, alpha=0.7)
    ax.set_xlim(1.0 + Z_INIT, 1.0)
    ax.set_xlabel(r'$z + 1$')
    ax.set_ylabel(r'$\sqrt{|\delta_b|^2}$')
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    ax.set_title(f'Maximally enhanced mode (ICs from {backend.upper()}):  '
                 rf'$k = {k_b:.0f}\,/{{\rm Mpc}}$, enhancement $E = {E_b:.2f}$',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"Fig. 5-style figure saved to {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--backend', choices=['class', 'camb'], default='class')
    ap.add_argument('--kmin', type=float, default=5e2, help='1/Mpc')
    ap.add_argument('--kmax', type=float, default=1e4, help='1/Mpc')
    ap.add_argument('--z-eval', type=float, default=0.0,
                    help='redshift at which the enhancement is measured')
    ap.add_argument('--nwalkers', type=int, default=32)
    ap.add_argument('--nsteps', type=int, default=400)
    ap.add_argument('--burn', type=int, default=100)
    ap.add_argument('--beta', type=float, default=10.0,
                    help='ln-prob sharpening, lnP = beta ln E (larger = '
                         'tighter concentration around the maximum)')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    lnk_min, lnk_max = np.log(args.kmin), np.log(args.kmax)
    a_end = 1.0 / (1.0 + args.z_eval)
    lnprob = make_lnprob(args.backend, lnk_min, lnk_max, a_end, args.beta)

    p0 = np.column_stack([rng.uniform(lnk_min, lnk_max, args.nwalkers),
                          rng.uniform(0.05, 0.95, args.nwalkers)])
    sampler = emcee.EnsembleSampler(args.nwalkers, 2, lnprob)
    print(f"running emcee: {args.nwalkers} walkers x {args.nsteps} steps, "
          f"k in [{args.kmin:g}, {args.kmax:g}]/Mpc, z_eval={args.z_eval:g}, "
          f"beta={args.beta:g} ...")
    sampler.run_mcmc(p0, args.nsteps)
    print(f"mean acceptance fraction: "
          f"{np.mean(sampler.acceptance_fraction):.2f}")

    flat = sampler.get_chain(discard=args.burn, flat=True)
    lnp = sampler.get_log_prob(discard=args.burn, flat=True)
    i_best = int(np.argmax(lnp))
    k_b, mu_b = np.exp(flat[i_best, 0]), flat[i_best, 1]
    E_b = np.exp(lnp[i_best] / args.beta)
    print(f"\nbest enhancement: E = {E_b:.3f} at k = {k_b:.1f}/Mpc, "
          f"cos(theta) = {mu_b:.4f}")
    for q in (16, 50, 84):
        kk, mm = np.exp(np.percentile(flat[:, 0], q)), \
            np.percentile(flat[:, 1], q)
        print(f"   {q:2d}th percentile: k = {kk:8.1f}/Mpc   cos(theta) = {mm:.3f}")

    os.makedirs('outputs', exist_ok=True)
    np.savez(os.path.join('outputs', 'mcmc_enhancement_chain.npz'),
             flat_chain=flat, lnprob=lnp, beta=args.beta,
             best=np.array([k_b, mu_b, E_b]))

    plot_samples(flat, lnp, args.beta, (k_b, mu_b, E_b),
                 'mcmc_enhancement_samples.png')
    plot_best_fig5(args.backend, (k_b, mu_b, E_b),
                   'figure5_best_enhancement.png')


if __name__ == '__main__':
    main()
