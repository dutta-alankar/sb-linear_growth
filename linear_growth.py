"""Reproduce Figure 5 of Shalaby & Broderick (2026), arXiv:2604.22665.

Solves their equation system D50 (Appendix D, eq. `eq:lincosm`): the linear
evolution of coupled baryon and dark-matter perturbations in an expanding
FLRW background in the presence of a uniform DM-baryon relative drift
v_DM(a) = 5 c_s(a_i) (a_i/a), starting at z_i = 1000.  With t_x = theta_x/H0
and s = ln a the system reads

  d(delta_b)/ds = -t_b / h
  d(t_b)/ds     = [c_s^2 k^2 / (a^2 H H0)] delta_b
                  - [3/(2 a^3 h)] (Ob0 delta_b + Oc0 delta_c) - 2 t_b
  d(delta_c)/ds = -t_c / h + i K delta_c
  d(t_c)/ds     =  i K t_c - [3/(2 a^3 h)] (Ob0 delta_b + Oc0 delta_c) - 2 t_c

with h = H/H0 and K = k v_DM cos(theta) / (a H).  (The Hubble-drag term
-2 t_c is kept; it is present in the unnumbered precursor equations of the
paper but dropped, apparently typographically, in the printed Eq. D50.
Use --no-dm-drag to integrate the system exactly as printed.)

Initial conditions (delta_b, delta_c, theta_b, theta_c at z = 1000) are read
from tabulated CLASS or CAMB transfer functions produced by
generate_tables.py, scaled to the rms primordial curvature amplitude
sqrt(A_s (k/k_pivot)^(n_s-1)).

Usage:
    python linear_growth.py [--backend class|camb|both] [--k 1e4]
                            [--no-dm-drag] [--out FIG.png]
"""

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from decimal import Decimal

def fexp(number):
    (sign, digits, exponent) = Decimal(number).as_tuple()
    return len(digits) + exponent - 1

def fman(number):
    return Decimal(number).scaleb(-fexp(number)).normalize()

from cosmo_params import (
    A_INIT, C_CMS, H0_S, MPC_CM, OMEGA_B0, OMEGA_DM0, Z_INIT,
    hubble_norm, sound_speed_cms, sound_speed_adiabatic_asymptote_cms, 
    v_dm_cms, primordial_curvature_rms,
)

TABLE_DIR = "tables"
use_adiabatic_asymptote = False
z_start_sim = 250.0

# the four cases of Figure 5: (label, cos(theta), drift on, color, linestyle)
FIG5_CASES = [
    (r'$\cos\theta = 0.2$   early subsonic', 0.2, True, 'k', '-'),
    (r'$\cos\theta = 0.47$  late subsonic', 0.47, True, 'b', '-'),
    # NB: the paper's Fig. 5 caption and body text disagree on the red
    # linestyles; we follow the body text (solid = no drift, dashed = cos=1).
    (r'$v_{\rm DM} = 0$  no relative drift', 0.0, False, 'r', '-'),
    (r'$\cos\theta = 1$   supersonic', 1.0, True, 'r', '--'),
]


def load_initial_conditions(backend, k):
    """Interpolate tabulated transfer functions at wavenumber k [1/Mpc] and
    return complex initial conditions (delta_b, t_b, delta_c, t_c) at Z_INIT,
    normalised to the rms primordial curvature amplitude."""
    path = os.path.join(TABLE_DIR, f"{backend}_transfer_z{Z_INIT:g}.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found - run 'python generate_tables.py' first.")
    tab = np.loadtxt(path)
    kt = tab[:, 0]
    if not kt[0] <= k <= kt[-1]:
        raise ValueError(f"k = {k:g}/Mpc outside table range "
                         f"[{kt[0]:.3g}, {kt[-1]:.3g}]")
    lnk = np.log(kt)
    d_b, d_c, th_b, th_c = (np.interp(np.log(k), lnk, tab[:, i])
                            for i in (1, 2, 3, 4))

    # amplitude normalisation: transfer functions are per unit curvature
    norm = primordial_curvature_rms(k)
    # theta [1/Mpc, conformal, c=1]  ->  t = theta_paper/H0 (dimensionless):
    # theta_paper = (1/a) d(delta)/dtau|_phys = theta_table * (c/Mpc) / a
    theta_to_t = (C_CMS / MPC_CM) / (A_INIT * H0_S)

    delta_b = d_b * norm + 0j
    delta_c = d_c * norm + 0j
    t_b = th_b * norm * theta_to_t + 0j
    t_c = th_c * norm * theta_to_t + 0j
    return delta_b, t_b, delta_c, t_c


def rhs(s, y, k, cos_theta, drift, dm_drag):
    """RHS of eq. D50 in s = ln a; y = [delta_b, t_b, delta_c, t_c] (complex)."""
    a = np.exp(s)
    h = hubble_norm(a)          # H/H0
    H = h * H0_S                # 1/s
    delta_b, t_b, delta_c, t_c = y

    grav = 1.5 / (a**3 * h) * (OMEGA_B0 * delta_b + OMEGA_DM0 * delta_c)
    if not use_adiabatic_asymptote:
        press = (sound_speed_cms(a) * k / MPC_CM)**2 / (a**2 * H * H0_S) * delta_b
    else:
        press = (sound_speed_adiabatic_asymptote_cms(a, z_start_sim) * k / MPC_CM)**2 / (a**2 * H * H0_S) * delta_b  

    if drift:
        K = k / MPC_CM * v_dm_cms(a) * cos_theta / (a * H)
    else:
        K = 0.0

    ddelta_b = -t_b / h
    dt_b = press - grav - 2.0 * t_b
    ddelta_c = -t_c / h + 1j * K * delta_c
    dt_c = 1j * K * t_c - grav - (2.0 * t_c if dm_drag else 0.0)
    return [ddelta_b, dt_b, ddelta_c, dt_c]


def evolve(y0, k, cos_theta, drift, dm_drag=True, n_out=2000):
    s_span = (np.log(A_INIT), 0.0)
    s_eval = np.linspace(*s_span, n_out)
    sol = solve_ivp(rhs, s_span, y0, t_eval=s_eval, method='DOP853',
                    rtol=1e-9, atol=1e-15,
                    args=(k, cos_theta, drift, dm_drag))
    if not sol.success:
        raise RuntimeError(f"ODE integration failed: {sol.message}")
    zp1 = 1.0 / np.exp(sol.t)   # z + 1
    return zp1, sol.y


def run_backend(backend, k, dm_drag, ax):
    y0 = load_initial_conditions(backend, k)
    print(f"[{backend.upper()}] initial conditions at z = {Z_INIT:g}, "
          f"k = {k:g}/Mpc:")
    print(f"   delta_b = {y0[0].real:+.4e}   t_b = {y0[1].real:+.4e}")
    print(f"   delta_c = {y0[2].real:+.4e}   t_c = {y0[3].real:+.4e}")

    curves = {}
    for label, cth, drift, color, ls in FIG5_CASES:
        zp1, y = evolve(list(y0), k, cth, drift, dm_drag)
        curves[label] = (zp1, np.abs(y[0]))
        ax.loglog(zp1, np.abs(y[0]), color=color, ls=ls, lw=1.4, label=label)
        print(f"   {label:42s} |delta_b|(z=0) = {np.abs(y[0][-1]):.4e}")

    # save the evolution curves as a table
    os.makedirs("outputs", exist_ok=True)
    zp1 = next(iter(curves.values()))[0]
    cols = [zp1] + [c[1] for c in curves.values()]
    hdr = ("Evolution of |delta_b| for k = %g /Mpc (Fig. 5, arXiv:2604.22665)\n"
           "z+1, then |delta_b| for cases: " % k
           + "; ".join(l.replace('$', '').replace('\\', '') for l in curves))
    out = os.path.join("outputs", f"delta_b_evolution_{backend}_k{k:g}.txt")
    np.savetxt(out, np.column_stack(cols), header=hdr)
    print(f"   saved {out}")

    ax.grid(True, which='both', ls=':', lw=0.5, alpha=0.7)
    ax.set_xlim(1.0 + Z_INIT, 1.0)
    ax.set_xlabel(r'$z + 1$')
    ax.set_ylabel(r'$\sqrt{|\delta_b|^2}$')
    ax.legend(loc='upper left', fontsize=8, frameon=False)
    ax.set_title(f'ICs from {backend.upper()},  $k = {k:g}\\,/{{\\rm Mpc}}$',
                 fontsize=10)
    ax.text(0.97, 0.05, rf'$k = {fman(k)} \times 10^{{{fexp(k)}}}/{{\rm Mpc}}$',
            transform=ax.transAxes, ha='right',
            bbox=dict(boxstyle='round', fc='lightyellow', ec='olive'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--backend', choices=['class', 'camb', 'both'],
                    default='class',
                    help="Boltzmann code used for the initial conditions "
                         "(paper uses CLASS; default class)")
    ap.add_argument('--k', type=float, default=1.0e4,
                    help='comoving wavenumber in 1/Mpc (default 1e4, as Fig. 5)')
    ap.add_argument('--no-dm-drag', action='store_true',
                    help='drop the -2 t_DM Hubble-drag term, i.e. integrate '
                         'Eq. D50 exactly as printed in the paper')
    ap.add_argument('--out', default=None, help='output figure name')
    args = ap.parse_args()

    backends = ['class', 'camb'] if args.backend == 'both' else [args.backend]
    fig, axes = plt.subplots(1, len(backends),
                             figsize=(5.2 * len(backends), 4.2), squeeze=False)
    for ax, backend in zip(axes[0], backends):
        run_backend(backend, args.k, not args.no_dm_drag, ax)

    # fig.suptitle('Cosmological evolution of the baryon density perturbation\n'
    #              '(reproduction of Fig. 5, Shalaby & Broderick, arXiv:2604.22665)',
    #              fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = args.out or f"figure5_{args.backend}.png"
    fig.savefig(out, dpi=200)
    print(f"figure saved to {out}")


if __name__ == '__main__':
    main()
