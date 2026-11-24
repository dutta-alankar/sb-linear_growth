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
                            [--no-dm-drag] [--temperature th2010|class|camb] [--out FIG.png]
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
    sound_speed_adiabatic_fit_cms, sound_speed_table_cms,
    v_dm_cms, primordial_curvature_rms,
)

TABLE_DIR = "tables"
use_adiabatic_asymptote = False  # if True, use the adiabatic asymptote for c_s in the linear evolution
use_adiabatic_fit = False  # if True, use the adiabatic fit for c_s in the linear evolution
# mean baryon temperature in c_s^2 = gamma k_B T_b / (mu m_p): 'th2010' (fit, default), or 'class' / 'camb' for the
# T_b(z) tables tables/{class,camb}_baryon_temperature.txt written by 'generate_tables.py --temperature-only';
# set with --temperature
baryon_temperature_model = 'th2010'

if use_adiabatic_asymptote and use_adiabatic_fit:
    raise ValueError("Cannot use both adiabatic asymptote and fit for c_s; choose one.")


def check_sound_speed_options():
    if baryon_temperature_model not in ('th2010', 'class', 'camb'):
        raise ValueError(f"unknown baryon_temperature_model '{baryon_temperature_model}'")
    if baryon_temperature_model != 'th2010' and (use_adiabatic_asymptote or use_adiabatic_fit):
        raise ValueError("a CLASS/CAMB temperature cannot be combined with the adiabatic asymptote / fit options.")


def output_suffix():
    """file-name suffix of the chosen sound-speed model"""
    if baryon_temperature_model != 'th2010':
        return f"_{baryon_temperature_model}_Tb"
    if use_adiabatic_asymptote:
        return "_adiabatic_asymptote"
    if use_adiabatic_fit:
        return "_adiabatic_fit"
    return ""

z_start_sim = 250.0

# the four cases of Figure 5: (label, cos(theta), drift on, color, linestyle)
FIG5_CASES = [
    (r'$\cos\theta = 0.2$   early subsonic', 0.2, True, 'k', '-'),
    (r'$\cos\theta = 0.5$  late subsonic', 0.50257, True, 'b', '-'),
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
    if baryon_temperature_model != 'th2010':
        press = (sound_speed_table_cms(a, baryon_temperature_model) * k / MPC_CM)**2 / (a**2 * H * H0_S) * delta_b
    elif not use_adiabatic_asymptote and not use_adiabatic_fit:
        press = (sound_speed_cms(a) * k / MPC_CM)**2 / (a**2 * H * H0_S) * delta_b
    elif use_adiabatic_asymptote:
        press = (sound_speed_adiabatic_asymptote_cms(a, z_start_sim) * k / MPC_CM)**2 / (a**2 * H * H0_S) * delta_b
    elif use_adiabatic_fit:
        press = (sound_speed_adiabatic_fit_cms(a, z_start_sim) * k / MPC_CM)**2 / (a**2 * H * H0_S) * delta_b

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


def find_crossings(zp1, y_case, y_ref):
    """Redshifts at which curve y_case(z) crosses the reference y_ref(z).

    Sign changes of r = ln(y_case/y_ref) are located on the (ln(z+1), r) grid
    and refined by linear interpolation there.  The first grid point is skipped:
    all cases share the same initial conditions, so they coincide there by
    construction.  Returns (z_cross, min|r|) - the latter is the closest
    approach (in ln ratio) over the whole range, useful when there is no
    crossing at all."""
    lz = np.log(zp1)
    r = np.log(y_case) - np.log(y_ref)
    z_cross = []
    for i in range(1, len(r) - 1):
        if r[i] == 0.0:
            z_cross.append(zp1[i] - 1.0)
        elif r[i] * r[i + 1] < 0.0:
            f = r[i] / (r[i] - r[i + 1])
            z_cross.append(np.exp(lz[i] + f * (lz[i + 1] - lz[i])) - 1.0)
    if r[-1] == 0.0:
        z_cross.append(zp1[-1] - 1.0)
    return np.array(z_cross), np.min(np.abs(r[1:]))


def report_crossings(curves, cos_thetas):
    """Print the redshifts where sqrt(|delta_b|^2) of each case with
    0 < cos(theta) < 1 matches the no-drift (cos(theta) = 0) case."""
    ref_labels = [l for l in curves if cos_thetas[l] == 0.0]
    if not ref_labels:
        return
    ref = ref_labels[0]
    zp1, db_ref = curves[ref][0], curves[ref][1]
    print(f"   crossings of $\\sqrt{{|\\delta_b|^2}}$ with the "
          f"cos(theta) = 0 case ({ref.replace('$', '')}):")
    for label in curves:
        cth = cos_thetas[label]
        if not (0.0 < cth < 1.0):
            continue
        z_cross, min_lnratio = find_crossings(zp1, curves[label][1], db_ref)
        if len(z_cross) == 0:
            print(f"     {label:42s} no crossing (closest approach: within a "
                  f"factor {np.exp(min_lnratio):.4f})")
        else:
            zs = ", ".join(f"z = {z:.4f}" for z in z_cross)
            print(f"     {label:42s} {len(z_cross)} crossing(s): {zs}")


def run_backend(backend, k, dm_drag, ax_b, ax_c):
    """Evolve the four Fig-5 cases and plot |delta_b| on ax_b (top) and |delta_c| on
    ax_c (bottom) for this backend's column."""
    y0 = load_initial_conditions(backend, k)
    print(f"[{backend.upper()}] initial conditions at z = {Z_INIT:g}, "
          f"k = {k:g}/Mpc:")
    print(f"   delta_b = {y0[0].real:+.4e}   t_b = {y0[1].real:+.4e}")
    print(f"   delta_c = {y0[2].real:+.4e}   t_c = {y0[3].real:+.4e}")

    curves = {}   # label -> (zp1, |delta_b|, |delta_c|)
    cos_thetas = {}   # label -> cos(theta) actually seen by the case
    for label, cth, drift, color, ls in FIG5_CASES:
        zp1, y = evolve(list(y0), k, cth, drift, dm_drag)
        db, dc = np.abs(y[0]), np.abs(y[2])
        curves[label] = (zp1, db, dc)
        cos_thetas[label] = cth if drift else 0.0
        ax_b.loglog(zp1, db, color=color, ls=ls, lw=1.4, label=label)
        ax_c.loglog(zp1, dc, color=color, ls=ls, lw=1.4, label=label)
        print(f"   {label:42s} |delta_b|(z=0) = {db[-1]:.4e}   |delta_c|(z=0) = {dc[-1]:.4e}")

    report_crossings(curves, cos_thetas)

    # save the evolution curves as a table: z+1, then |delta_b| per case, then |delta_c| per case
    os.makedirs("outputs", exist_ok=True)
    zp1 = next(iter(curves.values()))[0]
    cols = [zp1] + [c[1] for c in curves.values()] + [c[2] for c in curves.values()]
    case_list = "; ".join(l.replace('$', '').replace('\\', '') for l in curves)
    hdr = ("Evolution of |delta_b| and |delta_c| for k = %g /Mpc (Fig. 5, arXiv:2604.22665)\n"
           "z+1, then |delta_b| for cases, then |delta_c| for cases: " % k + case_list)
    out = os.path.join("outputs", f"delta_bc_evolution_{backend}_k{k:g}{output_suffix()}.txt")
    np.savetxt(out, np.column_stack(cols), header=hdr)
    print(f"   saved {out}")

    # top row: delta_b (Fig-5 y-range); bottom row: delta_c (autoscaled, it grows larger)
    for ax in (ax_b, ax_c):
        ax.grid(True, which='both', ls=':', lw=0.5, alpha=0.7)
        ax.set_xlim(1.0 + Z_INIT, 1.0)
    ax_b.set_ylim(5.0e-08, 2.2)
    ax_b.set_ylabel(r'$\sqrt{|\delta_b|^2}$')
    ax_c.set_ylabel(r'$\sqrt{|\delta_c|^2}$')
    ax_c.set_xlabel(r'$z + 1$')                       # x-label on the bottom row only
    ax_b.legend(loc='upper left', fontsize=8, frameon=False)
    ax_b.set_title(f'ICs from {backend.upper()},  $k = {k:g}\\,/{{\\rm Mpc}}$',
                   fontsize=10)
    for ax in (ax_b, ax_c):
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
    ap.add_argument('--temperature', choices=['th2010', 'class', 'camb'], default=None,
                    help='mean baryon temperature in the sound speed: the Tseliakhovich & Hirata (2010) fit '
                         '(default) or the CLASS / CAMB T_b(z) table written by generate_tables.py --temperature-only')
    ap.add_argument('--out', default=None, help='output figure name')
    args = ap.parse_args()
    global baryon_temperature_model
    if args.temperature is not None:
        baryon_temperature_model = args.temperature
    check_sound_speed_options()

    backends = ['class', 'camb'] if args.backend == 'both' else [args.backend]
    # rows: delta_b (top), delta_c (bottom); columns: one per backend
    fig, axes = plt.subplots(2, len(backends),
                             figsize=(5.2 * len(backends), 7.6), squeeze=False, sharex=True)
    for j, backend in enumerate(backends):
        run_backend(backend, args.k, not args.no_dm_drag, axes[0][j], axes[1][j])

    # fig.suptitle('Cosmological evolution of the baryon density perturbation\n'
    #              '(reproduction of Fig. 5, Shalaby & Broderick, arXiv:2604.22665)',
    #              fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = args.out or f"figure5_{args.backend}{output_suffix()}.png"
    fig.savefig(out, dpi=200)
    print(f"figure saved to {out}")


if __name__ == '__main__':
    main()
