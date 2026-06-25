"""Generate CAMB-format transfer-function / power-spectrum tables for setting
cosmological-simulation initial conditions, consistent with the cosmology of
Shalaby & Broderick (2026), arXiv:2604.22665 (their Appendix D / Eq. D50):

    h = 0.71, Omega_b = 0.044, Omega_DM = 0.226, Omega_Lambda = 0.73,
    massless neutrinos only; A_s, n_s as in cosmo_params.py.

For each requested redshift two files are written in exactly the format of
the Fortran-CAMB samples in extras/ (so an existing reader keeps working):

  <root>_z<Z>_transfer_k.dat : 13 columns
      k/h [h/Mpc], then Delta_i/k^2 (k in 1/Mpc, unit primordial curvature)
      for CDM, baryon, photon, massless nu, massive nu (=0), total, no-nu
      total, total-de, Weyl potential, and the Newtonian-gauge velocity
      transfers v_Newt_cdm, v_Newt_baryon, v_baryon-cdm.
  <root>_z<Z>_power_k.dat    : k/h [h/Mpc], P_tot(k) [(Mpc/h)^3]
      with P = (2 pi^2 / k^3) A_s (k/k_p)^(n_s-1) (Delta_tot)^2 h^3
      (verified to reproduce the sample files given their cosmology).

Either CAMB or CLASS can produce the tables (--backend); CLASS output is
converted to the CAMB conventions above.  The default k range spans ~9
decades up to 3e4/Mpc so the paper's k = 1e4/Mpc mode is well inside it.

Usage:
    python transfer_ic_for_cosmo_sim.py [--backend camb|class]
        [--redshifts 1000 900 250 200 127 0] [--kmax 3e4]
        [--k-per-decade 30] [--output-root ic_tables/<backend>_massless]
"""

import argparse
import os
import time

import numpy as np

from cosmo_params import (
    A_S, N_S, K_PIVOT, LITTLE_H, OMEGA_B0, OMEGA_DM0, H0_KMS_MPC,
    primordial_curvature_rms,
)

C_KMS = 299792.458
N_UR = 3.046  # massless neutrinos, as in the sample ini


# ---------------------------------------------------------------- formatting
def fortran_e(x, digits):
    """Format x like Fortran E-descriptor output: 0.526580E-05."""
    if x == 0.0:
        return f"0.{'0' * digits}E+00"
    sign = '-' if x < 0 else ''
    mant, exp = f"{abs(x):.{digits - 1}e}".split('e')
    return f"{sign}0.{mant.replace('.', '')}E{int(exp) + 1:+03d}"


def write_transfer_file(path, cols):
    with open(path, 'w') as f:
        for row in np.column_stack(cols):
            f.write(''.join(fortran_e(v, 6).rjust(14) for v in row) + '\n')
    print(f"  wrote {path}  ({len(cols[0])} rows)")


def write_power_file(path, kh, pk):
    with open(path, 'w') as f:
        for k, p in zip(kh, pk):
            f.write(fortran_e(k, 5).rjust(15) + fortran_e(p, 5).rjust(15) + '\n')
    print(f"  wrote {path}  ({len(kh)} rows)")


def power_total(k_mpc, T_tot):
    """P_tot(k) in (Mpc/h)^3 from the total transfer column (Delta/k^2)."""
    delta2 = (T_tot * k_mpc**2) ** 2 * primordial_curvature_rms(k_mpc) ** 2
    return (2.0 * np.pi**2 / k_mpc**3) * delta2 * LITTLE_H**3


def emit(root, z, k_mpc, cols13):
    base = f"{root}_z{z:f}"
    write_transfer_file(base + "_transfer_k.dat", cols13)
    write_power_file(base + "_power_k.dat", cols13[0], power_total(k_mpc, cols13[6]))


# ------------------------------------------------------------------ backends
def run_camb(redshifts, kmax, k_per_decade, root):
    import camb
    from camb import model
    
    k_per_logint = int(round(k_per_decade / np.log(10.0)))
    print(f"CAMB: transfer functions at z = {redshifts}, kmax = {kmax:.3g}/Mpc ...")
    t0 = time.time()
    pars = camb.set_params(
        H0=H0_KMS_MPC, ombh2=OMEGA_B0 * LITTLE_H**2, omch2=OMEGA_DM0 * LITTLE_H**2,
        As=A_S, ns=N_S, pivot_scalar=K_PIVOT, omk=0.0, mnu=0.0, nnu=N_UR,
    )
    pars.set_matter_power(redshifts=sorted(redshifts), kmax=kmax,
                          k_per_logint=k_per_logint)
    pars.Transfer.high_precision = True
    # NB: do not raise AccuracyBoost here - boosted tolerances make CAMB's
    # Dverk integrator fail (error -3) for kmax above ~1e4/Mpc
    res = camb.get_results(pars)
    td = res.get_matter_transfer_data()
    print(f"  CAMB done in {time.time() - t0:.1f} s, "
          f"{td.transfer_data.shape[1]} k-modes")

    zs = np.array(res.transfer_redshifts)
    for z in redshifts:
        iz = int(np.argmin(np.abs(zs - z)))
        # the 13 transfer_data entries are exactly the Fortran file columns
        cols = [td.transfer_data[i, :, iz] for i in range(13)]
        k_mpc = cols[0] * LITTLE_H
        emit(root, z, k_mpc, cols)


def run_class(redshifts, kmax, k_per_decade, root, extra=None):
    from classy import Class

    print(f"CLASS: transfer functions at z = {redshifts}, kmax = {kmax:.3g}/Mpc ...")
    t0 = time.time()
    cosmo = Class()
    if extra:
        cosmo.set(extra)
    cosmo.set({
        'output': 'dTk vTk',
        'gauge': 'newtonian',
        'h': LITTLE_H,
        'Omega_b': OMEGA_B0,
        'Omega_cdm': OMEGA_DM0,
        'N_ur': N_UR,
        'A_s': A_S,
        'n_s': N_S,
        'k_pivot': K_PIVOT,
        'P_k_max_1/Mpc': kmax,
        'k_per_decade_for_pk': k_per_decade,
        'z_max_pk': max(max(redshifts) + 50.0, 100.0),
    })
    cosmo.compute()
    print(f"  CLASS done in {time.time() - t0:.1f} s")

    for z in redshifts:
        tr = cosmo.get_transfer(z, output_format='class')
        kh = tr['k (h/Mpc)']
        k = kh * LITTLE_H                       # 1/Mpc
        a = 1.0 / (1.0 + z)
        conf_H = a * cosmo.Hubble(z)            # conformal Hubble aH/c [1/Mpc]

        # CLASS (unit curvature, Newtonian gauge) -> CAMB file conventions:
        # CAMB density columns are in the CDM-comoving frame, so apply the
        # gauge shift delta -> delta + 3(aH/c)(1+w) theta_cdm/k^2, then
        #   density columns:  T_i = -delta_i / k^2       (k in 1/Mpc)
        #   velocity columns: v_i = +theta_i/(aH/c k^2)  (ddelta/dtau = -theta)
        #   v_baryon_cdm:     (theta_cdm - theta_b)/k^3  (physical v_bc/c / k^2)
        shift = 3.0 * conf_H * tr['t_cdm'] / k**2

        def dcol(name, w=0.0):
            return -(tr[name] + (1.0 + w) * shift) / k**2

        def vcol(theta):
            return theta / (conf_H * k**2)

        T_c, T_b = dcol('d_cdm'), dcol('d_b')
        T_g, T_ur = dcol('d_g', w=1.0 / 3.0), dcol('d_ur', w=1.0 / 3.0)
        T_tot = (OMEGA_B0 * T_b + OMEGA_DM0 * T_c) / (OMEGA_B0 + OMEGA_DM0)
        weyl = -(tr['phi'] + tr['psi']) / 2.0
        v_c, v_b = vcol(tr['t_cdm']), vcol(tr['t_b'])
        v_bc = (tr['t_cdm'] - tr['t_b']) / k**3
        cols = [kh, T_c, T_b, T_g, T_ur, np.zeros_like(kh),
                T_tot, T_tot, T_tot, weyl, v_c, v_b, v_bc]
        emit(root, z, k, cols)
    cosmo.struct_cleanup()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--backend', choices=['camb', 'class'], default='camb')
    ap.add_argument('--redshifts', type=float, nargs='+',
                    default=[1000.0, 900.0, 250.0, 200.0, 127.0, 0.0],
                    help='output redshifts (default includes the paper IC '
                         'epoch z=1000 and the redshifts of the old setup)')
    ap.add_argument('--kmax', type=float, default=1.5e4,
                    help='max wavenumber in 1/Mpc (default 1.5e4, so the paper '
                         'mode k=1e4/Mpc is inside the range). Empirical limits: '
                         'CAMB\'s integrator hard-fails above ~1.5e4-2e4; CLASS '
                         'runs up to ~3e4 with clean output but beyond that its '
                         'high-k modes are integration-noise dominated (tighter '
                         'tolerances or evolver=0 make it worse, not better)')
    ap.add_argument('--k-per-decade', type=int, default=30,
                    help='k sampling density per decade (default 30)')
    ap.add_argument('--output-root', default=None,
                    help='file prefix (default ic_tables/<backend>_massless)')
    ap.add_argument('--class-set', action='append', default=[],
                    metavar='KEY=VALUE',
                    help='extra CLASS precision/input setting, repeatable '
                         '(e.g. --class-set tol_perturbations_integration=1e-8; '
                         'needed to keep high-k modes noise-free for kmax '
                         'beyond ~3e4/Mpc)')
    args = ap.parse_args()

    root = args.output_root or os.path.join('ic_tables', f'{args.backend}_massless')
    if os.path.dirname(root):
        os.makedirs(os.path.dirname(root), exist_ok=True)

    if args.backend == 'camb':
        run_camb(args.redshifts, args.kmax, args.k_per_decade, root)
    else:
        extra = dict(kv.split('=', 1) for kv in args.class_set)
        run_class(args.redshifts, args.kmax, args.k_per_decade, root, extra)

    # provenance file alongside the tables
    with open(root + '_params.txt', 'w') as f:
        f.write(f"backend = {args.backend}\n"
                f"h = {LITTLE_H}\nOmega_b = {OMEGA_B0}\nOmega_cdm = {OMEGA_DM0}\n"
                f"Omega_Lambda(flat) = {1 - OMEGA_B0 - OMEGA_DM0} (+rad.)\n"
                f"A_s = {A_S}\nn_s = {N_S}\nk_pivot = {K_PIVOT} 1/Mpc\n"
                f"massless neutrinos N_ur = {N_UR}, no massive neutrinos\n"
                f"redshifts = {args.redshifts}\nkmax = {args.kmax} 1/Mpc\n"
                f"cosmology of Shalaby & Broderick 2026 (arXiv:2604.22665)\n")
    print(f"  wrote {root}_params.txt")


if __name__ == '__main__':
    main()
