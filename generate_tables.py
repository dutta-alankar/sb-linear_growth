"""Generate tabulated transfer functions and power spectra with CLASS and CAMB.

The tables are evaluated at z = Z_INIT (= 1000) and provide, on a grid of
wavenumbers, everything needed to set the initial conditions of the coupled
baryon-DM system (Eq. D50 of Shalaby & Broderick 2026, arXiv:2604.22665):

    delta_b, delta_cdm           (density transfer functions)
    theta_b, theta_cdm  [1/Mpc]  (Newtonian-gauge velocity divergences,
                                  conformal-time convention: ddelta/dtau = -theta)

All transfer functions are normalised to unit primordial curvature
perturbation R = 1 and follow the CLASS sign convention (CAMB output is
converted: delta -> -T_delta k^2,  theta -> +(aH/c) T_vel k^2, k in 1/Mpc).
Power spectra P_x(k) = (2 pi^2 / k^3) P_R(k) |T_x(k)|^2 are tabulated in Mpc^3.

Usage:
    python generate_tables.py [--backend class|camb|both] [--kmax 1.2e4]
    python generate_tables.py [--backend class|camb|both] --temperature-only   # only the T_b(z) tables
"""

import argparse
import os
import time

import numpy as np

from cosmo_params import (
    A_S, N_S, K_PIVOT, LITTLE_H, OMEGA_B0, OMEGA_DM0, H0_KMS_MPC, Z_INIT,
    primordial_curvature_rms,
)

C_KMS = 299792.458
TABLE_DIR = "tables"


def power_from_transfer(k, T):
    """P(k) [Mpc^3] from a transfer function normalised to unit curvature."""
    return (2.0 * np.pi**2 / k**3) * primordial_curvature_rms(k) ** 2 * T**2


def write_table(fname, header, cols):
    os.makedirs(TABLE_DIR, exist_ok=True)
    path = os.path.join(TABLE_DIR, fname)
    np.savetxt(path, np.column_stack(cols), header=header)
    print(f"  wrote {path}  ({len(cols[0])} rows)")


def run_class(kmax, k_per_decade=100):
    from classy import Class

    print(f"CLASS: computing transfer functions up to k = {kmax:.3g}/Mpc "
          f"at z = {Z_INIT:g} ({k_per_decade} k/decade) ...")
    t0 = time.time()
    cosmo = Class()
    cosmo.set({
        'output': 'mPk dTk vTk',
        'gauge': 'newtonian',
        'h': LITTLE_H,
        'Omega_b': OMEGA_B0,
        'Omega_cdm': OMEGA_DM0,
        'A_s': A_S,
        'n_s': N_S,
        'k_pivot': K_PIVOT,
        'P_k_max_1/Mpc': kmax,
        'k_per_decade_for_pk': k_per_decade,
        'z_pk': f'{Z_INIT:g}',
        'z_max_pk': Z_INIT + 50.0,
        'matter_source_in_current_gauge': 'yes',
    })
    cosmo.compute()
    tr = cosmo.get_transfer(Z_INIT, output_format='class')
    k = tr['k (h/Mpc)'] * LITTLE_H            # 1/Mpc
    d_b, d_c = tr['d_b'], tr['d_cdm']
    t_b, t_c = tr['t_b'], tr['t_cdm']         # theta [1/Mpc], conformal time
    print(f"  CLASS done in {time.time() - t0:.1f} s, {len(k)} k-modes")

    header = (f"CLASS v3 transfer functions at z = {Z_INIT:g}, Newtonian gauge,\n"
              f"normalised to unit primordial curvature (output_format='class').\n"
              f"theta convention: d(delta)/dtau = -theta, theta in 1/Mpc (c=1).\n"
              f"cosmology: h={LITTLE_H}, Omega_b={OMEGA_B0}, Omega_cdm={OMEGA_DM0},"
              f" A_s={A_S}, n_s={N_S}, k_pivot={K_PIVOT}/Mpc\n"
              f"k[1/Mpc]  delta_b  delta_cdm  theta_b[1/Mpc]  theta_cdm[1/Mpc]")
    write_table(f"class_transfer_z{Z_INIT:g}.txt", header, [k, d_b, d_c, t_b, t_c])

    P_b = power_from_transfer(k, d_b)
    P_c = power_from_transfer(k, d_c)
    P_m = power_from_transfer(k, tr['d_m'])
    # direct CLASS P_m(k) as a cross-check column
    P_m_direct = np.array([cosmo.pk(kk, Z_INIT) for kk in k])
    header = (f"Power spectra at z = {Z_INIT:g} [Mpc^3] from CLASS transfer functions;\n"
              f"last column is the direct CLASS P_m(k) for cross-checking.\n"
              f"k[1/Mpc]  P_b  P_cdm  P_m  P_m_direct")
    write_table(f"class_power_z{Z_INIT:g}.txt", header, [k, P_b, P_c, P_m, P_m_direct])
    cosmo.struct_cleanup()


def run_camb(kmax, k_per_decade=100):
    import camb
    from camb import model

    # CAMB samples per natural-log interval; convert from per-decade.
    k_per_logint = int(round(k_per_decade / np.log(10.0)))
    print(f"CAMB: computing transfer functions up to k = {kmax:.3g}/Mpc "
          f"at z = {Z_INIT:g} ({k_per_decade} k/decade -> k_per_logint={k_per_logint}) ...")
    t0 = time.time()
    pars = camb.set_params(
        H0=H0_KMS_MPC, ombh2=OMEGA_B0 * LITTLE_H**2, omch2=OMEGA_DM0 * LITTLE_H**2,
        As=A_S, ns=N_S, pivot_scalar=K_PIVOT, omk=0.0,
    )
    pars.set_matter_power(redshifts=[Z_INIT], kmax=kmax, k_per_logint=k_per_logint,
                          accurate_massive_neutrino_transfers=False)
    pars.Transfer.high_precision = True
    res = camb.get_results(pars)
    td = res.get_matter_transfer_data()
    iz = int(np.argmin(np.abs(np.array(res.transfer_redshifts) - Z_INIT)))
    k = td.transfer_data[model.Transfer_kh - 1, :, iz] * LITTLE_H   # 1/Mpc

    # Convert CAMB conventions to the CLASS ones used in the tables:
    #   CAMB matter transfer columns are delta/k^2 (k in 1/Mpc), opposite
    #   overall sign to CLASS; Newtonian velocity columns satisfy
    #   theta = -(aH/c) T_vel k^2 (verified against d(delta)/dtau = -theta).
    Hz = res.hubble_parameter(Z_INIT)                # km/s/Mpc
    aH_c = Hz / (1.0 + Z_INIT) / C_KMS               # conformal Hubble, 1/Mpc
    d_b = -td.transfer_data[model.Transfer_b - 1, :, iz] * k**2
    d_c = -td.transfer_data[model.Transfer_cdm - 1, :, iz] * k**2
    t_b = aH_c * td.transfer_data[model.Transfer_Newt_vel_baryon - 1, :, iz] * k**2
    t_c = aH_c * td.transfer_data[model.Transfer_Newt_vel_cdm - 1, :, iz] * k**2
    d_m = -td.transfer_data[model.Transfer_nonu - 1, :, iz] * k**2
    print(f"  CAMB done in {time.time() - t0:.1f} s, {len(k)} k-modes")

    header = (f"CAMB transfer functions at z = {Z_INIT:g}, converted to the CLASS\n"
              f"convention (Newtonian gauge, unit primordial curvature, theta in 1/Mpc\n"
              f"with d(delta)/dtau = -theta).\n"
              f"cosmology: h={LITTLE_H}, Omega_b={OMEGA_B0}, Omega_cdm={OMEGA_DM0},"
              f" A_s={A_S}, n_s={N_S}, k_pivot={K_PIVOT}/Mpc\n"
              f"k[1/Mpc]  delta_b  delta_cdm  theta_b[1/Mpc]  theta_cdm[1/Mpc]")
    write_table(f"camb_transfer_z{Z_INIT:g}.txt", header, [k, d_b, d_c, t_b, t_c])

    P_b = power_from_transfer(k, d_b)
    P_c = power_from_transfer(k, d_c)
    P_m = power_from_transfer(k, d_m)
    header = (f"Power spectra at z = {Z_INIT:g} [Mpc^3] from CAMB transfer functions.\n"
              f"k[1/Mpc]  P_b  P_cdm  P_m")
    write_table(f"camb_power_z{Z_INIT:g}.txt", header, [k, P_b, P_c, P_m])


def run_class_temperature(zmax=1500.0, n=4000):
    """Mean baryon temperature T_b(z) and free-electron fraction x_e(z) from CLASS's thermal history, written to
    tables/class_baryon_temperature.txt. Same cosmology as the N-GenIC IC tables (ic_tables/class_massless_params.txt,
    N_ur = 3.046). linear_growth.py --temperature class uses it for the baryon sound speed, and Gadget-4
    (CLASS_ENTROPY_FLOOR, parameter CMBHeatingTemperatureTable) for its entropy floor. NB: CLASS couples the gas
    to the CMB again after reionisation (z_reio ~ 7.7), so T_b rises at z < 9 (16x the TH2010 fit at z = 0)."""
    from classy import Class

    cosmo = Class()
    cosmo.set({'h': LITTLE_H, 'Omega_b': OMEGA_B0, 'Omega_cdm': OMEGA_DM0, 'N_ur': 3.046,
               'A_s': A_S, 'n_s': N_S, 'k_pivot': K_PIVOT})
    cosmo.compute()
    z = np.logspace(np.log10(1.0 + zmax), 0.0, n) - 1.0
    z[-1] = 0.0
    tb = np.array([cosmo.baryon_temperature(zz) for zz in z])
    xe = np.array([cosmo.ionization_fraction(zz) for zz in z])
    der = cosmo.get_current_derived_parameters(['YHe', 'z_reio'])
    print(f"CLASS (T_CMB = {cosmo.T_cmb()} K, Y_He = {der['YHe']:.5f}, z_reio = {der['z_reio']:.3f}): "
          f"T_b(z={z[0]:g}) = {tb[0]:.5g} K ... T_b(z=0) = {tb[-1]:.5g} K")
    header = (f"Mean baryon temperature from CLASS (thermal history incl. reionisation, z_reio = {der['z_reio']:.4f})\n"
              f"cosmology: h={LITTLE_H}, Omega_b={OMEGA_B0}, Omega_cdm={OMEGA_DM0}, N_ur=3.046, "
              f"T_CMB={cosmo.T_cmb()} K, Y_He={der['YHe']:.5f}\n"
              f"z  T_b[K]  x_e")
    write_table("class_baryon_temperature.txt", header, [z, tb, xe])
    cosmo.struct_cleanup()


def run_camb_temperature(zmax=1500.0, n=4000):
    """Mean baryon temperature T_b(z) and free-electron fraction x_e(z) from CAMB's thermal history (same cosmology
    as run_camb), written to tables/camb_baryon_temperature.txt (used by linear_growth.py --temperature camb). Unlike
    CLASS, CAMB's T_b has no reionisation feature: it keeps cooling adiabatically below z ~ 9."""
    import camb

    pars = camb.set_params(
        H0=H0_KMS_MPC, ombh2=OMEGA_B0 * LITTLE_H**2, omch2=OMEGA_DM0 * LITTLE_H**2,
        As=A_S, ns=N_S, pivot_scalar=K_PIVOT, omk=0.0,
    )
    res = camb.get_background(pars)
    z = np.logspace(np.log10(1.0 + zmax), 0.0, n) - 1.0
    z[-1] = 0.0
    ev = res.get_background_redshift_evolution(z, ['T_b', 'x_e'], format='array')
    recomb = type(pars.Recomb).__name__
    print(f"CAMB {camb.__version__} ({recomb}, T_CMB = {pars.TCMB} K, Y_He = {pars.YHe:.5f}): "
          f"T_b(z={z[0]:g}) = {ev[0, 0]:.5g} K ... T_b(z=0) = {ev[-1, 0]:.5g} K")
    header = (f"Mean baryon temperature from CAMB {camb.__version__} ({recomb} thermal history, T_b without "
              f"reionisation feature)\n"
              f"cosmology: h={LITTLE_H}, Omega_b={OMEGA_B0}, Omega_cdm={OMEGA_DM0}, T_CMB={pars.TCMB} K, "
              f"Y_He={pars.YHe:.5f}\n"
              f"z  T_b[K]  x_e")
    write_table("camb_baryon_temperature.txt", header, [z, ev[:, 0], ev[:, 1]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--backend', choices=['class', 'camb', 'both'], default='both')
    ap.add_argument('--temperature-only', action='store_true',
                    help='only write the mean baryon temperature tables tables/{class,camb}_baryon_temperature.txt '
                         '(T_b(z), x_e(z) of the chosen backend(s)), no transfer functions')
    ap.add_argument('--kmax', type=float, default=1.2e4,
                    help='max wavenumber in 1/Mpc (default 1.2e4, must exceed the '
                         'k used in linear_growth.py)')
    ap.add_argument('--k-per-decade', type=int, default=100,
                    help='CLASS k sampling density k_per_decade_for_pk (default 100)')
    args = ap.parse_args()

    if args.backend in ('class', 'both'):
        if not args.temperature_only:
            run_class(args.kmax, args.k_per_decade)
        run_class_temperature()
    if args.backend in ('camb', 'both'):
        if not args.temperature_only:
            run_camb(args.kmax, args.k_per_decade)
        run_camb_temperature()


if __name__ == '__main__':
    main()
