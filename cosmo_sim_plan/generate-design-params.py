"""Generate Gadget-4 simulation design parameters for the DM-baryon drift run.

Implements the resolution relations of Sec. 2 of ``sim_plan.md`` (the plan for
reproducing the Shalaby & Broderick resonant drift instability).  For a target
comoving wavenumber ``k_t`` [1/Mpc] the box and particle properties are fixed by

    k_Ny = pi N / L ,   k_t = k_Ny / ratio   =>   L = pi N / (ratio * k_t)
    k_F  = 2 pi / L                          (fundamental mode)
    k_Ny = pi N / L                          (particle Nyquist)

with N = 256 particles per dimension per species and ratio = 16 (>= 32 particle
spacings per target wavelength).  The particle masses come from the mean
comoving density of each species,

    m_X = rho_crit,0 * Omega_X0 * L^3 / N^3 ,

and the gravitational softening is a fixed fraction of the mean interparticle
spacing, eps = (L / N) / soft_factor with soft_factor = 30.

The cosmology (Omega_b0, Omega_DM0, h) is taken from the repo's
``cosmo_params.py`` so it stays consistent with the IC tables and the linear
solver.

Usage:
    python generate-design-params.py 1e3                 # one target k_t
    python generate-design-params.py 1e3 1e4             # several at once
    python generate-design-params.py 1e3 --N 512         # resolution ladder
    python generate-design-params.py 1e3 --ratio 16 --soft-factor 30
"""

import argparse
import os
import sys

import numpy as np

# --- cosmology from the repo's single source of truth -----------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cosmo_params import (  # noqa: E402
    LITTLE_H, OMEGA_B0, OMEGA_DM0, H0_S, MPC_CM,
    T_CMB_K, A1_FIT, baryon_temperature, sound_speed_cms, 
)

# --- physical constants (cgs) ----------------------------------------------
G_CGS = 6.67430e-8          # gravitational constant [cm^3 g^-1 s^-2]
MSUN_G = 1.98892e33         # solar mass [g]
PC_CM = MPC_CM / 1.0e6      # 1 pc [cm]

# present-day critical density [g/cm^3]
RHO_CRIT0_CGS = 3.0 * H0_S**2 / (8.0 * np.pi * G_CGS)


def compute_design(k_t, N=256, ratio=16, soft_factor=30):
    """Return the design parameters for a target wavenumber k_t [1/Mpc].

    Parameters
    ----------
    k_t : float          target comoving wavenumber [1/Mpc]
    N : int              particles per dimension per species
    ratio : float        k_Ny / k_t (particle spacings per wavelength = 2*ratio)
    soft_factor : float  softening = mean spacing / soft_factor
    """
    # Box size [Mpc] from  k_t = k_Ny/ratio = (pi N / L)/ratio
    L = np.pi * N / (ratio * k_t)          # Mpc (comoving)
    L_cm = L * MPC_CM

    k_F = 2.0 * np.pi / L                   # 1/Mpc, fundamental mode
    k_Ny = np.pi * N / L                    # 1/Mpc, particle Nyquist

    # particle masses from the mean comoving density of each species
    Npart = N**3
    m_DM = RHO_CRIT0_CGS * OMEGA_DM0 * L_cm**3 / Npart / MSUN_G   # M_sun
    m_gas = RHO_CRIT0_CGS * OMEGA_B0 * L_cm**3 / Npart / MSUN_G   # M_sun

    spacing = L / N                         # Mpc, mean interparticle spacing
    softening = spacing / soft_factor       # Mpc (comoving)

    return {
        "k_t": k_t,
        "N": N,
        "Npart": Npart,
        "ratio": ratio,
        "soft_factor": soft_factor,
        "L_Mpc": L,
        "L_kpc": L * 1.0e3,
        "L_Mpc_h": L * LITTLE_H,            # Gadget BoxSize units (Mpc/h)
        "k_F": k_F,
        "k_Ny": k_Ny,
        "m_DM": m_DM,
        "m_gas": m_gas,
        "softening_pc": softening * 1.0e6,        # comoving pc
        "softening_Mpc_h": softening * LITTLE_H,  # Gadget Softening units (Mpc/h)
    }


def compute_thermal_start(z_start):
    """Return the start-time and gas-temperature parameters for z_start.

    Sec. 5/6 of sim_plan.md: an adiabatic (gamma=5/3, no cooling) run reproduces
    the TH2010 baryon temperature *asymptote* T_b -> T_cmb a1 / a^2 if it is
    started from the asymptote-matched value

        InitGasTemp = T_cmb * a1 * (1 + z_start)^2 ,   a1 = 1/119 ,

    rather than from the TH2010 fit value at z_start (which would leave c_s ~27%
    low for the whole run).  TimeBegin = a_start = 1/(1+z_start).
    """
    a_start = 1.0 / (1.0 + z_start)
    # init_gas_temp = T_CMB_K * A1_FIT * (1.0 + z_start)**2   # asymptote-matched
    init_gas_temp = baryon_temperature(1.0) * (1.0 + z_start)**2   # asymptote-matched
    th2010_temp = baryon_temperature(a_start)              # TH2010 fit value
    return {
        "z_start": z_start,
        "TimeBegin": a_start,
        "InitGasTemp": init_gas_temp,
        "InitGasTemp_fit": th2010_temp,
    }

def compute_streaming_speed(z_start, mach_vdm_z1000):
    """Return the DM-baryon streaming speed at z_start."""
    # init_gas_temp = baryon_temperature(1.0) * (1.0 + z_start)**2   # asymptote-matched
    # th2010_temp = baryon_temperature(1.0/(1.0+1000))              # TH2010 fit value
    # v_stream_z1000 = mach_vdm_z1000 * np.sqrt(5.0 / 3.0 * 1.380649e-16 * th2010_temp / (1.22 * 1.67262192369e-24))  # cm/s
    v_stream_z1000 = mach_vdm_z1000 * sound_speed_cms(1.0/(1.0+1000))
    a_start = 1.0 / (1.0 + z_start)
    return v_stream_z1000/a_start

def report_streaming_speed(t):
    """Pretty-print the DM-baryon streaming speed at z_start."""
    print(f"  DM-baryon streaming speed at z_start = {t['v_stream']:.3e} cm/s")

def report_thermal(t):
    """Pretty-print the start-time / gas-temperature block."""
    print(f"\n=== start redshift z_start = {t['z_start']:.0f} ===")
    print(f"  TimeBegin        = {t['TimeBegin']:.8f}   "
          f"(a_start = 1/(1+z); run to TimeMax = 1.0, z = 0)")
    print(f"  InitGasTemp      = {t['InitGasTemp']:.1f} K   "
          f"(asymptote-matched adiabatic start -- recommended)")
    print(f"    alternative    = {t['InitGasTemp_fit']:.1f} K   "
          f"(TH2010 fit value at z_start -- leaves c_s ~27% low, NOT recommended)")


def report(d):
    """Pretty-print a single design point."""
    print(f"\n=== target k_t = {d['k_t']:.4g} /Mpc "
          f"(N = {d['N']}^3/species, k_t = k_Ny/{d['ratio']:g}) ===")
    print(f"  Box size  L        = {d['L_kpc']:.4g} ckpc "
          f"= {d['L_Mpc']:.6g} Mpc = {d['L_Mpc_h']:.6g} Mpc/h  (Gadget BoxSize)")
    print(f"  Particles N        = {d['N']}^3 per species "
          f"({d['Npart']:.4g} each; gas + DM)")
    print(f"  Fundamental  k_F   = {d['k_F']:.4g} /Mpc")
    print(f"  Nyquist      k_Ny  = {d['k_Ny']:.4g} /Mpc  "
          f"(k_Ny/k_F = {d['k_Ny']/d['k_F']:.0f})")
    print(f"  DM mass   m_DM     = {d['m_DM']:.4g} Msun")
    print(f"  Gas mass  m_gas    = {d['m_gas']:.4g} Msun")
    print(f"  Softening eps      = {d['softening_pc']:.4g} cpc "
          f"= {d['softening_Mpc_h']:.4g} Mpc/h  (Gadget Softening)")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("k_t", type=float, nargs="+",
                    help="target comoving wavenumber(s) in 1/Mpc (e.g. 1e3)")
    ap.add_argument("--N", type=int, default=256,
                    help="particles per dimension per species")
    ap.add_argument("--ratio", type=float, default=16.0,
                    help="k_Ny / k_t (target sits this many factors below Nyquist)")
    ap.add_argument("--soft-factor", type=float, default=30.0,
                    help="softening = mean interparticle spacing / soft_factor")
    ap.add_argument("--z-start", type=float, default=200.0,
                    help="start redshift (fiducial 200; conservative 127; "
                         "early-resonance 900)")
    ap.add_argument("--mach-z-1000", type=float, default=5.0,
                    help="DM-baryon streaming speed at z = 1000 in units of c_s")
    args = ap.parse_args()

    print(f"Cosmology: Omega_b0 = {OMEGA_B0}, Omega_DM0 = {OMEGA_DM0}, "
          f"h = {LITTLE_H}  (from cosmo_params.py)")
    print(f"rho_crit,0 = {RHO_CRIT0_CGS:.4g} g/cm^3 "
          f"= {RHO_CRIT0_CGS * MPC_CM**3 / MSUN_G:.4g} Msun/Mpc^3")

    report_thermal(compute_thermal_start(args.z_start))
    report_streaming_speed({
        "z_start": args.z_start,
        "v_stream": compute_streaming_speed(args.z_start, args.mach_z_1000),
    })

    for k_t in args.k_t:
        report(compute_design(k_t, N=args.N, ratio=args.ratio,
                              soft_factor=args.soft_factor))


if __name__ == "__main__":
    main()
