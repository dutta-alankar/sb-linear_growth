"""Compare the adiabatic gas-temperature evolution of a non-radiative
simulation with the Tseliakhovich & Hirata (2010) baryon temperature, which
includes the residual (partial) Compton coupling to the CMB.

A Gadget-4 run without radiative physics evolves T proportional to 1/a^2 (for
gamma = 5/3 in the linear regime); the real intergalactic gas cools more
slowly at 150 <~ z <~ 1000 because Compton scattering off CMB photons keeps
it partially coupled (T -> T_cmb/a when fully coupled). The TH2010 fit used
by the paper (and by cosmo_params.baryon_temperature) interpolates between
the two regimes and tends to T -> T_cmb*a1/a^2 (a1 = 1/119) at late times.

This script quantifies the choice of InitGasTemp discussed in sim_plan.md
Sec. 5 by evolving the two candidate initialisations adiabatically from
z_start and comparing temperature and sound speed against the TH2010 fit:

  fit-matched:        T_i = T_TH(z_start)            (exact at start,
                                                      c_s low forever after)
  asymptote-matched:  T_i = T_cmb*a1*(1+z_start)^2   (hot at start, converges
                                                      to the truth at z <~ 20)

Usage:
    python compare_temperature.py [--z-start 200] [--out temperature_comparison.png]
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from cosmo_params import baryon_temperature, T_CMB_K, A1_FIT


def adiabatic(z, z_start, T_init):
    """T(z) for gamma=5/3 adiabatic expansion from T_init at z_start."""
    return T_init * ((1.0 + z) / (1.0 + z_start)) ** 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--z-start', type=float, default=200.0)
    ap.add_argument('--out', default=None, help='output figure (default '
                    'temperature_comparison.png next to this script)')
    args = ap.parse_args()
    zs = args.z_start

    T_fit_init = baryon_temperature(1.0 / (1.0 + zs))
    # T_asym_init = T_CMB_K * A1_FIT * (1.0 + zs) ** 2
    T_asym_init = baryon_temperature(1.0) * (1.0 + zs) ** 2
    print(f"z_start = {zs:g}:")
    print(f"  fit-matched       InitGasTemp = {T_fit_init:8.1f} K")
    print(f"  asymptote-matched InitGasTemp = {T_asym_init:8.1f} K   <- recommended")

    z = np.logspace(np.log10(zs), np.log10(0.01), 400)
    T_th = baryon_temperature(1.0 / (1.0 + z))          # TH2010 (partial Compton)
    T_cmb = T_CMB_K * (1.0 + z)                          # fully coupled
    T_ad_fit = adiabatic(z, zs, T_fit_init)
    T_ad_asym = adiabatic(z, zs, T_asym_init)

    print(f"\n  {'z':>6s} {'T_TH2010':>10s} {'T_ad(asym)':>11s} {'cs err':>7s} "
          f"{'T_ad(fit)':>10s} {'cs err':>7s}")
    for zp in [zs, 150, 100, 50, 20, 10, 5, 1]:
        if zp > zs:
            continue
        Tt = baryon_temperature(1.0 / (1.0 + zp))
        Ta = adiabatic(zp, zs, T_asym_init)
        Tf = adiabatic(zp, zs, T_fit_init)
        print(f"  {zp:6.0f} {Tt:10.2f} {Ta:11.2f} {np.sqrt(Ta/Tt)-1:+7.1%} "
              f"{Tf:10.2f} {np.sqrt(Tf/Tt)-1:+7.1%}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.0, 6.5), sharex=True,
                                   height_ratios=[2, 1])
    ax1.loglog(1 + z, T_th, 'k-', lw=2,
               label='TH2010 fit (partial Compton coupling)')
    ax1.loglog(1 + z, T_cmb, 'k:', lw=1, alpha=0.6,
               label=r'fully coupled, $T_{\rm CMB}(1+z)$')
    ax1.loglog(1 + z, T_ad_asym, 'C0-', lw=1.5,
               label=f'adiabatic from {T_asym_init:.0f} K (asymptote-matched)')
    ax1.loglog(1 + z, T_ad_fit, 'C3--', lw=1.5,
               label=f'adiabatic from {T_fit_init:.0f} K (fit-matched)')
    ax1.set_ylabel(r'$T_b$ [K]')
    ax1.legend(fontsize=8, frameon=False)
    ax1.grid(True, which='both', ls=':', lw=0.5, alpha=0.7)
    ax1.set_title(f'Gas temperature: adiabatic simulation vs TH2010, '
                  f'$z_{{\\rm start}} = {zs:g}$', fontsize=10)

    ax2.semilogx(1 + z, np.sqrt(T_ad_asym / T_th), 'C0-', lw=1.5)
    ax2.semilogx(1 + z, np.sqrt(T_ad_fit / T_th), 'C3--', lw=1.5)
    ax2.axhline(1.0, color='k', lw=0.8)
    ax2.set_ylabel(r'$c_s^{\rm adiab}/c_s^{\rm TH2010}$')
    ax2.set_xlabel(r'$z + 1$')
    ax2.set_ylim(0.6, 1.5)
    ax2.grid(True, which='both', ls=':', lw=0.5, alpha=0.7)
    ax2.invert_xaxis()

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'temperature_comparison.png')
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print(f"\nfigure saved to {out}")


if __name__ == '__main__':
    main()
