"""Shared cosmology / numerical parameters for the Shalaby & Broderick (2026),
arXiv:2604.22665, Figure 5 reproduction (equation system D50, their Appendix D).

The background cosmology is the one quoted below their Eq. (D50):
Omega_b0 = 0.044, Omega_DM0 = 0.226, Omega_Lambda0 = 0.73, H0 = 71 km/s/Mpc.
"""

import os

import numpy as np

# --- background cosmology (paper, Appendix D) ---
H0_KMS_MPC = 71.0
LITTLE_H = H0_KMS_MPC / 100.0
OMEGA_B0 = 0.044
OMEGA_DM0 = 0.226
OMEGA_L0 = 0.73

# --- primordial spectrum (not specified in the paper; standard values) ---
A_S = 2.1e-9
N_S = 0.965
K_PIVOT = 0.05  # 1/Mpc

# --- epoch where the integration starts (v_DM = 5 c_s there, TH2010) ---
Z_INIT = 1000.0
A_INIT = 1.0 / (1.0 + Z_INIT)
VDM_OVER_CS_INIT = 5.0

# --- physical constants (cgs) ---
MPC_CM = 3.0856775814913673e24
C_CMS = 2.99792458e10
KB_CGS = 1.380649e-16
MPROTON_G = 1.67262192369e-24

H0_S = H0_KMS_MPC * 1.0e5 / MPC_CM  # H0 in 1/s

# --- baryon temperature fit (TH2010 / paper Appendix D) ---
T_CMB_K = 2.7
A1_FIT = 1.0 / 119.0
A2_FIT = 1.0 / 115.0
MU_MOL = 1.22
GAMMA_AD = 5.0 / 3.0


def hubble_norm(a):
    """h(a) = H(a)/H0 as adopted in the paper (matter + Lambda only)."""
    return np.sqrt((OMEGA_B0 + OMEGA_DM0) / a**3 + OMEGA_L0)


def baryon_temperature(a):
    """T_b(a) in K, fit from Tseliakhovich & Hirata (2010), as in the paper."""
    return (T_CMB_K / a) / (1.0 + (a / A1_FIT) / (1.0 + (A2_FIT / a) ** 1.5))


def adiabatic_baryon_temperature(z, z_start, T_init):
    """T(z) for gamma=5/3 adiabatic expansion from T_init at z_start."""
    return T_init * ((1.0 + z) / (1.0 + z_start)) ** 2


def asymptote_baryon_temperature(z, zstart):
    """T(z) for the TH2010 asymptote T_b -> T_cmb a1 / a^2."""
    # T_asym_init = T_CMB_K * A1_FIT * (1.0 + zstart) ** 2
    T_asym_init = baryon_temperature(1.0) * (1.0 + zstart) ** 2 
    return adiabatic_baryon_temperature(z, zstart, T_asym_init)

def fit_baryon_temperature(z, zstart):
    """T(z) for the TH2010 fit T_b(zstart) at z_start, then adiabatic expansion."""
    T_fit_init = baryon_temperature(1.0/(1.0 + zstart)) 
    return adiabatic_baryon_temperature(z, zstart, T_fit_init)


def sound_speed_cms(a):
    """Baryon sound speed c_s(a) in cm/s."""
    return np.sqrt(GAMMA_AD * KB_CGS * baryon_temperature(a) / (MU_MOL * MPROTON_G))


def sound_speed_adiabatic_asymptote_cms(a, zstart):
    """Baryon sound speed c_s(a) in cm/s."""
    z = 1.0 / a - 1.0
    return np.sqrt(GAMMA_AD * KB_CGS * asymptote_baryon_temperature(z, zstart) / (MU_MOL * MPROTON_G))

def sound_speed_adiabatic_fit_cms(a, zstart):
    """Baryon sound speed c_s(a) in cm/s."""
    z = 1.0 / a - 1.0
    return np.sqrt(GAMMA_AD * KB_CGS * fit_baryon_temperature(z, zstart) / (MU_MOL * MPROTON_G))

# --- optional: mean baryon temperature from CLASS or CAMB instead of the TH2010 fit ---
# tables/{class,camb}_baryon_temperature.txt (z, T_b [K], x_e) are written by
# 'python generate_tables.py --temperature-only'; Gadget-4 reads such a table with CLASS_ENTROPY_FLOOR
# (parameter CMBHeatingTemperatureTable). NB: CLASS's T_b rises again after reionisation (z < 9), CAMB's does not.
TB_TABLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
_tb_splines = {}


def tb_table_path(code):
    """path of the T_b(z) table of code = 'class' or 'camb'"""
    return os.path.join(TB_TABLE_DIR, f"{code}_baryon_temperature.txt")


def table_baryon_temperature(a, code="class"):
    """T_b(a) in K from the thermal history of code = 'class' or 'camb': cubic spline of ln T_b in ln a (as in
    Gadget-4)."""
    if code not in _tb_splines:
        from scipy.interpolate import CubicSpline
        path = tb_table_path(code)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found - run 'python generate_tables.py --backend {code} "
                                    f"--temperature-only'")
        z, tb = np.loadtxt(path, usecols=(0, 1), unpack=True)
        o = np.argsort(z)[::-1]                                    # increasing a
        _tb_splines[code] = CubicSpline(-np.log1p(z[o]), np.log(tb[o]), bc_type="natural")
    spl = _tb_splines[code]
    lna = np.log(a)
    if np.any(lna < spl.x[0] - 1e-10) or np.any(lna > spl.x[-1] + 1e-10):
        raise ValueError(f"a outside the range of {tb_table_path(code)}")
    return np.exp(spl(lna))


def sound_speed_table_cms(a, code="class"):
    """Baryon sound speed c_s(a) in cm/s, c_s^2 = gamma k_B T_b / (mu m_p) with T_b(a) from CLASS or CAMB."""
    return np.sqrt(GAMMA_AD * KB_CGS * table_baryon_temperature(a, code) / (MU_MOL * MPROTON_G))


def v_dm_cms(a):
    """DM-baryon relative drift speed, v_DM = 5 c_s(a_i) * (a_i/a), in cm/s."""
    return VDM_OVER_CS_INIT * sound_speed_cms(A_INIT) * (A_INIT / a)


def primordial_curvature_rms(k):
    """sqrt(P_R(k)) per ln k: dimensionless curvature amplitude at wavenumber k [1/Mpc]."""
    return np.sqrt(A_S * (k / K_PIVOT) ** (N_S - 1.0))

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    zp1 = np.logspace(0, np.log10( 1.0 + Z_INIT), 400)
    a_vals = 1.0 / zp1
    cs = sound_speed_cms(a_vals)
    Tb = baryon_temperature(a_vals)
    vrel = v_dm_cms(a_vals)

    plt.figure(figsize=(8, 6))
    plt.semilogx(zp1, vrel/cs, label=r'$v_{\rm DM}/c_s$', color='k')
    plt.xlim(1.0 + Z_INIT, 1.0)
    plt.xlabel('1+z')
    plt.ylabel('Mach number')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.title('Mach number evolution')
    plt.show()
