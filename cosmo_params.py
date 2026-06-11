"""Shared cosmology / numerical parameters for the Shalaby & Broderick (2026),
arXiv:2604.22665, Figure 5 reproduction (equation system D50, their Appendix D).

The background cosmology is the one quoted below their Eq. (D50):
Omega_b0 = 0.044, Omega_DM0 = 0.226, Omega_Lambda0 = 0.73, H0 = 71 km/s/Mpc.
"""

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


def sound_speed_cms(a):
    """Baryon sound speed c_s(a) in cm/s."""
    return np.sqrt(GAMMA_AD * KB_CGS * baryon_temperature(a) / (MU_MOL * MPROTON_G))


def v_dm_cms(a):
    """DM-baryon relative drift speed, v_DM = 5 c_s(a_i) * (a_i/a), in cm/s."""
    return VDM_OVER_CS_INIT * sound_speed_cms(A_INIT) * (A_INIT / a)


def primordial_curvature_rms(k):
    """sqrt(P_R(k)) per ln k: dimensionless curvature amplitude at wavenumber k [1/Mpc]."""
    return np.sqrt(A_S * (k / K_PIVOT) ** (N_S - 1.0))
