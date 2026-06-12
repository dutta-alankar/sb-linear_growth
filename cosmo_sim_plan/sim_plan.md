# Simulating the resonant DM–baryon drift instability with Gadget-4

A plan for seeing the resonant gravitational instability of Shalaby &
Broderick (arXiv:2604.22665) — and its non-linear evolution — in a
cosmological hydro + N-body simulation with
[Gadget-4](https://wwwmpa.mpa-garching.mpg.de/gadget4/), using the linear
tools and IC tables already in this repository.

All numbers below use the paper cosmology from `cosmo_params.py`
(Ω_b0 = 0.044, Ω_DM0 = 0.226, Ω_Λ0 = 0.73, h = 0.71) and the TH2010 baryon
temperature fit; they were computed with the repo code and are reproducible
with `linear_growth.py` / `cosmo_params.py`.

---

## 1. Physics target — what we are trying to see

The instability lives at wavenumbers **below the baryon Jeans length but
above the resonance scale**: a mode k with angle θ to the drift is
resonantly driven when its projected drift is subsonic,

```
v_r(a) = v_bc(a) cosθ / c_s(a) < 1 ,      v_bc(a) = 5 c_s(a_i) (a_i/a)
```

and the resonant wavenumber is k_res = k_J / √(1−v_r²) ≥ k_J, with
k_J² = (3 Ω_b0 / 2a) H0²/c_s². Key background numbers (from
`cosmo_params.py`):

| z | c_s [km/s] | v_bc [km/s] | v_bc/c_s | μ_crit = c_s/v_bc | k_J [1/Mpc] |
|---|---|---|---|---|---|
| 1000 | 5.51 | 27.5 | 5.0 | 0.20 | 105 |
| 500 | 3.86 | 13.8 | 3.6 | 0.28 | 106 |
| 200 | 2.28 | 5.53 | 2.4 | 0.41 | 113 |
| 127 | 1.65 | 3.52 | 2.1 | 0.47 | 125 |
| 50 | 0.74 | 1.40 | 1.9 | 0.53 | 175 |
| 10 | 0.17 | 0.30 | 1.8 | 0.56 | 355 |
| 0 | 0.016 | 0.028 | 1.7 | 0.58 | 1145 |

Reading of this table: a mode is resonantly driven once **cosθ < μ_crit(z)**;
the window opens at cosθ ≈ 0.2 around recombination and widens to ≈ 0.5 by
z ~ 100 (the paper's Fig. D-vr). The linear expectation for each
(k, cosθ) is exactly what `linear_growth.py` computes (eq. D50): subsonic
projections get amplified relative to no-drift, supersonic ones suppressed,
by factors of a few in δ_b (Fig. 5).

So the simulation must (i) resolve modes with k ≫ k_J ≈ 110/Mpc, (ii) carry
a uniform decaying drift between the two fluids, (iii) start early enough
that the targeted resonant growth has not yet happened, and (iv) run to low
z where these scales go non-linear.

---

## 2. Box size and particle number

### Interpreting the resolution requirements

* "k of interest ≤ k_Ny/16": the target mode must have **≥ 32 particle
  spacings per wavelength** — safely away from the PM/SPH/shot-noise damaged
  region near Nyquist.
* "box kmin = kmax/100": the fundamental mode k_F = 2π/L and the Nyquist
  k_Ny = πN/L must satisfy k_Ny/k_F = N/2 ≈ 100 → **N ≥ 200 per dimension**,
  i.e. **N = 256³ particles per species** (giving k_Ny/k_F = 128).

With N fixed, choosing the target mode k_t fixes the box:

```
k_Ny = πN/L ,   k_t = k_Ny/16   ⇒   L = πN/(16 k_t) = 50.27/k_t  [Mpc]
```

(k_t then sits at 8 k_F, so a k-shell at k_t contains ~400 independent
modes — enough for μ-binned statistics, Sec. 9.)

### Two design points

| | fiducial | stretch (paper's mode) |
|---|---|---|
| target k_t | 10³/Mpc | 10⁴/Mpc |
| box L (comoving) | 50.3 kpc | 5.03 kpc |
| N per species | 256³ | 256³ |
| k_F | 125/Mpc | 1250/Mpc |
| k_Ny | 1.6×10⁴/Mpc | 1.6×10⁵/Mpc |
| m_DM | 0.24 M⊙ | 2.4×10⁻⁴ M⊙ |
| m_gas | 0.047 M⊙ | 4.7×10⁻⁵ M⊙ |
| softening (≈ spacing/30) | 6.5 cpc | 0.7 cpc |

**Recommend the fiducial k_t = 10³/Mpc.** Rationale:

* k_t ≈ 10 k_J at the start — comfortably inside the sub-Jeans resonant
  regime, and every box mode is sub-Jeans at z = 200 (k_F = 125 > k_J = 113).
* The paper's k = 10⁴/Mpc mode shows the same physics (the dispersion
  relation depends on k only through k/k_J and k/k_res); the fiducial box
  reaches it at k_Ny/1.6 where it is still usable for qualitative checks.
* The 5 ckpc stretch box pushes gas physics towards scales where the global
  expansion-driven approximations are less comfortable; do it only as a
  follow-up.

CDM at box scales reaches δ ~ 1 around z ≈ 10–15 (δ_c rms per ln k at k_t is
~0.065 at z = 200 and grows ∝ a), so the run naturally transitions from the
linear/resonant phase (z = 200 → ~20) to the requested **non-linear phase**
(z ≲ 10). Baryon Jeans instability at k_t switches on only near z ≈ 1.6
(k_J(z) crosses 10³/Mpc), so late-time baryon collapse at the target scale is
itself a drift-modified observable.

---

## 3. Initial conditions

* **Two-fluid ICs are mandatory**: δ_b ≠ δ_c and θ_b ≠ θ_c at these k.
  The Gadget-4 branch used here has a **patched N-GenIC**
  (`NGENIC_CREATE_BARYONS` + `NGENIC_CAMB_TRANSFERFUNCTION` in `Config.sh`,
  with the `TransferFunction_*` parameters in `param.txt`) that builds
  two-species ICs directly from Fortran-CAMB-format transfer tables — exactly
  the files this repo generates in `ic_tables/`
  (`*_z200.000000_transfer_k.dat` etc.; both a CAMB and a CLASS set exist —
  running both is a free systematics check). Use it as the primary IC route;
  MUSIC/monofonIC remain available as an independent cross-check.
  One thing to **verify in the patch**: whether it assigns species velocities
  from the velocity transfer columns (11–12 of the table) or from a
  growth-factor scale-back — see next bullet.
* Use the **velocity transfer columns** (v_Newt_cdm, v_Newt_baryon, columns
  11–12) so each fluid starts on the true two-fluid growing mode rather than
  a scale-back single growth factor; this avoids contaminating the
  oscillatory baryon modes with decaying-mode transients.
* **Quiet start**: grid (or glass) pre-IC per species, with the gas grid
  staggered by half a cell against the DM grid to avoid two-species particle
  pairing. The baryon perturbations at k_t are tiny at the start
  (P_b ≪ P_c); a lattice keeps particle noise below them until shell
  crossing — random (Poisson) pre-ICs would bury the signal
  (P_shot = (L/N)³ ≈ 7.6×10⁻¹² Mpc³ is only ~10× below P_c(k_t, z=200)).
* 2LPT for the CDM displacement is fine; the box is linear at the start
  (Δ_c(k_Ny) ≈ 0.07), so transients are mild either way.
* IC k-range: the tables cover ~10⁻⁵ → 3×10⁴/Mpc (CLASS set), comfortably
  bracketing k_F = 125/Mpc → k_Ny = 1.6×10⁴/Mpc.

---

## 4. Imposing the relative drift

### Is a uniform drift across the box a good assumption?

**Yes, excellent.** v_bc is generated by k ~ 0.01–1/Mpc perturbations and is
coherent over patches of several comoving Mpc (TH2010; the paper quotes
coherence up to ~hundreds of Mpc). The box is 50 ckpc — at least a factor
~100 smaller than the coherence scale, so the box samples a single coherent
patch with one well-defined v_bc vector. (This is the same "separate
patches" approximation used by every moving-background v_bc simulation since
TH2010.) The price is that the box cannot capture the large-scale modulation
of v_bc — handled, if desired, by an ensemble of runs with different
|v_bc| values (runs A2/A3, Sec. 7).

### How to implement it

The boost is a momentum-conserving split along ẑ:

```
Δv_DM  = + (Ω_b0/Ω_m0) v_bc(a_start) ẑ = +0.901 km/s ẑ     (PartType1)
Δv_gas = − (Ω_DM0/Ω_m0) v_bc(a_start) ẑ = −4.629 km/s ẑ     (PartType0)
```

so the box's centre-of-momentum stays at rest while the *relative* velocity
is v_bc(a_start) = 5.53 km/s at z = 200.

Because the patched N-GenIC creates ICs **on the fly** at startup
(`CREATE_GRID`, no IC file), there are two ways to inject the boost:

1. **Two-step snapshot route (no code changes, recommended first)**: the
   control run A0 writes `snapshot_000` at a_start (first entry of
   `OutputList.txt` = TimeBegin) — that file *is* the unboosted IC. Boost it
   with a ~10-line h5py script and use it as `InitCondFile` for the drift
   run, using a **second executable compiled without `NGENIC`** (if NGENIC
   is compiled in, Gadget-4 regenerates ICs instead of reading the file).
2. **Patch route**: the N-GenIC code is already patched in this branch;
   adding a `BaryonDriftVelocity` parameter that applies the split boost at
   creation time is a few lines and avoids the double build.

**Gadget velocity convention (pitfall!)**: cosmological Gadget HDF5
ICs/snapshots store u = v_pec/√a. The boost added to the `Velocities`
dataset must therefore be Δv_pec/√a_start — at z = 200 that is
u_DM = +12.78 and u_gas = −65.63 in code velocity units (km/s).

### The 1/a decay is automatic — do not force it

In comoving coordinates a force-free particle conserves canonical momentum
p = a² ẋ, so its peculiar velocity v_pec = a ẋ decays exactly as 1/a under
Gadget's cosmological integration. The imposed drift therefore follows the
paper's v_bc ∝ 1/a law by itself; no source term, no parameter, nothing to
hack. (Verify it in the output: the mean gas–DM velocity difference must
track 5.53 km/s × (a_200/a); deviations measure the momentum exchange /
collisionless drag predicted by the paper — that is signal, not error.)

### One run probes all cosθ at once

The drift picks out ẑ; every mode k in the box has its own
μ = k̂·ẑ = cosθ. A single drift run therefore contains the whole Fig. 5
family — no-drift-like behaviour at μ ≈ 0, resonant amplification at
0 < μ < μ_crit(z), supersonic suppression at μ → 1 — to be extracted as
P(k, μ) (Sec. 9). Separate runs per angle are *not* needed; what is needed
is a matched no-drift control run (same random seed) as the reference.

---

## 5. Gas thermodynamics

The instability depends on c_s(a), so the gas temperature must follow the
TH2010 evolution (CMB-coupled T ∝ 1/a until z ≈ 150, adiabatic T ∝ 1/a²
after).

* **Starting at z ≤ 200 (recommended)**: run plain non-radiative SPH
  (γ = 5/3, no cooling, no SFR). The choice of `InitGasTemp` is then the
  *only* knob controlling c_s(a) for the whole run — see the prescription
  below.

### Choosing `InitGasTemp` (important)

An adiabatic simulation evolves T ∝ a⁻² (γ = 5/3, linear regime), but the
real gas is still partially Compton-coupled to the CMB at 150 ≲ z ≲ 1000 and
cools *more slowly* than adiabatic there. The TH2010 fit tends exactly to

```
T_b(a) → T_cmb · a₁ / a²  =  0.02269 (1+z)² K        (a₁ = 1/119)
```

at late times — i.e. the real thermal history *is* adiabatic below z ≈ 100,
but with an "effective decoupling" normalisation set at a₁ = 1/119, **not**
at the simulation start. Two possible initialisations:

| z | true T_b [K] | adiabatic from **916.7 K** (asymptote-matched) | adiabatic from 460 K (fit value at z=200) |
|---|---|---|---|
| 200 | 460 | 917 (c_s 1.41×) | 460 (c_s 1.00×) |
| 100 | 166 | 232 (c_s 1.18×) | 116 (c_s 0.84×) |
| 50 | 49 | 59 (c_s 1.10×) | 30 (c_s 0.78×) |
| 20 | 9.1 | 10.0 (c_s 1.05×) | 5.0 (c_s 0.74×) |
| 10 | 2.6 | 2.7 (c_s 1.03×) | 1.4 (c_s 0.73×) |

**Recommendation: `InitGasTemp` = T_cmb·a₁·(1+z_start)² = 916.7 K for
z_start = 200** (371.7 K for z_start = 127; 1429.4 K for z_start = 250).
Matching the fit value at the start instead (460 K) looks exact initially
but leaves c_s **27% low forever** — biasing k_J, μ_crit and every resonance
property through the entire science window. The asymptote-matched value is
41% high in c_s at the very start, but the error halves by z ≈ 100 and is
≲ 5% for z ≲ 20, where the bulk of the resonant growth and all of the
non-linear evolution happen. The residual early-time mismatch can be
quantified exactly by re-running the `linear_growth.py` ODE with
`baryon_temperature()` replaced by the adiabatic law — use that ODE variant,
not the TH2010 one, as the like-for-like reference when validating the
simulation.

Two more details: Gadget-4 converts `InitGasTemp` to specific energy
assuming neutral primordial gas (μ ≈ 1.22 for T < 10⁴ K) — the same μ as the
paper's c_s, so no correction needed. And `MinEgySpec`/any temperature floor
must stay at 0: T_b(z=0) ~ 0.02 K is physical.
* **Starting earlier (z ≥ 500, needed for the early cosθ ≈ 0.2 growth)**:
  add Compton heating/cooling against the CMB,
  `du/dt = (4 σ_T a_r T_γ⁴ x_e / (m_e c (1 + x_e + f_He))) k_B (T_γ − T) / (γ−1) / μ m_p`
  with x_e(z) tabulated from CLASS/recfast — a ~20-line patch in Gadget-4's
  source (or its cooling module). Only worth it for the dedicated
  early-resonance run.
* **Artificial viscosity matters**: the resonant modes are *travelling sound
  waves*; classic constant-α viscosity damps them. Use Gadget-4's
  time-dependent (Cullen & Dehnen-type) viscosity switch with a low floor,
  and verify damping rates on a no-drift linear sound wave test in the same
  box before production.

---

## 6. Start redshift

Constraints pulling in opposite directions:

* early enough that the resonant amplification of interest has not yet
  happened, and the targeted angles are still evolving through their
  subsonic transition (μ_crit(z) table in Sec. 1);
* late enough that (i) plain adiabatic thermodynamics is honest
  (z ≲ 150–200), (ii) IC transients have room to decay before the signal
  window, and (iii) one avoids integrating thousands of acoustic
  oscillations needlessly.

**Recommendation:**

| run | z_start | what it captures |
|---|---|---|
| fiducial | **200** | the late subsonic window: cosθ ≈ 0.47 modes cross v_r = 1 around z ≈ 130 inside the run; cosθ ≈ 0.2 modes are already subsonic and keep growing resonantly; supersonic suppression at μ → 1. Uses `ic_tables/*_z200*` directly. |
| conservative variant | 127 | fully decoupled thermodynamics, cosθ ≤ 0.47 already subsonic; cross-check of fiducial. Uses `ic_tables/*_z127*`. |
| early-resonance (optional) | 900 | the z ~ 1000–300 oscillatory amplification of the cosθ ≈ 0.2 family (paper Fig. 5 black curve); requires the Compton patch of Sec. 5. Uses `ic_tables/*_z900*`. |

End at z = 0, with the science-rich stretch being z ≈ 200 → 5 (resonant +
quasi-linear) and z ≲ 10 (non-linear).

Before committing, generate the per-angle linear predictions for the actual
box modes with the repo solver, e.g.
`python linear_growth.py --k 1e3` (and k = 2, 4, 8 × 10³ …) — extend
`FIG5_CASES` with the μ values of interest — and check the expected
amplification factors between z_start and z ≈ 10 are O(2–10): that is the
window the simulation must resolve in time and statistics.

---

## 7. Run matrix

| run | drift | seed | purpose |
|---|---|---|---|
| A0 | none | S1 | control; denominator of all ratios |
| A1 | v_bc(a_s), ẑ | S1 | fiducial drift run (same seed ⇒ cosmic variance cancels mode-by-mode) |
| A2 | 2× v_bc | S1 | amplitude scaling of the resonance (≈ 2σ patch) |
| A3 | v_bc | S2, S3 | seed variance |
| B1 | v_bc | S1 | z_start = 127 variant |
| C1 | v_bc | S1 | 512³ and 128³ resolution ladder for convergence |
| D (optional) | v_bc | S1 | DM-only twin of A1 — isolates which effects need the baryon fluid |

A 2×256³, 50 ckpc box to z = 0 is a small job (order of a day on a single
modern node, dominated by the late non-linear phase); the whole matrix is
cheap.

---

## 8. Gadget-4 configuration files (`Config.sh`, `param.txt`)

Ready-to-use versions of both files for the fiducial run live next to this
plan (`cosmo_sim_plan/Config.sh`, `cosmo_sim_plan/param.txt`); they are the
old simulation's files updated to this design, with every change marked
`%% CHANGED` / `## CHANGED` inline. Summary of what changed and why:

**`Config.sh`:**

| change | reason |
|---|---|
| `PMGRID=4096 → 512` | 2× the 256³ particle grid is enough force resolution at k_t = k_Ny/16; 4096³ FFTs would be wasted on a 50 ckpc box |
| `NGENIC=4096 → 256` | must match the IC `GridSize` = N |
| `NTYPES=6 → 2` | only gas (type 0) + DM (type 1) exist in this run |
| `FOF_SECONDARY_LINK_TYPES=1+4+8+16+32 → 1` | with `NTYPES=2` only type 0 remains as FoF secondary |
| `INCLUDE_RELATIVISTIC_OMEGAS` removed | eq. D50 / `linear_growth.py` use a matter+Λ background; radiation would add ~6% to H(z=200) and spoil the direct ODE comparison |
| `BINS_PS=20000 → 2000` | ~180 P(k) bins per decade is plenty |
| `OUTPUT_PRESSURE` added | convenient for the sound-wave analysis |
| `TIMEDEP_ART_VISC` (commented) | enable your branch's time-dependent viscosity switch — constant-α viscosity damps the driven waves (Sec. 5); flag name varies between branches, verify before enabling |
| kept: `CREATE_GRID`, `NGENIC_2LPT`, `NGENIC_FIX_MODE_AMPLITUDES`, `NGENIC_CREATE_BARYONS`, `NGENIC_CAMB_TRANSFERFUNCTION`, double precision everywhere | grid pre-IC = quiet start; fixed amplitudes suppress variance (keep ON with matched seeds); the baryon/CAMB patches are the two-fluid IC machinery; the box is deeply linear early on, so double precision is cheap insurance |

**`param.txt`:**

| change | reason |
|---|---|
| `TimeBegin = 0.00497512` (z = 200), `TimeMax = 1.0` | fiducial start (Sec. 6), run to z = 0 for the non-linear phase |
| `BoxSize = 0.035688493` Mpc/h | L = πN/(16 k_t) = 50.27 ckpc for k_t = 10³/Mpc (Sec. 2) |
| `Omega0/OmegaLambda/OmegaBaryon/HubbleParam = 0.27/0.73/0.044/0.71` | paper cosmology (`cosmo_params.py`); was the old 0.306/0.694/0.0487/0.68 |
| relativistic-components block removed | paired with dropping `INCLUDE_RELATIVISTIC_OMEGAS` |
| `Softening = 4.6e-6` Mpc/h | mean interparticle spacing / 30 |
| `CourantFac = 0.3 → 0.15` | tighter time integration for travelling sound waves |
| `InitGasTemp = 916.7` K | asymptote-matched adiabatic start — see Sec. 5; **not** the fit value 460 K |
| `MinEgySpec = 0` | no temperature floor; T_b(z=0) ~ 0.02 K is physical |
| `TransferFunction_FileName_* → ../ic_tables/camb_massless_z200/..z0` | the paper-consistent tables from this repo (CLASS twin available for the systematics check) |
| `TransferFunction_As = 2.1e-9`, `SpectralIndex = 0.965` | must match the tables' primordial spectrum (`cosmo_params.py`), else the IC normalisation is inconsistent |
| `NumFilesPerSnapshot = 1`, `MaxFilesWithConcurrentIO = 1` | single-node-scale run |
| `Seed = 291100` | S1; keep **identical** between drift and control runs |

`OutputList.txt` (~60 outputs log-spaced in a, first entry = TimeBegin so
`snapshot_000` doubles as the unboosted IC for the boost route of Sec. 4):

```bash
python -c "import numpy as np
for a in np.logspace(np.log10(1/201), 0, 60): print(f'{a:.8f}')" > OutputList.txt
```

For the **drift run**: compile a second executable without `NGENIC`, point
`InitCondFile` at the boosted `snapshot_000`, and leave everything else
identical.

---

## 9. What to measure

**Linear / quasi-linear phase (z ≈ 200 → 10) — validation against eq. D50:**

1. **Anisotropic power spectra** P_X(k, μ, z), X ∈ {b, c, tot}, with
   μ = k̂·ẑ: the smoking gun is the ratio
   `R_X(k, μ, z) = P_X^drift / P_X^no-drift` (matched seeds): R_b > 1 for
   0 < μ < μ_crit(z), R_b < 1 for μ → 1, R ≈ 1 at μ ≈ 0. Overlay the same
   ratio predicted by the D50 ODE (`linear_growth.py` run per (k, μ)).
   A k_t-shell holds ~400 modes → ~40 per μ-decile; ratios beat cosmic
   variance because the seeds match.
2. **Multipoles** P_ℓ(k, z) (ℓ = 0, 2, 4) of P(k, μ): the drift generates a
   characteristic quadrupole in *baryon* power at k > k_J that no standard
   effect mimics at these scales.
3. **Per-mode tracking**: complex δ_b(k), δ_c(k) for a handful of individual
   modes vs the ODE solution — amplitude *and* phase (the iK advection terms
   make driven modes travel; measure the phase drift of δ_b(k, t) and the
   δ_b–δ_c cross-spectrum phase).
4. **Relative velocity and drag**: volume-averaged v̄_gas − v̄_DM vs a.
   Deviation from the 1/a law = the collisionless momentum-exchange drag of
   the paper; also monitor the gas bulk-flow energy budget (where the wave
   energy comes from).
5. **δ_b/δ_c ratio** vs (k, μ, z): the resonance drives baryons specifically;
   sub-Jeans baryon power approaching/exceeding the no-drift CDM-tracking
   expectation is the instability at work.
6. **Thermal check**: T̄(a) against the TH2010 fit (Sec. 5 systematic), and
   ΔT maps — the driven sound waves should show up as anisotropic
   compressive temperature fluctuations.

**Non-linear phase (z ≲ 10):**

7. **Onset of non-linearity vs μ**: redshift at which Δ²(k_t, μ) = 1, drift
   vs control — resonantly amplified directions collapse earlier.
8. **Halo statistics**: FoF/Subfind mass functions, formation redshifts, and
   **halo baryon fractions** in drift vs control (the paper predicts
   enhanced dense gas pockets; A2 tests scaling).
9. **Anisotropy of structure**: alignment of filaments/pancakes with ẑ
   (inertia-tensor or 2D power of slices) — the driven waves are
   crescent-like, nearly perpendicular to the drift.
10. **Gas-wave imagery**: δ_b slices in planes containing ẑ vs perpendicular
    to it; the resonant sound waves are visible by eye as striations with
    wavefronts at the resonant angle.

**Numerical health checks throughout**: P(k) at k > k_Ny/4 vs the resolution
ladder (C1); viscous damping of a calibration sound wave; conservation of
total momentum; shot-noise plateau location; gas–DM spurious-coupling
heating (unequal-mass two-species artefact — keep an eye on T̄ in the
control run).

---

## 10. Main risks and mitigations

| risk | mitigation |
|---|---|
| SPH artificial viscosity damping the driven waves | time-dependent AV with low floor; calibrate damping on a linear sound-wave test; consider the stretch goal with Gadget-4's MFM if available in your branch |
| baryon signal at start below particle noise | lattice/glass quiet start, staggered grids, matched-seed ratios; verify P_b(k, z_start) in the IC snapshot reproduces the input table |
| missing Compton coupling at z > 150 | quantify with the linear ODE solved with both temperature laws; or patch Compton heating for the z = 900 run |
| adiabatic vs real T at 150 < z < 200 | run B1 (z_start = 127) and compare |
| two-fluid IC transients | growing-mode velocities from the tables' velocity transfer columns; discard the first e-fold (z = 200 → 160) from analysis |
| uniform-v_bc patch approximation | by construction good (box ≪ coherence length); explore patch ensemble via A2/A3 |
| box too small for late non-linear scales (k_F = 125/Mpc means no power above 50 ckpc) | accepted by design — this is a controlled-physics experiment, not a cosmological volume; say so when presenting results |

---

## 11. Suggested order of work

1. Linear predictions for the box's (k, μ) grid with `linear_growth.py`
   (extend `FIG5_CASES` / add a `--cos-theta` option); pick z_start windows.
2. Build Gadget-4 with the provided `Config.sh` (one binary with `NGENIC`,
   one without); generate z = 200 two-fluid ICs with the patched N-GenIC and
   the provided `param.txt` (tables from
   `ic_tables/camb_massless_z200.000000_transfer_k.dat`, CLASS twin as the
   systematics check); verify the IC-stage P_b, P_c against the tables; write
   + test the drift-boost h5py script on `snapshot_000` (√a convention!).
3. 128³ pathfinder of A0/A1 to z = 10; verify the 1/a drift decay, the
   no-drift linear growth against the ODE, and the first R_b(k, μ) signal.
4. Production 256³ matrix (Sec. 7); analysis pipeline of Sec. 9.
5. Optional: z = 900 Compton-patched early-resonance run; 5 ckpc stretch box
   at k_t = 10⁴/Mpc to compare directly with the paper's Fig. 5 mode.
