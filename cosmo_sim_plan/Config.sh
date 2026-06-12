
# Gadget-4 Config.sh for the resonant DM-baryon drift instability run
# (fiducial design of sim_plan.md: 2 x 256^3 particles, 50 ckpc box).
# Changes relative to the old setup are marked with  ## CHANGED / ## NEW.

# Basic code operation

    PERIODIC
    SELFGRAVITY
    RANDOMIZE_DOMAINCENTER
    ALLOC_TOLERANCE=0.1
    NUMBER_OF_MPI_LISTENERS_PER_NODE=2

# Gravity options

    MULTIPOLE_ORDER=2
    PMGRID=512                  ## CHANGED (was 4096): 2 x N = 2 x 256 - enough force
                                ## resolution at k_t = k_Ny/16 without wasting FFTs
    DOUBLEPRECISION_FFTW
    ASMTH=1.5
    TREEPM_NOTIMESPLIT

# Softening types and particle types

    NSOFTCLASSES=1
    NTYPES=2                    ## CHANGED (was 6): only gas (type 0) + DM (type 1);
                                ## smaller memory and cleaner bookkeeping

# Floating point accuracy and storage

    POSITIONS_IN_64BIT
    IDS_64BIT
    DOUBLEPRECISION=1
    OUTPUT_IN_DOUBLEPRECISION

# Group finding (only meaningful at z <~ 10, harmless before)

    FOF
    FOF_PRIMARY_LINK_TYPES=2
    FOF_SECONDARY_LINK_TYPES=1  ## CHANGED (was 1+4+8+16+32): with NTYPES=2 only
                                ## type 0 remains as secondary
    SUBFIND
#    SUBFIND_HBT
#    MERGERTREE

# Hydro
## NEW: the science signal is travelling sound waves - constant-alpha viscosity
## damps them. Enable your branch's time-dependent (Cullen & Dehnen-type)
## viscosity switch; the flag is TIMEDEP_ART_VISC in stock Gadget-4 derivatives
## (check the exact name in your version, and add ViscosityAlphaMin~0.025 to
## param.txt if it requires it). Calibrate on a linear sound-wave test first.
#    TIMEDEP_ART_VISC
#    PRESSURE_ENTROPY_SPH       ## optional; default density-entropy SPH is fine
    OUTPUT_PRESSURE             ## NEW: convenient for the wave analysis

# Miscellaneous code options

    POWERSPEC_ON_OUTPUT         # per-type P(k) at every output - primary diagnostic
    BINS_PS=2000                ## CHANGED (was 20000): ~180 bins/decade is plenty

# IC generation via N-GenIC (patched: two-species ICs from CAMB transfer tables)

    NGENIC=256                  ## CHANGED (was 4096): = GridSize = N
    NGENIC_2LPT
    CREATE_GRID                 # lattice pre-IC = quiet start (essential: baryon
                                # signal at k_t is far below Poisson shot noise)
    NGENIC_FIX_MODE_AMPLITUDES  # variance suppression; keep ON and use matched
                                # seeds between drift and control runs
#    BINS_PS=20000              ## (old value kept here for reference)

#    INCLUDE_RELATIVISTIC_OMEGAS  ## CHANGED: disabled - eq. D50 and linear_growth.py
                                  ## use a matter+Lambda background only; keeping
                                  ## radiation (~6% of H at z=200) would spoil the
                                  ## direct ODE comparison. (If re-enabled, restore
                                  ## CMBTemperature/Neutrino* params in param.txt.)
    NGENIC_CREATE_BARYONS
    NGENIC_CAMB_TRANSFERFUNCTION
