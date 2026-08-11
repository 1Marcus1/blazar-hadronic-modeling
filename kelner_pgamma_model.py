"""
Photo-hadronic (p-gamma) secondary particle production model.

Implements the analytical parametrization of Kelner & Aharonian (2008), PRD 78,
034013, https://arxiv.org/abs/0803.0688, for the spectra of gamma-rays,
electrons/positrons, and neutrinos produced by photopion production in
proton-photon (p-gamma) interactions.

The parametrization gives the dimensionless spectrum F(eta, x) of each secondary
particle, where:

- eta = 4 * epsilon * E_p / m_p^2 c^4 is a dimensionless parameter combining the
  target photon energy `epsilon` and the proton energy `E_p`,
- x = E_secondary / E_p is the fraction of the proton energy carried by the
  secondary particle,

fit to tabulated coefficients (s, delta, B) as a function of eta / eta0 for each
particle species, digitized in the `*param.txt` files distributed alongside
this module.
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Physical constants (particle masses in GeV, matching the tabulated fits)
# ---------------------------------------------------------------------------
PROTON_MASS_GEV = 0.9382720813
NEUTRAL_PION_MASS_GEV = 0.1349770
CHARGED_PION_MASS_GEV = 0.13957061

# Guard value used where a kinematic threshold would otherwise divide by ~0.
_SMALL_D = 1.0e-200

# Fraction of neutrino flavor produced by muon decay assumed to survive
# oscillation as each flavor (nu_mu <-> nu_e <-> nu_tau, equal mixing).
NEUTRINO_OSCILLATION_FACTOR = 1.0 / 3.0

# Normalization coefficient from Kelner & Aharonian (2008); the "* 1.3" factor
# is an empirical correction fixing the CMB-normalization case (see original
# reference implementation notes).
KELNER_KOEFFICIENT = 1.1634328 * 1.3


class ParticleType(str, Enum):
    """Secondary particle species produced in p-gamma interactions."""

    GAMMA = "gamma"
    POSITRON = "positron"
    ELECTRON = "electron"
    NU_MU = "nu-mu"
    ANTI_NU_MU = "anti-nu-mu"
    NU_E = "nu-e"
    ANTI_NU_E = "anti-nu-e"


# Default parameter-table filenames, as distributed with this module.
DEFAULT_PARAM_FILENAMES = {
    ParticleType.GAMMA: "gammaparam.txt",
    ParticleType.POSITRON: "positronparam.txt",
    ParticleType.ELECTRON: "electronparam.txt",
    ParticleType.NU_MU: "numuparam.txt",
    ParticleType.ANTI_NU_MU: "antinumuparam.txt",
    ParticleType.NU_E: "nueparam.txt",
    ParticleType.ANTI_NU_E: "antinueparam.txt",
}


@dataclass
class KelnerFitTable:
    """Tabulated Kelner & Aharonian (2008) fit coefficients for one particle species.

    Each row of the source file corresponds to one value of eta/eta0, with
    columns (eta/eta0, s, delta, B).

    Attributes
    ----------
    eta_over_eta0 : np.ndarray
        Tabulated values of eta/eta0.
    s, delta, B : np.ndarray
        Tabulated fit coefficients, same length as `eta_over_eta0`.
    """

    eta_over_eta0: np.ndarray
    s: np.ndarray
    delta: np.ndarray
    B: np.ndarray

    @classmethod
    def from_file(cls, filepath: str) -> "KelnerFitTable":
        """Load a fit table from a semicolon-delimited file (eta/eta0; s; delta; B)."""
        eta_over_eta0, s, delta, B = np.loadtxt(filepath, delimiter=";", unpack=True)
        return cls(eta_over_eta0=eta_over_eta0, s=s, delta=delta, B=B)


def load_fit_tables(
    directory: str = ".", filenames: dict[ParticleType, str] | None = None
) -> dict[ParticleType, KelnerFitTable]:
    """Load the Kelner & Aharonian fit tables for all seven particle species.

    Loading is done once per call; pass the resulting dict to `calculate_F`
    instead of re-reading the tables from disk on every evaluation (the
    original implementation reloaded the relevant file inside the innermost
    loop of the SED integration, which is very costly at scale).

    Parameters
    ----------
    directory : str, default "."
        Directory containing the `*param.txt` files.
    filenames : dict[ParticleType, str], optional
        Override the default filenames (see `DEFAULT_PARAM_FILENAMES`).

    Returns
    -------
    dict[ParticleType, KelnerFitTable]
        One fit table per particle species.
    """
    filenames = filenames or DEFAULT_PARAM_FILENAMES
    directory = Path(directory)
    return {
        particle: KelnerFitTable.from_file(str(directory / filename))
        for particle, filename in filenames.items()
    }


def proton_spectrum(energy_p_eV: np.ndarray, index: float, cutoff_energy_eV: float | None) -> np.ndarray:
    """Evaluate the primary proton injection spectrum dN/dE (unnormalized shape).

    A pure power law if `cutoff_energy_eV` is None (or negative, matching the
    original convention), otherwise a power law with an exponential cutoff.

    Parameters
    ----------
    energy_p_eV : np.ndarray
        Proton energy [eV].
    index : float
        Power-law spectral index (dN/dE ~ E^-index).
    cutoff_energy_eV : float or None
        Exponential cutoff energy [eV]. None (or a negative value) disables
        the cutoff, matching the original `E_cut_p < 0` convention.

    Returns
    -------
    np.ndarray
        Unnormalized spectral shape, same shape as `energy_p_eV`.
    """
    reference_energy_eV = 1.0
    shape = np.power(energy_p_eV / reference_energy_eV, -index)
    if cutoff_energy_eV is None or cutoff_energy_eV < 0:
        return shape
    return shape * np.exp(-energy_p_eV / cutoff_energy_eV)


def _interpolate_fit_coefficients(
    particle: ParticleType, eta: float, eta0: float, table: KelnerFitTable
) -> tuple[float, float, float]:
    """Interpolate (s, delta, B) at a given eta/eta0 from a `KelnerFitTable`.

    Cubic interpolation is used within the tabulated range; outside of it, the
    boundary value is held constant, except for the `B` coefficient of the
    electron and anti-nu-e tables below the tabulated range, which is instead
    linearly extrapolated (following the original reference implementation).

    Parameters
    ----------
    particle : ParticleType
        Particle species (only used to select the low-end `B` extrapolation).
    eta : float
        Raw (non-normalized) dimensionless photon-proton interaction parameter.
    eta0 : float
        Species-specific normalization eta0 = 2r + r^2.
    table : KelnerFitTable
        Tabulated fit coefficients for this particle.

    Returns
    -------
    s, delta, B : float
        Interpolated (or extrapolated) fit coefficients.

    Notes
    -----
    For the ELECTRON and ANTI_NU_E tables, the low-end extrapolation of `B`
    reproduces the original reference implementation exactly, which divides
    by `eta0` twice (`rho = (eta / eta0) / eta0`) rather than once. This looks
    unusual, but is kept as-is here rather than "corrected" silently, since
    the intended behavior is not fully clear from the original code/references
    alone — worth double-checking against Kelner & Aharonian (2008) if this
    low-eta regime matters for your application.
    """
    eta_table = table.eta_over_eta0
    eta_over_eta0 = eta / eta0

    if eta_table[0] < eta_over_eta0 < eta_table[-1]:
        s = interp1d(eta_table, table.s, kind="cubic")(eta_over_eta0)
        delta = interp1d(eta_table, table.delta, kind="cubic")(eta_over_eta0)
        B = interp1d(eta_table, table.B, kind="cubic")(eta_over_eta0)
        return float(s), float(delta), float(B)

    if eta_over_eta0 <= eta_table[0]:
        s, delta, B = table.s[0], table.delta[0], table.B[0]
        if particle in (ParticleType.ELECTRON, ParticleType.ANTI_NU_E):
            rho = eta_over_eta0 / eta0
            rho0 = eta_table[0] / eta0
            B = table.B[0] * (rho - 2.14) / (rho0 - 2.14)
        return float(s), float(delta), float(B)

    # eta_over_eta0 >= eta_table[-1]
    return float(table.s[-1]), float(table.delta[-1]), float(table.B[-1])


def _kelner_shape(x: np.ndarray, x_lower: float, x_upper: float, s: float, delta: float, B: float, p: float) -> np.ndarray:
    """Evaluate the common Kelner & Aharonian piecewise F(x) shape.

    All seven particle species share this same functional form once their
    species-specific kinematic bounds (`x_lower`, `x_upper`) and exponent `p`
    are known: a constant plateau below `x_lower`, a smoothly decaying
    log-log shape between `x_lower` and `x_upper`, and zero above `x_upper`.

    Parameters
    ----------
    x : np.ndarray
        Secondary-to-proton energy fraction(s) at which to evaluate F.
    x_lower, x_upper : float
        Species-specific kinematic bounds on x.
    s, delta, B : float
        Interpolated fit coefficients (see `_interpolate_fit_coefficients`).
    p : float
        Species- and energy-dependent shape exponent.

    Returns
    -------
    np.ndarray
        F(x), same shape as `x`.
    """
    x = np.asarray(x, dtype=float)
    F = np.zeros_like(x)

    plateau = x <= x_lower
    F[plateau] = B * np.power(np.log(2.0), p)

    rising = (x > x_lower) & (x < x_upper)
    if np.any(rising):
        y = (x[rising] - x_lower) / (x_upper - x_lower)
        term_1 = np.exp(-s * np.power(np.log(x[rising] / x_lower), delta))
        term_2 = np.power(np.log(2.0 / (1.0 + y**2)), p)
        F[rising] = B * term_1 * term_2

    # x >= x_upper stays at the initialized 0.0
    return F if F.ndim else float(F)


def calculate_F(
    particle: ParticleType, eta: float, x: float, fit_tables: dict[ParticleType, KelnerFitTable]
) -> float:
    """Evaluate the Kelner & Aharonian (2008) spectral function F(eta, x).

    Parameters
    ----------
    particle : ParticleType
        Secondary particle species.
    eta : float
        Dimensionless photon-proton interaction parameter,
        eta = 4 * epsilon * E_p / (m_p c^2)^2.
    x : float
        Fraction of the proton energy carried by the secondary particle,
        x = E_secondary / E_p.
    fit_tables : dict[ParticleType, KelnerFitTable]
        Preloaded fit tables, as returned by `load_fit_tables`.

    Returns
    -------
    float
        F(eta, x). Zero outside the kinematically allowed range of x.
    """
    table = fit_tables[particle]

    if particle == ParticleType.GAMMA:
        r = NEUTRAL_PION_MASS_GEV / PROTON_MASS_GEV
        eta0 = 2.0 * r + r * r
        # NOTE: the original reference implementation has no threshold guard
        # for this channel, so calling it with eta below the pion-production
        # threshold (eta0) makes xp/xm NaN, every subsequent comparison False,
        # and F never assigned -> UnboundLocalError. Using "unguarded" mode
        # here reproduces the same NaN bounds, but `_kelner_shape` initializes
        # F to zero rather than leaving it unset, so calling this below
        # threshold now returns 0.0 instead of crashing.
        x_upper, x_lower = _photopion_kinematic_bounds(eta, r, below_threshold="unguarded")
        s, delta, B = _interpolate_fit_coefficients(particle, eta, eta0, table)
        p = 2.5 + 0.4 * np.log(eta / eta0)
        return _kelner_shape(x, x_lower, x_upper, s, delta, B, p)

    if particle == ParticleType.POSITRON:
        r = CHARGED_PION_MASS_GEV / PROTON_MASS_GEV
        eta0 = 2.0 * r + r * r
        x_upper, x_lower = _photopion_kinematic_bounds(eta, r, below_threshold="zero")
        x_lower = x_lower / 4.0
        s, delta, B = _interpolate_fit_coefficients(particle, eta, eta0, table)
        p = 2.5 + 1.4 * np.log(eta / eta0)
        return _kelner_shape(x, x_lower, x_upper, s, delta, B, p)

    if particle in (ParticleType.NU_MU, ParticleType.ANTI_NU_MU, ParticleType.NU_E):
        r = CHARGED_PION_MASS_GEV / PROTON_MASS_GEV
        eta0 = 2.0 * r + r * r
        # NU_MU falls back to x_upper = x_lower = (eta + r^2) / (2*(1+eta))
        # below threshold; ANTI_NU_MU and NU_E fall back to (0.0, 0.0).
        below_threshold = "equal" if particle == ParticleType.NU_MU else "zero"
        x_upper, x_lower = _photopion_kinematic_bounds(eta, r, below_threshold=below_threshold)
        s, delta, B = _interpolate_fit_coefficients(particle, eta, eta0, table)
        p = 2.5 + 1.4 * np.log(eta / eta0)

        if particle == ParticleType.NU_MU:
            rho = eta / eta0
            if rho < 2.14:
                x_upper = 0.427 * x_upper
            elif rho < 10.0:
                x_upper = (0.427 + 0.0729 * (rho - 2.14)) * x_upper
            x_lower = 0.427 * x_lower
        else:
            x_lower = x_lower / 4.0

        return _kelner_shape(x, x_lower, x_upper, s, delta, B, p)

    if particle in (ParticleType.ELECTRON, ParticleType.ANTI_NU_E):
        r = CHARGED_PION_MASS_GEV / PROTON_MASS_GEV
        eta0 = 2.0 * r + r * r
        x_max, x_min = _electron_kinematic_bounds(eta, r)
        x_upper, x_lower = x_max, x_min / 2.0
        s, delta, B = _interpolate_fit_coefficients(particle, eta, eta0, table)

        rho = eta / eta0
        p = 6.0 * (1.0 - np.exp(1.5 * (4.0 - rho))) if rho >= 4.0 else 0.0

        F = _kelner_shape(x, x_lower, x_upper, s, delta, B, p)
        if rho < 2.14:
            F = np.zeros_like(np.asarray(x, dtype=float)) if np.ndim(x) else 0.0
        return F

    raise ValueError(f"Unknown particle type: {particle!r}")


def _photopion_kinematic_bounds(
    eta: float, r: float, below_threshold: str = "zero"
) -> tuple[float, float]:
    """Kinematic bounds (x_upper, x_lower) shared by gamma/positron/nu-mu/nu-e.

    Below the pion-production threshold (discriminant <= 0), the four
    particle channels behave differently in the original reference
    implementation:

    - GAMMA leaves the discriminant unguarded (matching the original, which
      has no threshold check for this channel) and will raise/propagate NaN,
      same as the original code would for out-of-domain (eta, x) — this is a
      known gap in the reference implementation, not something introduced
      here.
    - POSITRON and NU_E fall back to (0.0, 0.0) ("zero").
    - NU_MU falls back to (x, x) with x = (eta + r^2) / (2*(1 + eta)) ("equal").

    Parameters
    ----------
    eta : float
        Dimensionless photon-proton interaction parameter.
    r : float
        Secondary-to-proton mass ratio (m_pi / m_p) for the relevant pion.
    below_threshold : {"zero", "equal", "unguarded"}, default "zero"
        Fallback behavior below the kinematic threshold; see above.

    Returns
    -------
    x_upper, x_lower : float
        Upper and lower kinematic bounds on x = E_secondary / E_p (before any
        species-specific rescaling applied by the caller).
    """
    xt1 = 2.0 * (1.0 + eta)
    xt2 = eta + r * r
    discriminant = (eta - r * r - 2.0 * r) * (eta - r * r + 2.0 * r)

    if below_threshold == "unguarded":
        sqrt_term = np.sqrt(discriminant)
        return (xt2 + sqrt_term) / xt1, (xt2 - sqrt_term) / xt1

    if discriminant > _SMALL_D:
        sqrt_term = np.sqrt(discriminant)
        return (xt2 + sqrt_term) / xt1, (xt2 - sqrt_term) / xt1

    if below_threshold == "equal":
        return xt2 / xt1, xt2 / xt1
    return 0.0, 0.0


def _electron_kinematic_bounds(eta: float, r: float) -> tuple[float, float]:
    """Kinematic bounds (x_max, x_min) shared by the electron and anti-nu-e channels.

    Parameters
    ----------
    eta : float
        Dimensionless photon-proton interaction parameter.
    r : float
        Charged-pion-to-proton mass ratio.

    Returns
    -------
    x_max, x_min : float
        Upper and lower kinematic bounds on x = E_secondary / E_p.
    """
    xt1 = 2.0 * (1.0 + eta)
    xt2 = eta - 2.0 * r
    discriminant = eta * (eta - 4.0 * r * (1.0 + r))
    if discriminant > _SMALL_D:
        sqrt_term = np.sqrt(discriminant)
        return (xt2 + sqrt_term) / xt1, (xt2 - sqrt_term) / xt1
    return 0.0, 0.0
