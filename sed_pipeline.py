"""
SED integration pipeline for the photo-hadronic (p-gamma) secondary particle model.

Given a primary proton spectrum and a target photon field, this module computes
the resulting gamma-ray, electron+positron, and neutrino (all flavors) SEDs by:

1. **Integrate**: for each proton energy bin, integrate the Kelner & Aharonian
   `F(eta, x)` function (see `kelner_pgamma_model.py`) over the target photon
   field to get each secondary's spectrum on a fixed grid of
   x = E_secondary / E_proton.
2. **Remap**: convert that x-grid result into a fixed secondary-particle energy
   grid (shared across all proton energies), via a nearest-bin lookup.
3. **Integrate over the proton spectrum**: weight and sum each proton energy
   bin's contribution by the (user-supplied) proton injection spectrum, giving
   the final, proton-energy-integrated SEDs.

This follows the two-stage integration scheme of the original reference
implementation, restructured to avoid re-reading the Kelner & Aharonian fit
tables from disk on every innermost-loop evaluation (previously ~10-20 million
calls for a typical configuration) and to avoid an unnecessary disk
round-trip for the target photon field.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kelner_pgamma_model import (
    KELNER_KOEFFICIENT,
    NEUTRAL_PION_MASS_GEV,
    NEUTRINO_OSCILLATION_FACTOR,
    PROTON_MASS_GEV,
    ParticleType,
    KelnerFitTable,
    calculate_F,
    proton_spectrum,
)

GEV_PER_EV = 1.0e-9


@dataclass
class SimulationGridConfig:
    """Grid sizes and numerical settings for the SED integration pipeline.

    Attributes
    ----------
    n_proton_bins : int
        Number of logarithmically spaced proton energy bins.
    n_neutrino_bins, n_electron_bins, n_gamma_bins : int
        Number of logarithmically spaced bins in the final secondary-particle
        energy grids.
    minimal_x_fraction : float
        Smallest value of x = E_secondary / E_proton considered, defining the
        lower edge of the internal x-grid used in `integrate_single_proton_energy`.
    n_x_bins : int
        Number of logarithmically spaced bins in x, between
        `minimal_x_fraction` and 1.
    n_photon_field_bins : int
        Number of target-photon-field energy bins to integrate over per
        (proton energy, x) pair. Must not exceed the number of rows in the
        target photon field array actually supplied.
    """

    n_proton_bins: int = 50
    n_neutrino_bins: int = 5000
    n_electron_bins: int = 5000
    n_gamma_bins: int = 5000
    minimal_x_fraction: float = 1.0e-4
    n_x_bins: int = 500
    n_photon_field_bins: int = 100


@dataclass
class TargetPhotonField:
    """Tabulated target photon field for p-gamma interactions, in the blob frame.

    Attributes
    ----------
    energy_eV : np.ndarray
        Photon energies [eV].
    number_density : np.ndarray
        Differential photon number density [cm^-3 eV^-1], same shape as `energy_eV`.
    """

    energy_eV: np.ndarray
    number_density: np.ndarray

    @classmethod
    def from_file(cls, filepath: str) -> "TargetPhotonField":
        """Load a target photon field from a semicolon-delimited file (energy_eV; density)."""
        energy_eV, number_density = np.loadtxt(filepath, delimiter=";", unpack=True)
        return cls(energy_eV=energy_eV, number_density=number_density)

    def save(self, filepath: str) -> None:
        """Save the target photon field to a semicolon-delimited file."""
        np.savetxt(filepath, np.c_[self.energy_eV, self.number_density], delimiter=";", fmt="%.6e")


def _log_spaced_grid(min_value: float, max_value: float, n_bins: int) -> np.ndarray:
    """Build a logarithmically spaced grid of `n_bins` points from `min_value` to just below `max_value`.

    Matches the original grid convention: `min_value * 10**(k * a)` for
    `k = 0, ..., n_bins - 1`, with `a = log10(max_value / min_value) / n_bins`
    (so the grid spans `[min_value, max_value)`, not inclusive of the upper edge).
    """
    step = np.log10(max_value / min_value) / n_bins
    return min_value * np.power(10.0, step * np.arange(n_bins))


def integrate_single_proton_energy(
    proton_energy_eV: float,
    target_field: TargetPhotonField,
    grid: SimulationGridConfig,
    fit_tables: dict[ParticleType, KelnerFitTable],
) -> dict[str, np.ndarray]:
    """Integrate the secondary-particle spectra over the target photon field, for one proton energy.

    For a single proton energy, this evaluates, on an internal grid of
    x = E_secondary / E_proton, the photon-field-integrated production rate of
    gamma-rays, electrons + positrons, and neutrinos (all flavors).

    Parameters
    ----------
    proton_energy_eV : float
        Proton energy [eV].
    target_field : TargetPhotonField
        Target photon field for the p-gamma interaction (already boosted into
        the relevant frame by the caller, e.g. the emitting-region frame).
    grid : SimulationGridConfig
        Grid sizes and numerical settings.
    fit_tables : dict[ParticleType, KelnerFitTable]
        Preloaded Kelner & Aharonian fit tables (see `load_fit_tables`).

    Returns
    -------
    dict[str, np.ndarray]
        Arrays of shape `(grid.n_x_bins,)` under keys "gamma", "electron",
        and "neutrino" (electron key already sums the positron channel;
        neutrino key already sums all four neutrino/antineutrino channels).
    """
    a_frac = np.log10(1.0 / grid.minimal_x_fraction) / grid.n_x_bins

    proton_energy_GeV = proton_energy_eV / 1.0e9
    r = NEUTRAL_PION_MASS_GEV / PROTON_MASS_GEV
    eta0 = 2.0 * r + r * r
    # Photon energy threshold [eV] for pion production at this proton energy
    energy_threshold_eV = 1.0e9 * (eta0 * PROTON_MASS_GEV**2) / (4.0 * proton_energy_GeV)

    n_photons = min(grid.n_photon_field_bins, len(target_field.energy_eV))
    photon_energy = target_field.energy_eV[:n_photons]
    photon_density = target_field.number_density[:n_photons]

    sed_gamma = np.zeros(grid.n_x_bins)
    sed_electron = np.zeros(grid.n_x_bins)  # electron + positron
    sed_neutrino = np.zeros(grid.n_x_bins)  # all four (anti)neutrino channels

    for j in range(1, grid.n_x_bins):
        x = grid.minimal_x_fraction * 10.0 ** (a_frac * j)

        s_gamma = s_positron = s_electron = 0.0
        s_numu = s_antinumu = s_nue = s_antinue = 0.0

        for i in range(1, n_photons):
            if photon_energy[i] <= energy_threshold_eV:
                continue

            eps = photon_energy[i]
            f = photon_density[i]
            d_eps = photon_energy[i] - photon_energy[i - 1]

            # eta = 4 * eps[GeV] * E_p[GeV] / m_p^2: dimensionless p-gamma parameter
            eta = (4.0 * GEV_PER_EV * eps * proton_energy_GeV) / PROTON_MASS_GEV**2

            s_gamma += f * calculate_F(ParticleType.GAMMA, eta, x, fit_tables) * d_eps
            s_electron += f * calculate_F(ParticleType.ELECTRON, eta, x, fit_tables) * d_eps
            s_positron += f * calculate_F(ParticleType.POSITRON, eta, x, fit_tables) * d_eps
            s_numu += f * calculate_F(ParticleType.NU_MU, eta, x, fit_tables) * d_eps
            s_antinumu += f * calculate_F(ParticleType.ANTI_NU_MU, eta, x, fit_tables) * d_eps
            s_nue += f * calculate_F(ParticleType.NU_E, eta, x, fit_tables) * d_eps
            s_antinue += f * calculate_F(ParticleType.ANTI_NU_E, eta, x, fit_tables) * d_eps

        x_squared_ep = x * x * proton_energy_eV
        sed_electron[j] = x_squared_ep * (s_positron + s_electron) * KELNER_KOEFFICIENT
        sed_neutrino[j] = x_squared_ep * (s_numu + s_antinumu + s_nue + s_antinue) * KELNER_KOEFFICIENT
        sed_gamma[j] = x_squared_ep * s_gamma * KELNER_KOEFFICIENT

    return {"gamma": sed_gamma, "electron": sed_electron, "neutrino": sed_neutrino}


def _remap_to_final_grid(
    sed_vs_x: np.ndarray,
    proton_energy: np.ndarray,
    final_energy: np.ndarray,
    grid: SimulationGridConfig,
) -> np.ndarray:
    """Remap a per-proton-energy SED(x) array onto the shared final energy grid.

    For each (final energy, proton energy) pair, looks up the x-grid bin
    corresponding to `x = final_energy / proton_energy` and copies that bin's
    value across. This reproduces the original nearest-bin lookup exactly
    (including discarding points that fall outside the internal x-grid).

    Parameters
    ----------
    sed_vs_x : np.ndarray, shape (n_x_bins, n_proton_bins)
        Per-proton-energy SED(x), as returned by `integrate_single_proton_energy`
        (stacked column-wise, one column per proton energy bin).
    proton_energy : np.ndarray
        Proton energy grid [eV].
    final_energy : np.ndarray
        Target secondary-particle energy grid [eV] to remap onto.
    grid : SimulationGridConfig
        Grid sizes and numerical settings (for `minimal_x_fraction`/`n_x_bins`).

    Returns
    -------
    np.ndarray, shape (len(final_energy), len(proton_energy))
        Remapped SED, ready to be integrated over the proton spectrum.
    """
    a_frac = np.log10(1.0 / grid.minimal_x_fraction) / grid.n_x_bins
    n_proton = len(proton_energy)
    remapped = np.zeros((len(final_energy), n_proton))

    for j in range(n_proton):
        x = final_energy / proton_energy[j]
        with np.errstate(divide="ignore"):
            bin_index = (np.log10(x / grid.minimal_x_fraction) / a_frac).astype(int)
        valid = (bin_index > 0) & (bin_index < grid.n_x_bins)
        remapped[valid, j] = sed_vs_x[bin_index[valid], j]

    return remapped


def _integrate_over_proton_spectrum(
    remapped_sed: np.ndarray,
    proton_energy: np.ndarray,
    spectral_index: float,
    cutoff_energy_eV: float | None,
) -> np.ndarray:
    """Weight and sum a remapped SED over the proton injection spectrum.

    Parameters
    ----------
    remapped_sed : np.ndarray, shape (n_final_energy, n_proton_bins)
        Output of `_remap_to_final_grid`.
    proton_energy : np.ndarray
        Proton energy grid [eV].
    spectral_index : float
        Proton spectral index (see `proton_spectrum`).
    cutoff_energy_eV : float or None
        Proton spectrum exponential cutoff energy [eV] (see `proton_spectrum`).

    Returns
    -------
    np.ndarray, shape (n_final_energy,)
        Proton-spectrum-weighted, energy-integrated secondary SED.
    """
    weights = proton_spectrum(proton_energy, spectral_index, cutoff_energy_eV)
    bin_widths = np.diff(proton_energy, prepend=proton_energy[0])
    # bin j=0 contributes zero width, matching the original loop starting at j=1
    bin_widths[0] = 0.0
    return remapped_sed[:, 1:] @ (weights[1:] * bin_widths[1:])


@dataclass
class PhotohadronicSEDResult:
    """Final, proton-spectrum-integrated secondary particle SEDs.

    Attributes
    ----------
    gamma_energy_eV, gamma_sed : np.ndarray
        Gamma-ray energy grid [eV] and E^2 dN/dE dt-like SED [eV/s] (same
        normalization as the original implementation's output files).
    electron_energy_eV, electron_sed : np.ndarray
        Electron + positron energy grid and SED.
    neutrino_energy_eV, neutrino_sed : np.ndarray
        All-flavor neutrino + antineutrino energy grid and SED, already
        multiplied by `NEUTRINO_OSCILLATION_FACTOR`.
    """

    gamma_energy_eV: np.ndarray
    gamma_sed: np.ndarray
    electron_energy_eV: np.ndarray
    electron_sed: np.ndarray
    neutrino_energy_eV: np.ndarray
    neutrino_sed: np.ndarray

    def save(self, output_dir: str = ".") -> None:
        """Save the three SEDs to disk, matching the original file layout."""
        output_dir = Path(output_dir)
        np.savetxt(output_dir / "gamma_SED.txt", np.c_[self.gamma_energy_eV, self.gamma_sed])
        np.savetxt(output_dir / "electron_SED.txt", np.c_[self.electron_energy_eV, self.electron_sed])
        np.savetxt(output_dir / "neutrino_SED.txt", np.c_[self.neutrino_energy_eV, self.neutrino_sed])


def compute_photohadronic_seds(
    proton_energy_min_eV: float,
    proton_energy_max_eV: float,
    spectral_index: float,
    cutoff_energy_eV: float | None,
    target_field: TargetPhotonField,
    fit_tables: dict[ParticleType, KelnerFitTable],
    grid: SimulationGridConfig = SimulationGridConfig(),
    verbose: bool = True,
) -> PhotohadronicSEDResult:
    """Compute the gamma-ray, electron/positron, and neutrino SEDs from p-gamma interactions.

    This is the main entry point of the pipeline: it builds the proton and
    secondary-particle energy grids, integrates the Kelner & Aharonian
    `F(eta, x)` function over the target photon field for each proton energy
    bin (`integrate_single_proton_energy`), remaps the result onto shared
    output energy grids (`_remap_to_final_grid`), and integrates over the
    proton injection spectrum (`_integrate_over_proton_spectrum`).

    Parameters
    ----------
    proton_energy_min_eV, proton_energy_max_eV : float
        Bounds of the proton energy grid [eV].
    spectral_index : float
        Proton injection spectral index.
    cutoff_energy_eV : float or None
        Proton spectrum exponential cutoff energy [eV], or None for a pure
        power law.
    target_field : TargetPhotonField
        Target photon field for the p-gamma interaction, in the frame where
        the proton spectrum is defined (typically the emitting-region frame).
    fit_tables : dict[ParticleType, KelnerFitTable]
        Preloaded Kelner & Aharonian fit tables (see `load_fit_tables`).
    grid : SimulationGridConfig, optional
        Grid sizes and numerical settings. Defaults match the original
        reference implementation.
    verbose : bool, default True
        Print progress (one line per proton energy bin), matching the
        original implementation's behavior.

    Returns
    -------
    PhotohadronicSEDResult
        The final gamma-ray, electron+positron, and neutrino SEDs.
    """
    proton_energy = _log_spaced_grid(proton_energy_min_eV, proton_energy_max_eV, grid.n_proton_bins)

    # Secondary-particle output grids share the same upper bound convention
    # as the original implementation: 1e12 eV to 0.75 * E_p,max.
    neutrino_energy = _log_spaced_grid(1.0e12, proton_energy_max_eV * 0.75, grid.n_neutrino_bins)
    electron_energy = _log_spaced_grid(1.0e12, proton_energy_max_eV * 0.75, grid.n_electron_bins)
    gamma_energy = _log_spaced_grid(1.0e12, proton_energy_max_eV * 0.75, grid.n_gamma_bins)

    sed_gamma_vs_x = np.zeros((grid.n_x_bins, grid.n_proton_bins))
    sed_electron_vs_x = np.zeros((grid.n_x_bins, grid.n_proton_bins))
    sed_neutrino_vs_x = np.zeros((grid.n_x_bins, grid.n_proton_bins))

    for k in range(grid.n_proton_bins):
        seds_at_x = integrate_single_proton_energy(proton_energy[k], target_field, grid, fit_tables)
        sed_gamma_vs_x[:, k] = seds_at_x["gamma"]
        sed_electron_vs_x[:, k] = seds_at_x["electron"]
        sed_neutrino_vs_x[:, k] = seds_at_x["neutrino"]
        if verbose:
            print(f"Integrating proton energy bin {k + 1}/{grid.n_proton_bins}")

    gamma_remapped = _remap_to_final_grid(sed_gamma_vs_x, proton_energy, gamma_energy, grid)
    electron_remapped = _remap_to_final_grid(sed_electron_vs_x, proton_energy, electron_energy, grid)
    neutrino_remapped = _remap_to_final_grid(sed_neutrino_vs_x, proton_energy, neutrino_energy, grid)

    gamma_sed = _integrate_over_proton_spectrum(gamma_remapped, proton_energy, spectral_index, cutoff_energy_eV)
    electron_sed = _integrate_over_proton_spectrum(electron_remapped, proton_energy, spectral_index, cutoff_energy_eV)
    neutrino_sed = (
        _integrate_over_proton_spectrum(neutrino_remapped, proton_energy, spectral_index, cutoff_energy_eV)
        * NEUTRINO_OSCILLATION_FACTOR
    )

    return PhotohadronicSEDResult(
        gamma_energy_eV=gamma_energy,
        gamma_sed=gamma_sed,
        electron_energy_eV=electron_energy,
        electron_sed=electron_sed,
        neutrino_energy_eV=neutrino_energy,
        neutrino_sed=neutrino_sed,
    )
